from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from tinyexp.exceptions import InvalidWorkerCountError
from tinyexp.utils.ray_utils import (
    _build_worker_env_vars,
    _launch_with_ray,
    _maybe_start_ray_redis_cache,
    _should_print_launcher,
    get_launcher,
    get_placement_group,
)


def test_get_launcher_defaults_to_python() -> None:
    assert get_launcher() == "python"


def test_get_launcher_detects_torchrun_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")

    assert get_launcher() == "torchrun"


def test_get_launcher_detects_torchelastic_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "01863564-2461-4a49-9c96-0984a091986f")

    assert get_launcher() == "torchrun"


def test_get_launcher_ignores_empty_torchelastic_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "none")

    assert get_launcher() == "python"


def test_get_launcher_detects_accelerate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACCELERATE_PROCESS_INDEX", "0")

    assert get_launcher() == "accelerate"


def test_get_launcher_prefers_accelerate_env_over_rank_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACCELERATE_MIXED_PRECISION", "no")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")

    assert get_launcher() == "accelerate"


def test_get_launcher_ignores_torchrun_in_hydra_override(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeProcess:
        pid = 123

        def cmdline(self):
            return ["python", "tinyexp/examples/resnet_exp.py", "exp_name=resnet_run_with_redis_torchrun_simple"]

        def name(self):
            return "python"

        def parent(self):
            return None

    monkeypatch.setattr("tinyexp.utils.ray_utils.psutil.Process", lambda pid: FakeProcess())

    assert get_launcher() == "python"


def test_get_launcher_detects_torchrun_executable(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeProcess:
        pid = 123

        def cmdline(self):
            return ["torchrun", "--standalone", "tinyexp/examples/resnet_exp.py"]

        def name(self):
            return "torchrun"

        def parent(self):
            return None

    monkeypatch.setattr("tinyexp.utils.ray_utils.psutil.Process", lambda pid: FakeProcess())

    assert get_launcher() == "torchrun"


def test_launch_with_ray_rejects_invalid_ray_worker_count_without_starting_ray(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = OmegaConf.create({"ray_num_worker": 0})
    monkeypatch.setattr("tinyexp.utils.ray_utils.ray.init", lambda: pytest.fail("ray.init should not be called"))

    with pytest.raises(InvalidWorkerCountError, match="Number of workers"):
        _launch_with_ray(cfg, object)


def test_launch_with_ray_resolves_auto_worker_count_from_cluster_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = OmegaConf.create({"ray_num_worker": -1})

    def stop_after_resolution(exp_class):  # type: ignore[no-untyped-def]
        raise RuntimeError("resolved")

    monkeypatch.setattr("tinyexp.utils.ray_utils.ray.init", lambda: None)
    monkeypatch.setattr("tinyexp.utils.ray_utils.ray.remote", stop_after_resolution)
    monkeypatch.setattr("tinyexp.utils.ray_utils.ray.cluster_resources", lambda: {"GPU": 2.0})
    monkeypatch.setattr("tinyexp.utils.ray_utils.ray.is_initialized", lambda: True)
    monkeypatch.setattr("tinyexp.utils.ray_utils.ray.shutdown", lambda: None)

    with pytest.raises(RuntimeError, match="resolved"):
        _launch_with_ray(cfg, object)

    assert cfg.ray_num_worker == 2


def test_launch_with_ray_rejects_missing_cluster_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = OmegaConf.create({"ray_num_worker": -1})
    monkeypatch.setattr("tinyexp.utils.ray_utils.ray.init", lambda: None)
    monkeypatch.setattr("tinyexp.utils.ray_utils.ray.cluster_resources", lambda: {"CPU": 8.0})
    monkeypatch.setattr("tinyexp.utils.ray_utils.ray.is_initialized", lambda: True)
    monkeypatch.setattr("tinyexp.utils.ray_utils.ray.shutdown", lambda: None)

    with pytest.raises(InvalidWorkerCountError, match="Number of workers"):
        _launch_with_ray(cfg, object)


def test_should_print_launcher_based_on_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RANK", raising=False)
    assert _should_print_launcher() is True

    monkeypatch.setenv("RANK", "1")
    assert _should_print_launcher() is False


def test_build_worker_env_vars_prefers_user_defined_ifname(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "eth9")
    env_vars = _build_worker_env_vars(num_worker=2, rank=1, local_rank=1, master_addr="127.0.0.1", master_port=12345)
    assert env_vars["GLOO_SOCKET_IFNAME"] == "eth9"


def test_build_worker_env_vars_omits_ifname_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GLOO_SOCKET_IFNAME", raising=False)
    env_vars = _build_worker_env_vars(num_worker=2, rank=1, local_rank=1, master_addr="127.0.0.1", master_port=12345)
    assert "GLOO_SOCKET_IFNAME" not in env_vars


def test_get_placement_group_defaults_to_pack(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    class FakePlacementGroup:
        def ready(self):
            return "ready"

    def fake_placement_group(bundles, strategy):
        captured["bundles"] = bundles
        captured["strategy"] = strategy
        return FakePlacementGroup()

    monkeypatch.setattr("tinyexp.utils.ray_utils.placement_group", fake_placement_group)
    monkeypatch.setattr("tinyexp.utils.ray_utils.ray.get", lambda ref: ref)
    monkeypatch.setattr("tinyexp.utils.ray_utils.ray.is_initialized", lambda: False)

    get_placement_group(num_worker=2, num_gpus_per_worker=0, num_cpus_per_worker=3)

    assert captured == {
        "bundles": [{"CPU": 3, "GPU": 0}, {"CPU": 3, "GPU": 0}],
        "strategy": "PACK",
    }


def test_get_placement_group_pins_first_bundle_to_head_when_ray_initialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    class FakePlacementGroup:
        def ready(self):
            return "ready"

    def fake_placement_group(bundles, strategy):
        captured["bundles"] = bundles
        captured["strategy"] = strategy
        return FakePlacementGroup()

    monkeypatch.setattr("tinyexp.utils.ray_utils.placement_group", fake_placement_group)
    monkeypatch.setattr("tinyexp.utils.ray_utils.ray.get", lambda ref: ref)
    monkeypatch.setattr("tinyexp.utils.ray_utils.ray.is_initialized", lambda: True)
    monkeypatch.setattr(
        "tinyexp.utils.ray_utils.ray.cluster_resources",
        lambda: {"node:__internal_head__": 1.0},
    )

    get_placement_group(num_worker=2, num_gpus_per_worker=1, num_cpus_per_worker=3)

    assert captured == {
        "bundles": [
            {"CPU": 3, "GPU": 1, "node:__internal_head__": 0.001},
            {"CPU": 3, "GPU": 1},
        ],
        "strategy": "PACK",
    }


def test_get_placement_group_accepts_explicit_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    class FakePlacementGroup:
        def ready(self):
            return "ready"

    def fake_placement_group(bundles, strategy):
        captured["strategy"] = strategy
        return FakePlacementGroup()

    monkeypatch.setattr("tinyexp.utils.ray_utils.placement_group", fake_placement_group)
    monkeypatch.setattr("tinyexp.utils.ray_utils.ray.get", lambda ref: ref)

    get_placement_group(num_worker=2, strategy="SPREAD")

    assert captured["strategy"] == "SPREAD"


def test_maybe_start_ray_redis_cache_returns_none_without_redis_cfg() -> None:
    assert _maybe_start_ray_redis_cache(OmegaConf.create({})) is None


def test_maybe_start_ray_redis_cache_returns_none_when_disabled() -> None:
    cfg = OmegaConf.create({"redis_cache_cfg": {"redis_cache_enabled": False}})
    assert _maybe_start_ray_redis_cache(cfg) is None


def test_maybe_start_ray_redis_cache_starts_standalone_for_world_size_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = []
    captured = {}

    class FakeRayRedisClusterManager:
        def __init__(self, redis_cache_cfg) -> None:  # type: ignore[no-untyped-def]
            self.redis_cache_cfg = redis_cache_cfg
            created.append(self)

        def start(self, *, cluster_enabled: bool) -> tuple[str, list[int], int]:
            captured["cluster_enabled"] = cluster_enabled
            return "10.0.0.1", [7000], 1

    monkeypatch.setattr("tinyexp.utils.ray_utils.RayRedisClusterManager", FakeRayRedisClusterManager)

    cfg = OmegaConf.create(
        {
            "redis_cache_cfg": {
                "redis_cache_enabled": True,
                "redis_cluster_host": "127.0.0.1",
                "redis_cluster_ports": [7000],
                "redis_rendezvous_world_size": 1,
            }
        }
    )

    manager = _maybe_start_ray_redis_cache(cfg)

    assert manager is created[0]
    assert captured["cluster_enabled"] is False
    assert cfg.redis_cache_cfg.redis_cluster_host == "10.0.0.1"
    assert list(cfg.redis_cache_cfg.redis_cluster_ports) == [7000]
    assert cfg.redis_cache_cfg.redis_rendezvous_world_size == 1


def test_maybe_start_ray_redis_cache_returns_none_for_external_cluster_cfg() -> None:
    cfg = OmegaConf.create(
        {
            "redis_cache_cfg": {
                "redis_cache_enabled": True,
                "redis_cluster_host": "10.0.0.1",
                "redis_cluster_ports": [7000, 7001, 7002],
                "redis_rendezvous_world_size": 2,
            }
        }
    )

    assert _maybe_start_ray_redis_cache(cfg) is None


def test_maybe_start_ray_redis_cache_rejects_invalid_world_size() -> None:
    cfg = OmegaConf.create(
        {
            "redis_cache_cfg": {
                "redis_cache_enabled": True,
                "redis_cluster_ports": [7000],
                "redis_rendezvous_world_size": 0,
            }
        }
    )

    with pytest.raises(ValueError, match="redis_rendezvous_world_size"):
        _maybe_start_ray_redis_cache(cfg)


def test_maybe_start_ray_redis_cache_writes_resolved_cfg_for_auto_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = []

    class FakeRayRedisClusterManager:
        def __init__(self, redis_cache_cfg) -> None:  # type: ignore[no-untyped-def]
            self.redis_cache_cfg = redis_cache_cfg
            created.append(self)

        def start(self, *, cluster_enabled: bool) -> tuple[str, list[int], int]:
            assert cluster_enabled is True
            return "10.0.0.1", [7000, 7001, 7002], 2

    monkeypatch.setattr("tinyexp.utils.ray_utils.RayRedisClusterManager", FakeRayRedisClusterManager)

    cfg = OmegaConf.create(
        {
            "redis_cache_cfg": {
                "redis_cache_enabled": True,
                "redis_cluster_host": "127.0.0.1",
                "redis_cluster_ports": [7000],
                "redis_rendezvous_world_size": -1,
            }
        }
    )

    manager = _maybe_start_ray_redis_cache(cfg)

    assert manager is created[0]
    assert cfg.redis_cache_cfg.redis_cluster_host == "10.0.0.1"
    assert list(cfg.redis_cache_cfg.redis_cluster_ports) == [7000, 7001, 7002]
    assert cfg.redis_cache_cfg.redis_rendezvous_world_size == 2
