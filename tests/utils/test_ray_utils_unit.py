from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from tinyexp.exceptions import InvalidWorkerCountError
from tinyexp.exp_mixins import RayCfgMixin
from tinyexp.utils.ray_utils import (
    _build_worker_env_vars,
    _maybe_start_ray_redis_cache,
    get_placement_group,
)


def test_ray_cfg_run_uses_explicit_resources_without_dataloader_cfg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = OmegaConf.create(
        {
            "ray_cfg": {
                "ray_num_worker": 1,
                "ray_num_cpus_per_worker": 3,
                "ray_num_gpus_per_worker": 0,
                "ray_placement_strategy": "PACK",
            }
        }
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr("tinyexp.exp_mixins.basic_mixins.ray.init", lambda: None)
    monkeypatch.setattr("tinyexp.exp_mixins.basic_mixins.ray.remote", lambda exp_class: object())
    monkeypatch.setattr("tinyexp.exp_mixins.basic_mixins.ray.cluster_resources", lambda: {"CPU": 3.0, "GPU": 0.0})
    monkeypatch.setattr("tinyexp.exp_mixins.basic_mixins.ray.is_initialized", lambda: True)
    monkeypatch.setattr("tinyexp.exp_mixins.basic_mixins.ray.shutdown", lambda **kwargs: None)

    def stop_after_resource_resolution(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        raise RuntimeError("resource resolution complete")  # noqa: TRY003

    monkeypatch.setattr("tinyexp.exp_mixins.basic_mixins.get_placement_group", stop_after_resource_resolution)

    with pytest.raises(RuntimeError, match="resource resolution complete"):
        RayCfgMixin.RayCfg.run(object, cfg)

    assert captured["num_worker"] == 1
    assert captured["num_cpus_per_worker"] == 3
    assert captured["num_gpus_per_worker"] == 0


def test_ray_cfg_run_rejects_invalid_ray_worker_count_without_starting_ray(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = OmegaConf.create(
        {"ray_cfg": {"ray_num_worker": 0, "ray_num_cpus_per_worker": 1, "ray_num_gpus_per_worker": 1}}
    )
    monkeypatch.setattr(
        "tinyexp.exp_mixins.basic_mixins.ray.init",
        lambda: pytest.fail("ray.init should not be called"),
    )

    with pytest.raises(InvalidWorkerCountError, match="Number of workers"):
        RayCfgMixin.RayCfg.run(object, cfg)


def test_ray_cfg_run_resolves_auto_worker_count_from_cluster_gpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = OmegaConf.create(
        {"ray_cfg": {"ray_num_worker": -1, "ray_num_cpus_per_worker": 1, "ray_num_gpus_per_worker": 1}}
    )

    def stop_after_resolution(exp_class):  # type: ignore[no-untyped-def]
        raise RuntimeError("resolved")

    monkeypatch.setattr("tinyexp.exp_mixins.basic_mixins.ray.init", lambda: None)
    monkeypatch.setattr("tinyexp.exp_mixins.basic_mixins.ray.remote", stop_after_resolution)
    monkeypatch.setattr("tinyexp.exp_mixins.basic_mixins.ray.cluster_resources", lambda: {"CPU": 8.0, "GPU": 2.0})
    monkeypatch.setattr("tinyexp.exp_mixins.basic_mixins.ray.is_initialized", lambda: True)
    monkeypatch.setattr("tinyexp.exp_mixins.basic_mixins.ray.shutdown", lambda **kwargs: None)

    with pytest.raises(RuntimeError, match="resolved"):
        RayCfgMixin.RayCfg.run(object, cfg)

    assert cfg.ray_cfg.ray_num_worker == 2


def test_ray_cfg_run_caps_auto_worker_count_by_cpu_and_gpu_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = OmegaConf.create(
        {
            "ray_cfg": {
                "ray_num_worker": -1,
                "ray_num_cpus_per_worker": 3,
                "ray_num_gpus_per_worker": 1,
            }
        }
    )

    def stop_after_resolution(exp_class):  # type: ignore[no-untyped-def]
        raise RuntimeError("resolved")

    monkeypatch.setattr("tinyexp.exp_mixins.basic_mixins.ray.init", lambda: None)
    monkeypatch.setattr("tinyexp.exp_mixins.basic_mixins.ray.remote", stop_after_resolution)
    monkeypatch.setattr("tinyexp.exp_mixins.basic_mixins.ray.cluster_resources", lambda: {"CPU": 8.0, "GPU": 4.0})
    monkeypatch.setattr("tinyexp.exp_mixins.basic_mixins.ray.is_initialized", lambda: True)
    monkeypatch.setattr("tinyexp.exp_mixins.basic_mixins.ray.shutdown", lambda **kwargs: None)

    with pytest.raises(RuntimeError, match="resolved"):
        RayCfgMixin.RayCfg.run(object, cfg)

    assert cfg.ray_cfg.ray_num_worker == 2


def test_ray_cfg_run_resolves_auto_worker_count_from_cluster_cpus_for_cpu_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = OmegaConf.create(
        {
            "ray_cfg": {
                "ray_num_worker": -1,
                "ray_num_cpus_per_worker": 2,
                "ray_num_gpus_per_worker": 0,
            }
        }
    )

    def stop_after_resolution(exp_class):  # type: ignore[no-untyped-def]
        raise RuntimeError("resolved")

    monkeypatch.setattr("tinyexp.exp_mixins.basic_mixins.ray.init", lambda: None)
    monkeypatch.setattr("tinyexp.exp_mixins.basic_mixins.ray.remote", stop_after_resolution)
    monkeypatch.setattr("tinyexp.exp_mixins.basic_mixins.ray.cluster_resources", lambda: {"CPU": 8.0, "GPU": 0.0})
    monkeypatch.setattr("tinyexp.exp_mixins.basic_mixins.ray.is_initialized", lambda: True)
    monkeypatch.setattr("tinyexp.exp_mixins.basic_mixins.ray.shutdown", lambda **kwargs: None)

    with pytest.raises(RuntimeError, match="resolved"):
        RayCfgMixin.RayCfg.run(object, cfg)

    assert cfg.ray_cfg.ray_num_worker == 4


@pytest.mark.parametrize(
    ("field_name", "value"),
    [("ray_num_cpus_per_worker", 0), ("ray_num_gpus_per_worker", -0.1)],
)
def test_ray_cfg_run_rejects_invalid_worker_resources_before_ray_init(
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
    value: float,
) -> None:
    cfg = OmegaConf.create(
        {
            "ray_cfg": {
                "ray_num_worker": 1,
                "ray_num_cpus_per_worker": 1,
                "ray_num_gpus_per_worker": 0,
                field_name: value,
            }
        }
    )
    monkeypatch.setattr(
        "tinyexp.exp_mixins.basic_mixins.ray.init",
        lambda: pytest.fail("ray.init should not be called"),
    )

    with pytest.raises(ValueError):
        RayCfgMixin.RayCfg.run(object, cfg)


def test_ray_cfg_run_rejects_missing_cluster_gpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = OmegaConf.create(
        {"ray_cfg": {"ray_num_worker": -1, "ray_num_cpus_per_worker": 1, "ray_num_gpus_per_worker": 1}}
    )
    monkeypatch.setattr("tinyexp.exp_mixins.basic_mixins.ray.init", lambda: None)
    monkeypatch.setattr("tinyexp.exp_mixins.basic_mixins.ray.cluster_resources", lambda: {"CPU": 8.0})
    monkeypatch.setattr("tinyexp.exp_mixins.basic_mixins.ray.is_initialized", lambda: True)
    monkeypatch.setattr("tinyexp.exp_mixins.basic_mixins.ray.shutdown", lambda **kwargs: None)

    with pytest.raises(InvalidWorkerCountError, match="Number of workers"):
        RayCfgMixin.RayCfg.run(object, cfg)


def test_custom_ray_cfg_run_receives_class_and_global_cfg() -> None:
    calls = []

    class CustomRayCfg(RayCfgMixin.RayCfg):
        @classmethod
        def run(cls, exp_class, experiment_cfg):
            calls.append((cls, exp_class, experiment_cfg))

    cfg = OmegaConf.create({"ray_cfg": {"ray_num_worker": 1}, "mode": "run"})
    CustomRayCfg.run(object, cfg)

    assert calls == [(CustomRayCfg, object, cfg)]


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
    cfg = OmegaConf.create({"redis_cfg": {"redis_cache_enabled": False}})
    assert _maybe_start_ray_redis_cache(cfg) is None


def test_maybe_start_ray_redis_cache_starts_standalone_for_world_size_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = []
    captured = {}

    class FakeRayRedisClusterManager:
        def __init__(self, redis_cfg) -> None:  # type: ignore[no-untyped-def]
            self.redis_cfg = redis_cfg
            created.append(self)

        def start(self, *, cluster_enabled: bool) -> tuple[str, list[int], int]:
            captured["cluster_enabled"] = cluster_enabled
            return "10.0.0.1", [7000], 1

    monkeypatch.setattr("tinyexp.utils.ray_utils.RayRedisClusterManager", FakeRayRedisClusterManager)

    cfg = OmegaConf.create(
        {
            "redis_cfg": {
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
    assert cfg.redis_cfg.redis_cluster_host == "10.0.0.1"
    assert list(cfg.redis_cfg.redis_cluster_ports) == [7000]
    assert cfg.redis_cfg.redis_rendezvous_world_size == 1


def test_maybe_start_ray_redis_cache_returns_none_for_external_cluster_cfg() -> None:
    cfg = OmegaConf.create(
        {
            "redis_cfg": {
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
            "redis_cfg": {
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
        def __init__(self, redis_cfg) -> None:  # type: ignore[no-untyped-def]
            self.redis_cfg = redis_cfg
            created.append(self)

        def start(self, *, cluster_enabled: bool) -> tuple[str, list[int], int]:
            assert cluster_enabled is True
            return "10.0.0.1", [7000, 7001, 7002], 2

    monkeypatch.setattr("tinyexp.utils.ray_utils.RayRedisClusterManager", FakeRayRedisClusterManager)

    cfg = OmegaConf.create(
        {
            "redis_cfg": {
                "redis_cache_enabled": True,
                "redis_cluster_host": "127.0.0.1",
                "redis_cluster_ports": [7000],
                "redis_rendezvous_world_size": -1,
            }
        }
    )

    manager = _maybe_start_ray_redis_cache(cfg)

    assert manager is created[0]
    assert cfg.redis_cfg.redis_cluster_host == "10.0.0.1"
    assert list(cfg.redis_cfg.redis_cluster_ports) == [7000, 7001, 7002]
    assert cfg.redis_cfg.redis_rendezvous_world_size == 2
