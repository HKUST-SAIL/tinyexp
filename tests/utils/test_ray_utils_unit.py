from __future__ import annotations

from types import SimpleNamespace

import pytest
import ray
from omegaconf import OmegaConf

from tinyexp.exceptions import InsufficientCPUError, InvalidWorkerCountError
from tinyexp.exp_mixins import RayCfgMixin
from tinyexp.utils.ray_utils import (
    _build_worker_env_vars,
    _maybe_start_ray_redis_cache,
    build_ray_worker_env_vars,
    get_network_config,
    get_num_worker_options,
    get_placement_group,
    get_placement_group_node_ids,
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
                "ray_placement_timeout_s": 7,
            }
        }
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr("tinyexp.exp_mixins.basic_mixins.ray.init", lambda: None)
    monkeypatch.setattr("tinyexp.exp_mixins.basic_mixins.ray.remote", lambda exp_class: object())
    monkeypatch.setattr(
        "tinyexp.exp_mixins.basic_mixins.ray.cluster_resources",
        lambda: {"CPU": 3.0, "GPU": 0.0},
    )
    monkeypatch.setattr("tinyexp.exp_mixins.basic_mixins.ray.is_initialized", lambda: True)
    monkeypatch.setattr("tinyexp.exp_mixins.basic_mixins.ray.shutdown", lambda **kwargs: None)

    def stop_after_resource_resolution(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        raise RuntimeError("resource resolution complete")  # noqa: TRY003

    monkeypatch.setattr(
        "tinyexp.exp_mixins.basic_mixins.get_placement_group",
        stop_after_resource_resolution,
    )

    with pytest.raises(RuntimeError, match="resource resolution complete"):
        RayCfgMixin.RayCfg.run(object, cfg)

    assert captured["num_worker"] == 1
    assert captured["num_cpus_per_worker"] == 3
    assert captured["num_gpus_per_worker"] == 0
    assert captured["timeout_s"] == 7.0


def test_ray_cfg_run_rejects_invalid_ray_worker_count_without_starting_ray(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = OmegaConf.create(
        {
            "ray_cfg": {
                "ray_num_worker": 0,
                "ray_num_cpus_per_worker": 1,
                "ray_num_gpus_per_worker": 1,
            }
        }
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
        {
            "ray_cfg": {
                "ray_num_worker": -1,
                "ray_num_cpus_per_worker": 1,
                "ray_num_gpus_per_worker": 1,
            }
        }
    )

    def stop_after_resolution(exp_class):  # type: ignore[no-untyped-def]
        raise RuntimeError("resolved")

    monkeypatch.setattr("tinyexp.exp_mixins.basic_mixins.ray.init", lambda: None)
    monkeypatch.setattr("tinyexp.exp_mixins.basic_mixins.ray.remote", stop_after_resolution)
    monkeypatch.setattr(
        "tinyexp.exp_mixins.basic_mixins.ray.cluster_resources",
        lambda: {"CPU": 8.0, "GPU": 2.0},
    )
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
    monkeypatch.setattr(
        "tinyexp.exp_mixins.basic_mixins.ray.cluster_resources",
        lambda: {"CPU": 8.0, "GPU": 4.0},
    )
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
    monkeypatch.setattr(
        "tinyexp.exp_mixins.basic_mixins.ray.cluster_resources",
        lambda: {"CPU": 8.0, "GPU": 0.0},
    )
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


def test_ray_cfg_run_rejects_explicit_gpu_shortage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = OmegaConf.create(
        {
            "ray_cfg": {
                "ray_num_worker": 2,
                "ray_num_cpus_per_worker": 1,
                "ray_num_gpus_per_worker": 1,
                "ray_placement_strategy": "PACK",
            }
        }
    )
    monkeypatch.setattr("tinyexp.exp_mixins.basic_mixins.ray.init", lambda: None)
    monkeypatch.setattr(
        "tinyexp.exp_mixins.basic_mixins.ray.cluster_resources",
        lambda: {"CPU": 8.0, "GPU": 1.0},
    )
    monkeypatch.setattr("tinyexp.exp_mixins.basic_mixins.ray.is_initialized", lambda: True)
    monkeypatch.setattr("tinyexp.exp_mixins.basic_mixins.ray.shutdown", lambda **kwargs: None)
    monkeypatch.setattr(
        "tinyexp.exp_mixins.basic_mixins.get_placement_group",
        lambda **kwargs: pytest.fail("placement group should not be created"),
    )

    with pytest.raises(InsufficientCPUError, match="needed GPU=2.0, available GPU=1.0"):
        RayCfgMixin.RayCfg.run(object, cfg)


def test_ray_cfg_run_rejects_non_positive_placement_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = OmegaConf.create(
        {
            "ray_cfg": {
                "ray_num_worker": 1,
                "ray_num_cpus_per_worker": 1,
                "ray_num_gpus_per_worker": 0,
                "ray_placement_timeout_s": 0,
            }
        }
    )
    monkeypatch.setattr(
        "tinyexp.exp_mixins.basic_mixins.ray.init",
        lambda: pytest.fail("ray.init should not be called"),
    )

    with pytest.raises(ValueError, match="ray_placement_timeout_s"):
        RayCfgMixin.RayCfg.run(object, cfg)


def test_ray_cfg_run_rejects_missing_cluster_gpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = OmegaConf.create(
        {
            "ray_cfg": {
                "ray_num_worker": -1,
                "ray_num_cpus_per_worker": 1,
                "ray_num_gpus_per_worker": 1,
            }
        }
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


def test_build_ray_worker_env_vars_uses_actual_node_local_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GLOO_SOCKET_IFNAME", raising=False)

    env_vars = build_ray_worker_env_vars(
        num_worker=4,
        node_ids=["node-a", "node-a", "node-b", "node-b"],
        master_addr="10.0.0.1",
        master_port=12345,
    )

    assert [set(env) for env in env_vars] == [
        {
            "WORLD_SIZE",
            "RANK",
            "MASTER_ADDR",
            "MASTER_PORT",
            "LOCAL_RANK",
        }
    ] * 4
    assert [(env["RANK"], env["LOCAL_RANK"]) for env in env_vars] == [
        ("0", "0"),
        ("1", "1"),
        ("2", "0"),
        ("3", "1"),
    ]


def test_build_ray_worker_env_vars_groups_global_ranks_by_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GLOO_SOCKET_IFNAME", raising=False)

    env_vars = build_ray_worker_env_vars(
        num_worker=4,
        node_ids=["node-a", "node-b", "node-a", "node-b"],
        master_addr="10.0.0.1",
        master_port=12345,
    )

    assert [(env["RANK"], env["LOCAL_RANK"]) for env in env_vars] == [
        ("0", "0"),
        ("2", "0"),
        ("1", "1"),
        ("3", "1"),
    ]


def test_build_ray_worker_env_vars_rejects_heterogeneous_worker_counts() -> None:
    with pytest.raises(ValueError, match="homogeneous.*node-a=2.*node-b=1"):
        build_ray_worker_env_vars(
            num_worker=3,
            node_ids=["node-a", "node-b", "node-a"],
            master_addr="10.0.0.1",
            master_port=12345,
        )


def test_build_ray_worker_env_vars_rejects_mismatched_worker_count() -> None:
    with pytest.raises(ValueError, match="one node id for every Ray worker"):
        build_ray_worker_env_vars(
            num_worker=2,
            node_ids=["node-a"],
            master_addr="10.0.0.1",
            master_port=12345,
        )


def test_get_placement_group_node_ids_reads_bundle_assignments(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "tinyexp.utils.ray_utils.ray.util.placement_group_table",
        lambda pg: {"bundles_to_node_id": {0: "node-a", 1: "node-b"}},
    )

    assert get_placement_group_node_ids(object(), 2) == ["node-a", "node-b"]


def test_get_placement_group_node_ids_accepts_string_bundle_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "tinyexp.utils.ray_utils.ray.util.placement_group_table",
        lambda pg: {"bundles_to_node_id": {"0": "node-a", "1": "node-b"}},
    )

    assert get_placement_group_node_ids(object(), 2) == ["node-a", "node-b"]


def test_get_placement_group_node_ids_rejects_missing_bundle_assignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tinyexp.utils.ray_utils.ray.util.placement_group_table",
        lambda pg: {"bundles_to_node_id": {0: "node-a"}},
    )

    with pytest.raises(RuntimeError, match="bundle 1"):
        get_placement_group_node_ids(object(), 2)


def test_get_num_worker_options_uses_final_topology_env_for_each_bundle() -> None:
    placement_group = SimpleNamespace(bundle_specs=[{"CPU": 2, "GPU": 0}])

    options_list = get_num_worker_options(
        placement_group,
        num_worker=4,
        gpu_ratio=0.0,
        num_cpus_per_worker=2,
        master_addr="10.0.0.1",
        master_port=12345,
        node_ids=["node-a", "node-b", "node-a", "node-b"],
    )

    assert [option["scheduling_strategy"].placement_group_bundle_index for option in options_list] == [
        0,
        1,
        2,
        3,
    ]
    assert [
        (
            option["runtime_env"]["env_vars"]["RANK"],
            option["runtime_env"]["env_vars"]["LOCAL_RANK"],
        )
        for option in options_list
    ] == [("0", "0"), ("2", "0"), ("1", "1"), ("3", "1")]
    assert all(
        set(option["runtime_env"]["env_vars"]) == {"WORLD_SIZE", "RANK", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"}
        for option in options_list
    )


def test_ray_cfg_run_resolves_topology_before_constructing_worker_actors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = OmegaConf.create(
        {
            "ray_cfg": {
                "ray_num_worker": 4,
                "ray_num_cpus_per_worker": 1,
                "ray_num_gpus_per_worker": 0,
                "ray_placement_strategy": "PACK",
            }
        }
    )
    events: list[object] = []

    class FakeWorker:
        def __init__(self, env_vars: dict[str, str]) -> None:
            self.set_cfg = SimpleNamespace(
                remote=lambda cfg: events.append(("set_cfg", env_vars["RANK"], cfg)),
            )
            self.run = SimpleNamespace(
                remote=lambda: events.append(("run", env_vars["RANK"])),
            )

    class FakeConfiguredActor:
        def __init__(self, options: dict[str, object]) -> None:
            self.options = options

        def remote(self) -> FakeWorker:
            env_vars = self.options["runtime_env"]["env_vars"]
            assert isinstance(env_vars, dict)
            events.append(("actor", dict(env_vars)))
            return FakeWorker(env_vars)

    class FakeRemoteClass:
        def options(self, **options: object) -> FakeConfiguredActor:
            return FakeConfiguredActor(options)

    monkeypatch.setattr("tinyexp.exp_mixins.basic_mixins.ray.init", lambda: None)
    monkeypatch.setattr("tinyexp.exp_mixins.basic_mixins.ray.remote", lambda exp_class: FakeRemoteClass())
    monkeypatch.setattr(
        "tinyexp.exp_mixins.basic_mixins.ray.cluster_resources",
        lambda: {"CPU": 4.0, "GPU": 0.0},
    )
    monkeypatch.setattr("tinyexp.exp_mixins.basic_mixins.ray.is_initialized", lambda: True)
    monkeypatch.setattr("tinyexp.exp_mixins.basic_mixins.ray.shutdown", lambda **kwargs: None)
    monkeypatch.setattr("tinyexp.exp_mixins.basic_mixins.ray.get", lambda refs, **kwargs: refs)
    monkeypatch.setattr(
        "tinyexp.exp_mixins.basic_mixins.get_placement_group",
        lambda **kwargs: SimpleNamespace(bundle_specs=[{"CPU": 1, "GPU": 0}]),
    )
    monkeypatch.setattr(
        "tinyexp.exp_mixins.basic_mixins.get_placement_group_node_ids",
        lambda pg, num_worker: (
            events.append(("topology",)),
            ["node-a", "node-b", "node-a", "node-b"],
        )[1],
    )
    monkeypatch.setattr(
        "tinyexp.exp_mixins.basic_mixins.get_network_config",
        lambda: ("10.0.0.1", 12345),
    )
    monkeypatch.setattr(
        "tinyexp.exp_mixins.basic_mixins.ray.util.remove_placement_group",
        lambda pg: None,
    )

    RayCfgMixin.RayCfg.run(object, cfg)

    topology_index = events.index(("topology",))
    actor_events = [event for event in events if event[0] == "actor"]
    assert topology_index < events.index(actor_events[0])
    assert [(event[1]["RANK"], event[1]["LOCAL_RANK"]) for event in actor_events] == [
        ("0", "0"),
        ("2", "0"),
        ("1", "1"),
        ("3", "1"),
    ]


def test_get_network_config_uses_public_ray_ip_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("tinyexp.utils.ray_utils.ray.util.get_node_ip_address", lambda: "10.0.0.7")

    master_addr, master_port = get_network_config()

    assert master_addr == "10.0.0.7"
    assert 0 < master_port <= 65535


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
    monkeypatch.setattr("tinyexp.utils.ray_utils.ray.get", lambda ref, **kwargs: ref)
    monkeypatch.setattr("tinyexp.utils.ray_utils.ray.is_initialized", lambda: False)

    get_placement_group(num_worker=2, num_gpus_per_worker=0, num_cpus_per_worker=3)

    assert captured == {
        "bundles": [{"CPU": 3, "GPU": 0}, {"CPU": 3, "GPU": 0}],
        "strategy": "PACK",
    }


def test_get_placement_group_timeout_removes_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    removed = []

    class FakePlacementGroup:
        def ready(self):
            return "ready"

    def fake_get(ref, **kwargs):  # type: ignore[no-untyped-def]
        assert kwargs == {"timeout": 3.5}
        raise ray.exceptions.GetTimeoutError("timed out")  # noqa: TRY003

    monkeypatch.setattr("tinyexp.utils.ray_utils.placement_group", lambda **kwargs: FakePlacementGroup())
    monkeypatch.setattr("tinyexp.utils.ray_utils.ray.get", fake_get)
    monkeypatch.setattr(
        "tinyexp.utils.ray_utils.ray.util.remove_placement_group",
        lambda pg: removed.append(pg),
    )

    with pytest.raises(TimeoutError, match="workers=2.*GPU/worker=0.5.*strategy=SPREAD"):
        get_placement_group(
            num_worker=2,
            num_gpus_per_worker=0.5,
            num_cpus_per_worker=3,
            strategy="SPREAD",
            timeout_s=3.5,
        )

    assert len(removed) == 1


def test_get_placement_group_error_removes_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    removed = []

    class FakePlacementGroup:
        def ready(self):
            return "ready"

    monkeypatch.setattr("tinyexp.utils.ray_utils.placement_group", lambda **kwargs: FakePlacementGroup())
    monkeypatch.setattr(
        "tinyexp.utils.ray_utils.ray.get",
        lambda ref, **kwargs: (_ for _ in ()).throw(RuntimeError("ready failed")),
    )
    monkeypatch.setattr(
        "tinyexp.utils.ray_utils.ray.util.remove_placement_group",
        lambda pg: removed.append(pg),
    )

    with pytest.raises(RuntimeError, match="ready failed"):
        get_placement_group(num_worker=1, num_gpus_per_worker=0, num_cpus_per_worker=1)

    assert len(removed) == 1


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
    monkeypatch.setattr("tinyexp.utils.ray_utils.ray.get", lambda ref, **kwargs: ref)
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
    monkeypatch.setattr("tinyexp.utils.ray_utils.ray.get", lambda ref, **kwargs: ref)

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
