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
    start_ray_rendezvous_store,
)

_MIXINS = "tinyexp.exp_mixins.basic_mixins"
_RAY_UTILS = "tinyexp.utils.ray_utils"


def _ray_cfg(**fields: object):  # type: ignore[no-untyped-def]
    return OmegaConf.create(
        {"ray_cfg": {"ray_num_worker": 1, "ray_num_cpus_per_worker": 1, "ray_num_gpus_per_worker": 0, **fields}}
    )


def _patch_driver_ray(
    monkeypatch: pytest.MonkeyPatch,
    *,
    total: dict[str, float],
    available: dict[str, float] | None = None,
) -> None:
    """Fake an already-initialized driver Ray runtime; ``available=None`` asserts it is never queried."""
    monkeypatch.setattr(f"{_MIXINS}.ray.init", lambda: None)
    monkeypatch.setattr(f"{_MIXINS}.ray.is_initialized", lambda: True)
    monkeypatch.setattr(f"{_MIXINS}.ray.shutdown", lambda **kwargs: None)
    monkeypatch.setattr(f"{_MIXINS}.ray.cluster_resources", lambda: total)
    monkeypatch.setattr(
        f"{_MIXINS}.ray.available_resources",
        (
            (lambda: available)
            if available is not None
            else (lambda: pytest.fail("explicit worker sizing should not query available resources"))
        ),
    )


def _stop_at(monkeypatch: pytest.MonkeyPatch, target: str) -> dict[str, object]:
    """Make ``target`` raise ``RuntimeError("stopped")``; returns the kwargs it was called with."""
    captured: dict[str, object] = {}

    def stop(*args: object, **kwargs: object) -> None:
        captured.update(kwargs)
        raise RuntimeError("stopped")

    monkeypatch.setattr(target, stop)
    return captured


def test_ray_cfg_run_uses_explicit_resources_and_shuts_down_its_own_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _ray_cfg(ray_num_cpus_per_worker=3, ray_placement_strategy="PACK", ray_placement_timeout_s=7)
    _patch_driver_ray(monkeypatch, total={"CPU": 3.0, "GPU": 0.0})
    ray_state = {"initialized": False}
    init_calls: list[None] = []
    shutdown_kwargs: list[dict[str, object]] = []

    def fake_init() -> None:
        init_calls.append(None)
        ray_state["initialized"] = True

    def fake_shutdown(**kwargs: object) -> None:
        shutdown_kwargs.append(kwargs)
        ray_state["initialized"] = False

    monkeypatch.setattr(f"{_MIXINS}.ray.init", fake_init)
    monkeypatch.setattr(f"{_MIXINS}.ray.is_initialized", lambda: ray_state["initialized"])
    monkeypatch.setattr(f"{_MIXINS}.ray.shutdown", fake_shutdown)
    monkeypatch.setattr(f"{_MIXINS}.ray.remote", lambda exp_class: object())
    captured = _stop_at(monkeypatch, f"{_MIXINS}.get_placement_group")

    with pytest.raises(RuntimeError, match="stopped"):
        RayCfgMixin.RayCfg.run(object, cfg)

    assert captured["num_worker"] == 1
    assert captured["num_cpus_per_worker"] == 3
    assert captured["num_gpus_per_worker"] == 0
    assert captured["timeout_s"] == 7.0
    assert init_calls == [None]
    assert shutdown_kwargs == [{"_exiting_interpreter": True}]
    assert ray_state["initialized"] is False


def test_ray_cfg_run_reuses_external_ray_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_driver_ray(monkeypatch, total={"CPU": 1.0, "GPU": 0.0})
    shutdown_calls: list[dict[str, object]] = []
    monkeypatch.setattr(f"{_MIXINS}.ray.init", lambda: pytest.fail("ray.init should not be called"))
    monkeypatch.setattr(f"{_MIXINS}.ray.shutdown", lambda **kwargs: shutdown_calls.append(kwargs))
    monkeypatch.setattr(f"{_MIXINS}.ray.remote", lambda exp_class: object())
    _stop_at(monkeypatch, f"{_MIXINS}.get_placement_group")

    with pytest.raises(RuntimeError, match="stopped"):
        RayCfgMixin.RayCfg.run(object, _ray_cfg(ray_placement_strategy="PACK"))

    assert shutdown_calls == []


@pytest.mark.parametrize(
    ("fields", "error", "match"),
    [
        ({"ray_num_worker": 0}, InvalidWorkerCountError, "Number of workers"),
        ({"ray_num_cpus_per_worker": 0}, ValueError, None),
        ({"ray_num_gpus_per_worker": -0.1}, ValueError, None),
        ({"ray_placement_timeout_s": 0}, ValueError, "ray_placement_timeout_s"),
    ],
)
def test_ray_cfg_run_rejects_invalid_cfg_before_ray_init(
    monkeypatch: pytest.MonkeyPatch, fields: dict[str, object], error: type, match: str | None
) -> None:
    monkeypatch.setattr(f"{_MIXINS}.ray.init", lambda: pytest.fail("ray.init should not be called"))

    with pytest.raises(error, match=match):
        RayCfgMixin.RayCfg.run(object, _ray_cfg(**fields))


@pytest.mark.parametrize(
    ("cpus", "gpus", "total", "available", "expected"),
    [
        # GPU workers: sized from the available (not total) GPUs.
        (1, 1, {"CPU": 8.0, "GPU": 2.0}, {"CPU": 8.0, "GPU": 1.0}, 1),
        # Capped by whichever of available CPU/GPU runs out first.
        (3, 1, {"CPU": 8.0, "GPU": 4.0}, {"CPU": 3.0, "GPU": 2.0}, 1),
        # CPU-only workers: sized from the available CPUs.
        (2, 0, {"CPU": 8.0, "GPU": 0.0}, {"CPU": 4.0, "GPU": 0.0}, 2),
    ],
)
def test_ray_cfg_run_resolves_auto_worker_count_from_available_resources(
    monkeypatch: pytest.MonkeyPatch,
    cpus: int,
    gpus: int,
    total: dict[str, float],
    available: dict[str, float],
    expected: int,
) -> None:
    cfg = _ray_cfg(ray_num_worker=-1, ray_num_cpus_per_worker=cpus, ray_num_gpus_per_worker=gpus)
    _patch_driver_ray(monkeypatch, total=total, available=available)
    _stop_at(monkeypatch, f"{_MIXINS}.ray.remote")

    with pytest.raises(RuntimeError, match="stopped"):
        RayCfgMixin.RayCfg.run(object, cfg)

    assert cfg.ray_cfg.ray_num_worker == expected


@pytest.mark.parametrize(
    ("ray_num_worker", "total", "available", "error", "match"),
    [
        (2, {"CPU": 8.0, "GPU": 1.0}, None, InsufficientCPUError, "needed GPU=2.0, available GPU=1.0"),
        (-1, {"CPU": 8.0}, {"CPU": 8.0}, InvalidWorkerCountError, "Number of workers"),
    ],
)
def test_ray_cfg_run_rejects_gpu_shortage_before_placement(
    monkeypatch: pytest.MonkeyPatch,
    ray_num_worker: int,
    total: dict[str, float],
    available: dict[str, float] | None,
    error: type,
    match: str,
) -> None:
    _patch_driver_ray(monkeypatch, total=total, available=available)
    monkeypatch.setattr(
        f"{_MIXINS}.get_placement_group", lambda **kwargs: pytest.fail("placement group should not be created")
    )

    with pytest.raises(error, match=match):
        RayCfgMixin.RayCfg.run(object, _ray_cfg(ray_num_worker=ray_num_worker, ray_num_gpus_per_worker=1))


def test_custom_ray_cfg_run_receives_class_and_global_cfg() -> None:
    calls = []

    class CustomRayCfg(RayCfgMixin.RayCfg):
        @classmethod
        def run(cls, exp_class, experiment_cfg):
            calls.append((cls, exp_class, experiment_cfg))

    cfg = OmegaConf.create({"ray_cfg": {"ray_num_worker": 1}, "mode": "run"})
    CustomRayCfg.run(object, cfg)

    assert calls == [(CustomRayCfg, object, cfg)]
    assert RayCfgMixin().get_ray_run_result() is None


def test_ray_cfg_run_resolves_topology_before_constructing_worker_actors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = _ray_cfg(ray_num_worker=4, ray_placement_strategy="PACK")
    events: list[object] = []
    rendezvous_actor = object()

    class FakeWorker:
        def __init__(self, env_vars: dict[str, str]) -> None:
            self.set_cfg = SimpleNamespace(
                remote=lambda cfg: events.append(("set_cfg", env_vars["RANK"], cfg)),
            )
            self.run = SimpleNamespace(
                remote=lambda: events.append(("run", env_vars["RANK"])),
            )
            self.get_ray_run_result = SimpleNamespace(
                remote=lambda: events.append(("result", env_vars["RANK"])) or f"result-{env_vars['RANK']}",
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

    _patch_driver_ray(monkeypatch, total={"CPU": 4.0, "GPU": 0.0})
    monkeypatch.setattr(f"{_MIXINS}.ray.remote", lambda exp_class: FakeRemoteClass())
    monkeypatch.setattr(f"{_MIXINS}.ray.get", lambda refs, **kwargs: refs)
    monkeypatch.setattr(
        f"{_MIXINS}.get_placement_group",
        lambda **kwargs: SimpleNamespace(bundle_specs=[{"CPU": 1, "GPU": 0}]),
    )
    monkeypatch.setattr(
        f"{_MIXINS}.get_placement_group_node_ids",
        lambda pg, num_worker: (
            events.append(("topology",)),
            ["node-a", "node-b", "node-a", "node-b"],
        )[1],
    )
    monkeypatch.setattr(
        f"{_MIXINS}.start_ray_rendezvous_store",
        lambda pg, world_size, timeout_s: (
            events.append(("rendezvous", world_size, timeout_s)),
            (rendezvous_actor, "10.0.0.1", 12345),
        )[1],
    )
    monkeypatch.setattr(
        f"{_MIXINS}.get_network_config",
        lambda master_node_id: pytest.fail("multi-worker Ray run should use the rendezvous store"),
    )
    monkeypatch.setattr(
        f"{_MIXINS}.ray.kill",
        lambda actor, no_restart: events.append(("kill", actor, no_restart)),
    )
    monkeypatch.setattr(
        f"{_MIXINS}.ray.util.remove_placement_group",
        lambda pg: None,
    )

    RayCfgMixin.RayCfg.run(object, cfg)

    topology_index = events.index(("topology",))
    rendezvous_index = events.index(("rendezvous", 4, 120.0))
    actor_events = [event for event in events if event[0] == "actor"]
    assert topology_index < rendezvous_index < events.index(actor_events[0])
    assert [(event[1]["RANK"], event[1]["LOCAL_RANK"]) for event in actor_events] == [
        ("0", "0"),
        ("2", "0"),
        ("1", "1"),
        ("3", "1"),
    ]
    assert {event[1]["MASTER_ADDR"] for event in actor_events} == {"10.0.0.1"}
    assert {event[1]["TORCHELASTIC_USE_AGENT_STORE"] for event in actor_events} == {"True"}
    assert [event for event in events if event[0] == "result"] == [("result", "0")]
    assert ("kill", rendezvous_actor, True) in events
    assert capsys.readouterr().out.endswith("result-0\n")


def test_build_worker_env_vars_ifname_and_agent_store(monkeypatch: pytest.MonkeyPatch) -> None:
    def build() -> dict[str, str]:
        return _build_worker_env_vars(num_worker=2, rank=1, local_rank=1, master_addr="10.0.0.1", master_port=12345)

    monkeypatch.delenv("GLOO_SOCKET_IFNAME", raising=False)
    env_vars = build()
    assert "GLOO_SOCKET_IFNAME" not in env_vars
    assert env_vars["TORCHELASTIC_USE_AGENT_STORE"] == "True"

    # A user-defined interface is forwarded to the workers.
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "eth9")
    assert build()["GLOO_SOCKET_IFNAME"] == "eth9"


_WORKER_ENV_KEYS = {"WORLD_SIZE", "RANK", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK", "TORCHELASTIC_USE_AGENT_STORE"}


@pytest.mark.parametrize(
    ("node_ids", "expected"),
    [
        # Bundles already grouped by node.
        (["node-a", "node-a", "node-b", "node-b"], [("0", "0"), ("1", "1"), ("2", "0"), ("3", "1")]),
        # Interleaved bundles: global ranks are grouped by node, local ranks count within a node.
        (["node-a", "node-b", "node-a", "node-b"], [("0", "0"), ("2", "0"), ("1", "1"), ("3", "1")]),
    ],
)
def test_build_ray_worker_env_vars_groups_ranks_by_node(
    monkeypatch: pytest.MonkeyPatch, node_ids: list[str], expected: list[tuple[str, str]]
) -> None:
    monkeypatch.delenv("GLOO_SOCKET_IFNAME", raising=False)

    env_vars = build_ray_worker_env_vars(num_worker=4, node_ids=node_ids, master_addr="10.0.0.1", master_port=12345)

    assert [set(env) for env in env_vars] == [_WORKER_ENV_KEYS] * 4
    assert [(env["RANK"], env["LOCAL_RANK"]) for env in env_vars] == expected


@pytest.mark.parametrize(
    ("num_worker", "node_ids", "match"),
    [
        (3, ["node-a", "node-b", "node-a"], "homogeneous.*node-a=2.*node-b=1"),
        (2, ["node-a"], "one node id for every Ray worker"),
    ],
)
def test_build_ray_worker_env_vars_rejects_invalid_topology(num_worker: int, node_ids: list[str], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        build_ray_worker_env_vars(num_worker=num_worker, node_ids=node_ids, master_addr="10.0.0.1", master_port=12345)


@pytest.mark.parametrize("bundles", [{0: "node-a", 1: "node-b"}, {"0": "node-a", "1": "node-b"}])
def test_get_placement_group_node_ids_reads_bundle_assignments(
    monkeypatch: pytest.MonkeyPatch, bundles: dict[object, str]
) -> None:
    monkeypatch.setattr(f"{_RAY_UTILS}.ray.util.placement_group_table", lambda pg: {"bundles_to_node_id": bundles})

    assert get_placement_group_node_ids(object(), 2) == ["node-a", "node-b"]


def test_get_placement_group_node_ids_rejects_missing_bundle_assignment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        f"{_RAY_UTILS}.ray.util.placement_group_table", lambda pg: {"bundles_to_node_id": {0: "node-a"}}
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

    assert [option["scheduling_strategy"].placement_group_bundle_index for option in options_list] == [0, 1, 2, 3]
    env_vars = [option["runtime_env"]["env_vars"] for option in options_list]
    assert [(env["RANK"], env["LOCAL_RANK"]) for env in env_vars] == [("0", "0"), ("2", "0"), ("1", "1"), ("3", "1")]
    assert all(set(env) == _WORKER_ENV_KEYS for env in env_vars)


def test_get_num_worker_options_requires_managed_endpoint_for_multiple_workers() -> None:
    placement_group = SimpleNamespace(bundle_specs=[{"CPU": 2, "GPU": 0}])

    with pytest.raises(ValueError, match="start_ray_rendezvous_store"):
        get_num_worker_options(placement_group, num_worker=2, gpu_ratio=0.0, node_ids=["node-a", "node-a"])


def test_get_network_config_uses_public_ray_ip_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(f"{_RAY_UTILS}.ray.util.get_node_ip_address", lambda: "10.0.0.7")

    master_addr, master_port = get_network_config()

    assert master_addr == "10.0.0.7"
    assert 0 < master_port <= 65535


def test_get_network_config_uses_requested_ray_node_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        f"{_RAY_UTILS}.ray.nodes",
        lambda: [
            {
                "Alive": True,
                "NodeID": "node-head",
                "NodeManagerAddress": "10.0.0.1",
            },
            {
                "Alive": True,
                "NodeID": "node-worker",
                "NodeManagerAddress": "10.0.0.2",
            },
        ],
    )

    master_addr, master_port = get_network_config("node-worker")

    assert master_addr == "10.0.0.2"
    assert 0 < master_port <= 65535


def test_get_network_config_rejects_missing_ray_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(f"{_RAY_UTILS}.ray.nodes", lambda: [])

    with pytest.raises(RuntimeError, match="node-worker.*not alive or is missing"):
        get_network_config("node-worker")


def _patch_rendezvous_actor(monkeypatch: pytest.MonkeyPatch, get) -> tuple[object, dict[str, object]]:  # type: ignore[no-untyped-def]
    """Fake ``ray.remote(...)(_RendezvousStore).options(...).remote(...)``; returns the actor and the captured args."""
    captured: dict[str, object] = {}
    actor = SimpleNamespace(get_endpoint=SimpleNamespace(remote=lambda: "endpoint-ref"))

    class FakeRemoteActorClass:
        def options(self, **options: object) -> SimpleNamespace:
            captured["options"] = options
            return SimpleNamespace(remote=lambda *args: captured.update(actor_args=args) or actor)

    def fake_remote(**kwargs: object):  # type: ignore[no-untyped-def]
        captured["remote_kwargs"] = kwargs
        return lambda actor_cls: FakeRemoteActorClass()

    monkeypatch.setattr(f"{_RAY_UTILS}.ray.remote", fake_remote)
    monkeypatch.setattr(f"{_RAY_UTILS}.ray.get", get)
    return actor, captured


def test_start_ray_rendezvous_store_uses_bundle_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    actor, captured = _patch_rendezvous_actor(monkeypatch, lambda ref, **kwargs: ("10.0.0.2", 23456))

    result = start_ray_rendezvous_store(object(), world_size=2, timeout_s=7.0)

    assert captured["remote_kwargs"] == {"num_cpus": 0}
    assert captured["actor_args"] == (2, 7.0)
    assert captured["options"]["scheduling_strategy"].placement_group_bundle_index == 0
    assert result == (actor, "10.0.0.2", 23456)


def test_start_ray_rendezvous_store_timeout_kills_actor(monkeypatch: pytest.MonkeyPatch) -> None:
    def timeout(ref, **kwargs):  # type: ignore[no-untyped-def]
        raise ray.exceptions.GetTimeoutError("timed out")  # noqa: TRY003

    actor, _captured = _patch_rendezvous_actor(monkeypatch, timeout)
    killed = []
    monkeypatch.setattr(
        f"{_RAY_UTILS}.ray.kill", lambda actor_handle, no_restart: killed.append((actor_handle, no_restart))
    )

    with pytest.raises(TimeoutError, match="rendezvous store timed out.*bundle 0"):
        start_ray_rendezvous_store(object(), world_size=2, timeout_s=3.5)

    assert killed == [(actor, True)]


class _FakePlacementGroup:
    def ready(self) -> str:
        return "ready"


@pytest.mark.parametrize(
    ("kwargs", "initialized", "expected"),
    [
        (
            {"num_worker": 2, "num_gpus_per_worker": 0, "num_cpus_per_worker": 3},
            False,
            {"bundles": [{"CPU": 3, "GPU": 0}] * 2, "strategy": "PACK"},
        ),
        # A connected head node does not get the first bundle pinned to it.
        (
            {"num_worker": 2, "num_gpus_per_worker": 1, "num_cpus_per_worker": 3},
            True,
            {"bundles": [{"CPU": 3, "GPU": 1}] * 2, "strategy": "PACK"},
        ),
        ({"num_worker": 1, "strategy": "SPREAD"}, False, {"bundles": [{"CPU": 10, "GPU": 1.0}], "strategy": "SPREAD"}),
    ],
)
def test_get_placement_group_builds_one_bundle_per_worker(
    monkeypatch: pytest.MonkeyPatch, kwargs: dict[str, object], initialized: bool, expected: dict[str, object]
) -> None:
    captured = {}

    def fake_placement_group(bundles, strategy):  # type: ignore[no-untyped-def]
        captured.update(bundles=bundles, strategy=strategy)
        return _FakePlacementGroup()

    monkeypatch.setattr(f"{_RAY_UTILS}.placement_group", fake_placement_group)
    monkeypatch.setattr(f"{_RAY_UTILS}.ray.get", lambda ref, **kwargs: ref)
    monkeypatch.setattr(f"{_RAY_UTILS}.ray.is_initialized", lambda: initialized)
    monkeypatch.setattr(f"{_RAY_UTILS}.ray.cluster_resources", lambda: {"node:__internal_head__": 1.0})

    get_placement_group(**kwargs)

    assert captured == expected


def test_get_placement_group_timeout_removes_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    removed = []
    events = []

    class FakePlacementGroup:
        def ready(self):
            return "ready"

    def fake_get(ref, **kwargs):  # type: ignore[no-untyped-def]
        assert kwargs == {"timeout": 3.5}
        raise ray.exceptions.GetTimeoutError("timed out")  # noqa: TRY003

    monkeypatch.setattr(f"{_RAY_UTILS}.placement_group", lambda **kwargs: FakePlacementGroup())
    monkeypatch.setattr(f"{_RAY_UTILS}.ray.get", fake_get)
    monkeypatch.setattr(
        f"{_RAY_UTILS}.ray.util.remove_placement_group",
        lambda pg: (removed.append(pg), events.append("remove")),
    )
    monkeypatch.setattr(
        f"{_RAY_UTILS}.ray.cluster_resources",
        lambda: (events.append("total"), {"CPU": 8.0, "GPU": 2.0})[1],
    )
    monkeypatch.setattr(
        f"{_RAY_UTILS}.ray.available_resources",
        lambda: (events.append("available"), {"CPU": 1.0, "GPU": 0.0})[1],
    )

    with pytest.raises(
        TimeoutError,
        match=(
            r"workers=2.*GPU/worker=0.5.*strategy=SPREAD.*"
            r"requested CPU=6.*requested GPU=1.0.*"
            r"total CPU=8.0.*total GPU=2.0.*"
            r"available CPU=1.0.*available GPU=0.0"
        ),
    ):
        get_placement_group(
            num_worker=2,
            num_gpus_per_worker=0.5,
            num_cpus_per_worker=3,
            strategy="SPREAD",
            timeout_s=3.5,
        )

    assert len(removed) == 1
    assert events == ["total", "available", "remove"]


def test_get_placement_group_error_removes_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    removed = []

    class FakePlacementGroup:
        def ready(self):
            return "ready"

    monkeypatch.setattr(f"{_RAY_UTILS}.placement_group", lambda **kwargs: FakePlacementGroup())
    monkeypatch.setattr(
        f"{_RAY_UTILS}.ray.get",
        lambda ref, **kwargs: (_ for _ in ()).throw(RuntimeError("ready failed")),
    )
    monkeypatch.setattr(
        f"{_RAY_UTILS}.ray.util.remove_placement_group",
        lambda pg: removed.append(pg),
    )

    with pytest.raises(RuntimeError, match="ready failed"):
        get_placement_group(num_worker=1, num_gpus_per_worker=0, num_cpus_per_worker=1)

    assert len(removed) == 1


@pytest.mark.parametrize(
    "redis_cfg",
    [
        None,
        {"redis_cache_enabled": False},
        # An external cluster (explicit world size > 1) is used as configured.
        {
            "redis_cache_enabled": True,
            "redis_cluster_host": "10.0.0.1",
            "redis_cluster_ports": [7000, 7001, 7002],
            "redis_rendezvous_world_size": 2,
        },
    ],
)
def test_maybe_start_ray_redis_cache_skips_unmanaged_cfg(redis_cfg: dict[str, object] | None) -> None:
    assert _maybe_start_ray_redis_cache(OmegaConf.create({} if redis_cfg is None else {"redis_cfg": redis_cfg})) is None


def test_maybe_start_ray_redis_cache_rejects_invalid_world_size() -> None:
    cfg = OmegaConf.create(
        {"redis_cfg": {"redis_cache_enabled": True, "redis_cluster_ports": [7000], "redis_rendezvous_world_size": 0}}
    )

    with pytest.raises(ValueError, match="redis_rendezvous_world_size"):
        _maybe_start_ray_redis_cache(cfg)


@pytest.mark.parametrize(
    ("world_size", "cluster_enabled", "started"),
    [(1, False, ("10.0.0.1", [7000], 1)), (-1, True, ("10.0.0.1", [7000, 7001, 7002], 2))],
)
def test_maybe_start_ray_redis_cache_writes_resolved_cfg(
    monkeypatch: pytest.MonkeyPatch, world_size: int, cluster_enabled: bool, started: tuple[str, list[int], int]
) -> None:
    created = []
    captured = {}

    class FakeRayRedisClusterManager:
        def __init__(self, redis_cfg) -> None:  # type: ignore[no-untyped-def]
            created.append(self)

        def start(self, *, cluster_enabled: bool) -> tuple[str, list[int], int]:
            captured["cluster_enabled"] = cluster_enabled
            return started

    monkeypatch.setattr(f"{_RAY_UTILS}.RayRedisClusterManager", FakeRayRedisClusterManager)
    cfg = OmegaConf.create(
        {
            "redis_cfg": {
                "redis_cache_enabled": True,
                "redis_cluster_host": "127.0.0.1",
                "redis_cluster_ports": [7000],
                "redis_rendezvous_world_size": world_size,
            }
        }
    )

    assert _maybe_start_ray_redis_cache(cfg) is created[0]
    assert captured["cluster_enabled"] is cluster_enabled
    host, ports, resolved_world_size = started
    assert cfg.redis_cfg.redis_cluster_host == host
    assert list(cfg.redis_cfg.redis_cluster_ports) == ports
    assert cfg.redis_cfg.redis_rendezvous_world_size == resolved_world_size
