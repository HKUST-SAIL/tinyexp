from __future__ import annotations

import subprocess
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import redis

from tinyexp.utils.redis_utils import (
    RayRedisClusterManager,
    RedisClientManager,
    RedisClusterManager,
    RedisClusterStartupError,
    wait_for_redis_cluster,
)


class _FakeNodeAffinitySchedulingStrategy:
    def __init__(self, node_id: str, soft: bool) -> None:
        self.node_id = node_id
        self.soft = soft


class _FakeRemoteMethod:
    def __init__(self, fn):  # type: ignore[no-untyped-def]
        self._fn = fn

    def remote(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return self._fn(*args, **kwargs)


class _FakeRayRedisActor:
    def __init__(self, host: str, captured: dict[str, Any]) -> None:
        self.host = host
        self.stopped = False
        self._captured = captured
        self.start = _FakeRemoteMethod(self._start)
        self.get_db_sizes = _FakeRemoteMethod(self._get_db_sizes)
        self.stop = _FakeRemoteMethod(self._stop)

    def _start(self, ports, max_memory_per_port, log_dir, cluster_enabled, log_startup):  # type: ignore[no-untyped-def]
        self._captured["start_args"].append((list(ports), max_memory_per_port, log_dir, cluster_enabled, log_startup))
        return {"host": self.host, "ports": list(ports)}

    def _get_db_sizes(self):  # type: ignore[no-untyped-def]
        return {"host": self.host, "dbsizes": {"7000": 1}}

    def _stop(self) -> None:
        self.stopped = True


class _FakeRemoteActorClass:
    def __init__(
        self,
        hosts: list[str],
        actors: list[_FakeRayRedisActor],
        captured: dict[str, Any],
    ) -> None:
        self._hosts = hosts
        self._actors = actors
        self._captured = captured

    def options(self, *, scheduling_strategy):  # type: ignore[no-untyped-def]
        self._captured["strategies"].append(scheduling_strategy)
        return self

    def remote(self):  # type: ignore[no-untyped-def]
        host = self._hosts[len(self._actors)]
        actor = _FakeRayRedisActor(host, self._captured)
        self._actors.append(actor)
        return actor


def _install_fake_ray(
    monkeypatch: pytest.MonkeyPatch,
    *,
    nodes: list[dict[str, Any]],
    actor_hosts: list[str],
) -> tuple[dict[str, Any], list[_FakeRayRedisActor], list[tuple[_FakeRayRedisActor, bool]]]:
    actors: list[_FakeRayRedisActor] = []
    killed: list[tuple[_FakeRayRedisActor, bool]] = []
    captured: dict[str, Any] = {"strategies": [], "start_args": [], "commands": []}

    def fake_remote(**kwargs):  # type: ignore[no-untyped-def]
        captured["remote_kwargs"] = kwargs

        def decorate(_cls):  # type: ignore[no-untyped-def]
            return _FakeRemoteActorClass(actor_hosts, actors, captured)

        return decorate

    monkeypatch.setattr("tinyexp.utils.redis_utils.ray.is_initialized", lambda: True)
    monkeypatch.setattr("tinyexp.utils.redis_utils.ray.nodes", lambda: nodes)
    monkeypatch.setattr("tinyexp.utils.redis_utils.ray.remote", fake_remote)
    monkeypatch.setattr("tinyexp.utils.redis_utils.ray.get", lambda ref: ref)
    monkeypatch.setattr(
        "tinyexp.utils.redis_utils.ray.kill",
        lambda actor, no_restart: killed.append((actor, no_restart)),
    )
    monkeypatch.setattr(
        "tinyexp.utils.redis_utils.NodeAffinitySchedulingStrategy",
        _FakeNodeAffinitySchedulingStrategy,
    )
    return captured, actors, killed


def test_redis_client_manager_hash_sharding_covers_all_clients_for_string_keys() -> None:
    shard_indexes = [RedisClientManager._shard_index_for_key(f"image:{index}", 6) for index in range(600)]

    assert set(shard_indexes) == set(range(6))


def test_redis_client_manager_accepts_non_integer_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, str, Any]] = []

    class FakeRedis:
        def __init__(self, *, host, port, **kwargs) -> None:  # type: ignore[no-untyped-def]
            self.port = port

        def ping(self) -> bool:
            return True

        def get(self, key):  # type: ignore[no-untyped-def]
            calls.append((self.port, "get", key))
            return b"cached"

        def set(self, key, value) -> bool:  # type: ignore[no-untyped-def]
            calls.append((self.port, "set", key))
            return True

    monkeypatch.setattr("tinyexp.utils.redis_utils.redis.Redis", FakeRedis)

    manager = RedisClientManager(redis_host="127.0.0.1", redis_ports=[7000, 7001, 7002])
    assert manager.safe_set("train:sample:0", b"data") is True
    assert manager.safe_get(b"train:sample:1") == b"cached"

    assert calls[0][2] == "train:sample:0"
    assert calls[1][2] == b"train:sample:1"
    assert {call[0] for call in calls} <= {7000, 7001, 7002}
    assert manager._redis_client_for_key("train:sample:0") is manager._redis_client_for_key("train:sample:0")


def test_wait_for_redis_cluster_retries_constructor_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def failing_constructor(**_kwargs: Any) -> None:
        calls.append(True)
        raise redis.exceptions.RedisClusterException("cluster is not reachable")  # noqa: TRY003

    monkeypatch.setattr("tinyexp.utils.redis_utils.redis.RedisCluster", failing_constructor)

    assert wait_for_redis_cluster("10.0.0.1", [7000, 7001, 7002], timeout_s=1.05) is False
    assert len(calls) >= 2


def test_wait_for_redis_cluster_recovers_from_cluster_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    closed = []

    class FakeRedisCluster:
        def __init__(self, **_kwargs: Any) -> None:
            calls.append(True)

        def ping(self) -> None:
            if len(calls) == 1:
                raise redis.exceptions.RedisClusterException("slots are not covered")  # noqa: TRY003

        def set(self, key: str, value: str, ex: int) -> None:
            pass

        def get(self, key: str) -> bytes:
            return b"ok"

        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr("tinyexp.utils.redis_utils.redis.RedisCluster", FakeRedisCluster)

    assert wait_for_redis_cluster("10.0.0.1", [7000, 7001, 7002], timeout_s=2) is True
    assert len(calls) == 2
    assert len(closed) == 2


def test_ray_redis_cluster_manager_create_cluster_uses_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_run(command, **kwargs):  # type: ignore[no-untyped-def]
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("tinyexp.utils.redis_utils.subprocess.run", fake_run)

    RayRedisClusterManager._create_cluster(
        "/usr/bin/redis-cli",
        [{"host": "10.0.0.1", "ports": [7000, 7001, 7002]}],
        timeout_s=4.5,
    )

    assert captured["kwargs"]["timeout"] == 4.5


def test_ray_redis_cluster_manager_create_cluster_timeout_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=1.5)

    monkeypatch.setattr("tinyexp.utils.redis_utils.subprocess.run", fake_run)

    with pytest.raises(RedisClusterStartupError, match="cluster create timed out after 1.5s"):
        RayRedisClusterManager._create_cluster(
            "/usr/bin/redis-cli",
            [{"host": "10.0.0.1", "ports": [7000, 7001, 7002]}],
            timeout_s=1.5,
        )


def test_redis_cluster_manager_validates_inputs() -> None:
    with pytest.raises(ValueError, match="ports must not be empty"):
        RedisClusterManager(ports=[], max_memory_per_port=1.0)

    with pytest.raises(ValueError, match="ports must be unique"):
        RedisClusterManager(ports=[7000, 7000], max_memory_per_port=1.0)

    with pytest.raises(ValueError, match="Invalid port"):
        RedisClusterManager(ports=[0], max_memory_per_port=1.0)

    with pytest.raises(ValueError, match="max_memory_per_port must be > 0 GB"):
        RedisClusterManager(ports=[7000], max_memory_per_port=0.0)


def test_redis_cluster_manager_converts_gb_to_bytes() -> None:
    mgr = RedisClusterManager(ports=[7000], max_memory_per_port=0.5)
    assert mgr.max_memory_per_port_gb == 0.5
    assert mgr.max_memory_per_port_bytes == int(0.5 * (1024**3))


def test_redis_cluster_manager_context_raises_when_redis_server_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("tinyexp.utils.redis_utils.shutil.which", lambda _cmd: None)
    with (
        pytest.raises(RedisClusterStartupError, match="redis-server command not found"),
        RedisClusterManager(ports=[7000], max_memory_per_port=0.5),
    ):
        pass


def test_redis_cluster_manager_cluster_mode_adds_expected_server_args(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    commands: list[list[str]] = []

    class FakeProcess:
        pid = 4321

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            pass

        def wait(self, timeout=None) -> int:  # type: ignore[no-untyped-def]
            return 0

    class FakeRedis:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def ping(self) -> None:
            pass

        def info(self, section: str) -> dict[str, int]:
            assert section == "server"
            return {"process_id": FakeProcess.pid}

        def close(self) -> None:
            pass

    def fake_popen(command, stdout, stderr, start_new_session):  # type: ignore[no-untyped-def]
        assert start_new_session is True
        commands.append(command)
        return FakeProcess()

    monkeypatch.setattr("tinyexp.utils.redis_utils.shutil.which", lambda _cmd: "/usr/bin/redis-server")
    monkeypatch.setattr("tinyexp.utils.redis_utils.subprocess.Popen", fake_popen)
    monkeypatch.setattr("tinyexp.utils.redis_utils.redis.StrictRedis", FakeRedis)

    manager = RedisClusterManager(
        ports=[7000],
        max_memory_per_port=0.5,
        host="10.0.0.1",
        log_dir=tmp_path,
        cluster_enabled=True,
    )

    assert manager.start_redis_cluster() is True
    command = commands[0]
    assert "--cluster-enabled" in command
    assert command[command.index("--cluster-enabled") + 1] == "yes"
    assert command[command.index("--cluster-announce-ip") + 1] == "10.0.0.1"
    assert command[command.index("--cluster-announce-port") + 1] == "7000"
    assert command[command.index("--cluster-announce-bus-port") + 1] == "17000"
    assert command[command.index("--cluster-config-file") + 1] == "tinyexp-redis-7000.nodes.conf"
    manager.stop_redis_cluster()


def test_redis_cluster_manager_refuses_external_process_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeProcess:
        pid = 4321

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            events.append("terminate-owned-process")

        def wait(self, timeout=None) -> int:  # type: ignore[no-untyped-def]
            events.append("wait-owned-process")
            return 0

        def kill(self) -> None:
            events.append("kill-owned-process")

    class ExternalRedis:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def ping(self) -> None:
            pass

        def info(self, section: str) -> dict[str, int]:
            assert section == "server"
            return {"process_id": 9876}

        def shutdown(self, **_kwargs: Any) -> None:
            pytest.fail("external Redis must not be shut down")

        def close(self) -> None:
            events.append("close-client")

    monkeypatch.setattr("tinyexp.utils.redis_utils.shutil.which", lambda _cmd: "/usr/bin/redis-server")
    monkeypatch.setattr("tinyexp.utils.redis_utils.subprocess.Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr("tinyexp.utils.redis_utils.redis.StrictRedis", ExternalRedis)

    manager = RedisClusterManager(ports=[7000], max_memory_per_port=0.5)

    assert manager.start_redis_cluster() is False
    assert isinstance(manager._last_startup_failure, RedisClusterStartupError)
    assert "external process 9876" in str(manager._last_startup_failure)
    assert events == ["close-client", "terminate-owned-process", "wait-owned-process"]


def test_redis_cluster_manager_cleans_partial_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class FakeProcess:
        def __init__(self, pid: int) -> None:
            self.pid = pid

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            events.append(f"terminate-{self.pid}")

        def wait(self, timeout=None) -> int:  # type: ignore[no-untyped-def]
            events.append(f"wait-{self.pid}")
            return 0

    class FakeRedis:
        def __init__(self, *, port: int, **_kwargs: Any) -> None:
            self.port = port

        def ping(self) -> None:
            pass

        def info(self, section: str) -> dict[str, int]:
            assert section == "server"
            return {"process_id": 1001}

        def close(self) -> None:
            events.append(f"close-client-{self.port}")

    owned_process = FakeProcess(1001)
    monkeypatch.setattr("tinyexp.utils.redis_utils.shutil.which", lambda _cmd: "/usr/bin/redis-server")
    monkeypatch.setattr(
        "tinyexp.utils.redis_utils.subprocess.Popen",
        Mock(side_effect=[owned_process, OSError()]),
    )
    monkeypatch.setattr("tinyexp.utils.redis_utils.redis.StrictRedis", FakeRedis)

    manager = RedisClusterManager(ports=[7000, 7001], max_memory_per_port=0.5)

    assert manager.start_redis_cluster() is False
    assert events == ["close-client-7000", "terminate-1001", "wait-1001"]
    assert manager.redis_processes == []
    assert manager.redis_clients == []


def test_ray_redis_cluster_manager_starts_pinned_actors_and_creates_cluster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured, actors, killed = _install_fake_ray(
        monkeypatch,
        nodes=[
            {"Alive": True, "NodeID": "node-a"},
            {"Alive": False, "NodeID": "node-dead"},
            {"Alive": True, "NodeID": "node-b"},
        ],
        actor_hosts=["10.0.0.1", "10.0.0.2"],
    )

    class FakeRedisCluster:
        def __init__(self, **kwargs: Any) -> None:
            captured["cluster_kwargs"] = kwargs

        def ping(self) -> None:
            pass

        def set(self, key: str, value: str, ex: int) -> None:
            captured.setdefault("readiness_values", {})[key] = value

        def get(self, key: str) -> bytes:
            return b"ok"

        def close(self) -> None:
            pass

    def fake_run(command, check, capture_output, text, **kwargs):  # type: ignore[no-untyped-def]
        captured["commands"].append(command)
        captured["command_kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("tinyexp.utils.redis_utils.shutil.which", lambda _cmd: "/usr/bin/redis-cli")
    monkeypatch.setattr("tinyexp.utils.redis_utils.subprocess.run", fake_run)
    monkeypatch.setattr("tinyexp.utils.redis_utils.redis.RedisCluster", FakeRedisCluster)

    manager = RayRedisClusterManager(SimpleNamespace(redis_cluster_ports=[7000, 7001, 7002], redis_cache_max_memory=60))

    startup_host, startup_ports, world_size = manager.start()

    assert captured["remote_kwargs"] == {"num_cpus": 0}
    assert [strategy.node_id for strategy in captured["strategies"]] == [
        "node-a",
        "node-b",
    ]
    assert [strategy.soft for strategy in captured["strategies"]] == [False, False]
    assert captured["start_args"] == [
        ([7000, 7001, 7002], 10.0, None, True, False),
        ([7000, 7001, 7002], 10.0, None, True, False),
    ]
    assert captured["commands"] == [
        [
            "/usr/bin/redis-cli",
            "--cluster",
            "create",
            "10.0.0.1:7000",
            "10.0.0.1:7001",
            "10.0.0.1:7002",
            "10.0.0.2:7000",
            "10.0.0.2:7001",
            "10.0.0.2:7002",
            "--cluster-replicas",
            "0",
            "--cluster-yes",
        ]
    ]
    assert captured["command_kwargs"] == {"timeout": 30.0}
    assert startup_host == "10.0.0.1"
    assert startup_ports == [7000, 7001, 7002]
    assert world_size == 2

    manager.stop()

    assert [actor.stopped for actor in actors] == [True, True]
    assert killed == [(actors[0], True), (actors[1], True)]


def test_ray_redis_cluster_manager_standalone_starts_on_all_alive_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured, _actors, _killed = _install_fake_ray(
        monkeypatch,
        nodes=[
            {"Alive": True, "NodeID": "node-worker"},
            {"Alive": True, "NodeID": "node-head"},
        ],
        actor_hosts=["127.0.0.1", "127.0.0.1"],
    )
    monkeypatch.setattr(
        "tinyexp.utils.redis_utils.subprocess.run",
        lambda *args, **kwargs: captured["commands"].append(args),
    )

    manager = RayRedisClusterManager(SimpleNamespace(redis_cluster_ports=[7000, 7001], redis_cache_max_memory=10))

    startup_host, startup_ports, world_size = manager.start(cluster_enabled=False)

    assert [strategy.node_id for strategy in captured["strategies"]] == [
        "node-worker",
        "node-head",
    ]
    assert captured["start_args"] == [
        ([7000, 7001], 5.0, None, False, False),
        ([7000, 7001], 5.0, None, False, False),
    ]
    assert captured["commands"] == []
    assert startup_host == "127.0.0.1"
    assert startup_ports == [7000, 7001]
    assert world_size == 1

    manager.stop()
