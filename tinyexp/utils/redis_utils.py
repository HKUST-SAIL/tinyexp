from __future__ import annotations

import contextlib
import ipaddress
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import zlib
from collections.abc import Sequence
from numbers import Integral
from pathlib import Path
from typing import Any

import ray
import redis
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

_REDIS_CLUSTER_EXCEPTION = getattr(redis.exceptions, "RedisClusterException", redis.exceptions.RedisError)


class RedisClusterConfigError(ValueError):
    """Invalid arguments to :class:`RedisClusterManager` (e.g. ports, memory)."""


class RedisClusterStartupError(RuntimeError):
    """Servers failed to start, or :meth:`RedisClusterManager.__enter__` could not bring them up."""

    def __init__(
        self,
        message: str,
        *,
        port: int | None = None,
        last_error: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.port = port
        self.last_error = last_error


class RedisClientManager:
    def __init__(
        self,
        redis_host: str,
        redis_ports: list[int],
        redis_world_size: int = 1,
    ) -> None:
        """Manage Redis clients and shard keys across standalone Redis ports."""

        self.redis_clients: list[Any] = []

        self._init_redis_connection(redis_host, redis_ports, redis_world_size=redis_world_size)

    def _init_redis_connection(self, redis_host: str, redis_ports: list[int], *, redis_world_size: int) -> None:
        try:
            if redis_world_size <= 1:
                self.redis_clients = [
                    redis.Redis(
                        host=redis_host,
                        port=int(redis_port),
                        decode_responses=False,
                        socket_connect_timeout=5,
                        socket_timeout=5,
                    )
                    for redis_port in redis_ports
                ]
            else:
                self.redis_clients = [
                    redis.RedisCluster(
                        startup_nodes=[
                            redis.cluster.ClusterNode(redis_host, int(redis_port)) for redis_port in redis_ports
                        ],
                        decode_responses=False,
                        socket_connect_timeout=5,
                        socket_timeout=5,
                    )
                ]
            for redis_client in self.redis_clients:
                redis_client.ping()
        except Exception as e:
            print(f"Redis connection failed: {e}")
            self.redis_clients = []

    @staticmethod
    def _shard_index_for_key(key: Any, shard_count: int) -> int:
        if isinstance(key, Integral):
            return int(key) % shard_count
        if isinstance(key, bytes):
            key_bytes = key
        elif isinstance(key, memoryview):
            key_bytes = key.tobytes()
        else:
            key_bytes = str(key).encode("utf-8")
        return zlib.crc32(key_bytes) % shard_count

    def _redis_client_for_key(self, key: Any) -> Any | None:
        if not self.redis_clients:
            return None
        shard_index = self._shard_index_for_key(key, len(self.redis_clients))
        return self.redis_clients[shard_index]

    def safe_get(self, key: Any) -> Any | None:
        redis_client = self._redis_client_for_key(key)
        if redis_client is None:
            return None
        try:
            return redis_client.get(key)
        except (redis.exceptions.RedisError, _REDIS_CLUSTER_EXCEPTION):
            return None

    def safe_set(self, key: Any, value: Any) -> bool:
        redis_client = self._redis_client_for_key(key)
        if redis_client is None:
            return False
        try:
            return bool(redis_client.set(key, value))
        except (redis.exceptions.RedisError, _REDIS_CLUSTER_EXCEPTION):
            return False

    def __del__(self) -> None:
        for redis_client in self.redis_clients:
            with contextlib.suppress(Exception):
                redis_client.close()
        self.redis_clients = []


def wait_for_redis_cluster(startup_host: str, startup_ports: list[int], timeout_s: float = 30.0) -> bool:
    """Wait for a Redis Cluster to accept requests within a bounded deadline."""
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        cluster: Any | None = None
        ready = False
        attempt_timeout = min(2.0, max(0.05, deadline - time.monotonic()))
        try:
            cluster = redis.RedisCluster(
                startup_nodes=[redis.cluster.ClusterNode(startup_host, int(port)) for port in startup_ports],
                decode_responses=False,
                socket_connect_timeout=attempt_timeout,
                socket_timeout=attempt_timeout,
            )
            cluster.ping()
            ready = True
            for index in range(16):
                if time.monotonic() >= deadline:
                    last_error = TimeoutError("Redis Cluster readiness deadline expired")
                    ready = False
                    break
                key = f"tinyexp:redis-cluster-readiness:{index}"
                cluster.set(key, "ok", ex=60)
                if cluster.get(key) != b"ok":
                    last_error = RuntimeError("readiness value mismatch")
                    ready = False
                    break
        except Exception as exc:
            if not isinstance(exc, (redis.exceptions.RedisError, _REDIS_CLUSTER_EXCEPTION)):
                raise
            last_error = exc
        finally:
            if cluster is not None:
                with contextlib.suppress(Exception):
                    cluster.close()

        if ready:
            return True

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(1.0, remaining))

    print(f"Redis Cluster readiness check failed: {last_error}", file=sys.stderr)
    return False


class RedisClusterManager:
    def __init__(
        self,
        ports: Sequence[int],
        max_memory_per_port: float,
        *,
        host: str = "127.0.0.1",
        startup_timeout_s: float = 15.0,
        log_dir: str | Path | None = None,
        cluster_enabled: bool = False,
        log_startup: bool = True,
    ) -> None:
        """
        Initialize RedisClusterManager with specified ports and max memory per port.

        Args:
            ports: Redis shard ports.
            max_memory_per_port: Max memory per shard in GB. This value can be a float.
                It will be converted to bytes before being passed to `redis-server`.
            host: Host/interface to bind to.
            startup_timeout_s: Max seconds to wait for each shard to become healthy.
            log_dir: If provided, write `redis-server` stdout/stderr to `redis-<port>.log` under this directory.
            cluster_enabled: Start Redis servers as Redis Cluster nodes instead of standalone shards.

        Using ``with RedisClusterManager(...)`` calls :meth:`start_redis_cluster` in :meth:`__enter__`; if startup
        fails, :exc:`RedisClusterStartupError` is raised and the context body does not run. Direct calls to
        :meth:`start_redis_cluster` still return a ``bool`` without raising.
        """
        normalized_ports = [int(p) for p in ports]
        if not normalized_ports:
            raise RedisClusterConfigError("ports must not be empty")  # noqa: TRY003
        if len(set(normalized_ports)) != len(normalized_ports):
            raise RedisClusterConfigError(f"ports must be unique, got {normalized_ports!r}")  # noqa: TRY003
        for port in normalized_ports:
            if port <= 0 or port > 65535:
                raise RedisClusterConfigError(f"Invalid port {port!r}, expected 1..65535")  # noqa: TRY003

        cluster_enabled = bool(cluster_enabled)
        if cluster_enabled:
            data_ports = set(normalized_ports)
            bus_ports = [port + 10000 for port in normalized_ports]
            for data_port, bus_port in zip(normalized_ports, bus_ports):
                if bus_port <= 0 or bus_port > 65535:
                    raise RedisClusterConfigError(  # noqa: TRY003
                        f"Invalid Redis Cluster bus port {bus_port!r} derived from data port {data_port!r}, "
                        "expected 1..65535"
                    )
            conflicts = [
                (data_port, bus_port)
                for data_port, bus_port in zip(normalized_ports, bus_ports)
                if bus_port in data_ports
            ]
            if conflicts:
                conflict_text = ", ".join(
                    f"data port {data_port} derives bus port {bus_port}" for data_port, bus_port in conflicts
                )
                raise RedisClusterConfigError(f"Redis Cluster data/bus port conflict: {conflict_text}")  # noqa: TRY003

        if max_memory_per_port <= 0:
            raise RedisClusterConfigError(  # noqa: TRY003
                f"max_memory_per_port must be > 0 GB, got {max_memory_per_port!r}"
            )

        self.redis_processes: list[subprocess.Popen[Any]] = []
        self.redis_clients: list[redis.Redis] = []
        self._log_files: list[Any] = []

        self.host = host
        self.ports = normalized_ports
        self.max_memory_per_port_gb = float(max_memory_per_port)
        self.max_memory_per_port_bytes = self._gb_to_bytes(self.max_memory_per_port_gb)
        self.startup_timeout_s = float(startup_timeout_s)
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.cluster_enabled = bool(cluster_enabled)
        self.log_startup = bool(log_startup)
        self._last_startup_failure: Exception | None = None
        self._cluster_dir: tempfile.TemporaryDirectory[str] | None = None

    @staticmethod
    def _gb_to_bytes(gb: float) -> int:
        # Use binary GB (GiB) to match common memory accounting: 1 GB = 1024^3 bytes.
        return max(1, int(gb * (1024**3)))

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.stop_redis_cluster()

    def __enter__(self) -> RedisClusterManager:
        if not self.start_redis_cluster():
            failure = self._last_startup_failure
            self._last_startup_failure = None
            if failure is None:
                raise RedisClusterStartupError("Failed to start Redis cluster.")  # noqa: TRY003
            if isinstance(failure, RedisClusterStartupError):
                raise failure
            raise RedisClusterStartupError(str(failure)) from failure
        self._last_startup_failure = None
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        self.stop_redis_cluster()

    def _wait_until_healthy(self, client: redis.Redis, *, port: int, process: subprocess.Popen[Any]) -> None:
        deadline = time.monotonic() + self.startup_timeout_s
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            returncode = process.poll()
            if returncode is not None:
                raise RedisClusterStartupError(  # noqa: TRY003
                    f"Redis shard on port {port} exited during startup with code {returncode}.",
                    port=port,
                )
            try:
                client.ping()
                server_info = client.info("server")
                process_id = int(server_info.get("process_id", -1))
            except redis.exceptions.RedisError as e:
                last_error = e
                time.sleep(0.1)
                continue
            except (TypeError, ValueError) as e:
                raise RedisClusterStartupError(  # noqa: TRY003
                    f"Redis shard on port {port} did not report a valid process_id.",
                    port=port,
                    last_error=e,
                ) from e

            if process_id != process.pid:
                raise RedisClusterStartupError(  # noqa: TRY003
                    f"Redis port {port} is served by external process {process_id}; "
                    f"TinyExp started process {process.pid} and will not take ownership of the external server.",
                    port=port,
                )
            return
        raise RedisClusterStartupError(  # noqa: TRY003
            f"Redis shard on port {port} failed to start: {last_error}",
            port=port,
            last_error=last_error,
        )

    def _build_server_command(self, redis_server_path: str, port: int) -> list[str]:
        command = [
            redis_server_path,
            "--bind",
            self.host,
            "--port",
            str(port),
            "--daemonize",
            "no",
            "--save",
            "",
            "--appendonly",
            "no",
            "--maxmemory",
            str(self.max_memory_per_port_bytes),
        ]
        if not self.cluster_enabled:
            return command

        if self._cluster_dir is None:
            self._cluster_dir = tempfile.TemporaryDirectory(prefix="tinyexp-redis-")
        command.extend(
            [
                "--protected-mode",
                "no",
                "--dir",
                self._cluster_dir.name,
                "--cluster-enabled",
                "yes",
                "--cluster-config-file",
                f"tinyexp-redis-{port}.nodes.conf",
                "--cluster-node-timeout",
                "5000",
                "--cluster-announce-ip",
                self.host,
                "--cluster-announce-port",
                str(port),
                "--cluster-announce-bus-port",
                str(port + 10000),
            ]
        )
        return command

    def start_redis_cluster(self) -> bool:
        """
        Start multiple Redis server instances

        Returns:
            bool: True if all Redis servers started successfully, False otherwise.
        """
        self.stop_redis_cluster()
        self._last_startup_failure = None

        redis_server_path = shutil.which("redis-server")
        if redis_server_path is None:
            self._last_startup_failure = RedisClusterStartupError(
                "redis-server command not found in PATH; install Redis before using this context manager.",
            )
            print("redis-server command not found. Please install it before enabling Redis cache.")
            return False

        try:
            if self.log_dir is not None:
                self.log_dir.mkdir(parents=True, exist_ok=True)

            for i, port in enumerate(self.ports):
                stdout: Any = subprocess.DEVNULL
                stderr: Any = subprocess.DEVNULL
                if self.log_dir is not None:
                    log_path = self.log_dir / f"redis-{port}.log"
                    log_file = open(log_path, "ab")  # noqa: SIM115
                    self._log_files.append(log_file)
                    stdout = log_file
                    stderr = log_file

                redis_process = subprocess.Popen(
                    self._build_server_command(redis_server_path, port),
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                )
                self.redis_processes.append(redis_process)

                # Create Redis client connection
                redis_client = redis.StrictRedis(
                    host=self.host,
                    port=port,
                    decode_responses=False,
                    socket_connect_timeout=1,
                    socket_timeout=1,
                )
                self.redis_clients.append(redis_client)
                self._wait_until_healthy(redis_client, port=port, process=redis_process)

                if self.log_startup:
                    print(f"==> Redis shard {i} started on port {port}", flush=True)

        except Exception as e:
            self._last_startup_failure = e
            print(f"Failed to start Redis cluster: {e}")
            self.stop_redis_cluster()
            return False
        else:
            self._last_startup_failure = None
            return True

    def stop_redis_cluster(self):
        """Stop all Redis servers"""
        for client in self.redis_clients:
            with contextlib.suppress(Exception):
                if hasattr(client, "close"):
                    client.close()
                else:
                    client.connection_pool.disconnect()

        for process in self.redis_processes:
            if process and process.poll() is None:  # Check if process is still running
                try:
                    process.terminate()
                    process.wait(timeout=5)  # Add timeout
                except subprocess.TimeoutExpired:
                    process.kill()  # Force terminate
                    process.wait()
                except Exception as e:
                    print(f"Error stopping Redis process: {e}")
        self.redis_processes.clear()
        self.redis_clients.clear()
        for f in self._log_files:
            with contextlib.suppress(Exception):
                f.close()
        self._log_files.clear()
        if self._cluster_dir is not None:
            with contextlib.suppress(Exception):
                self._cluster_dir.cleanup()
            self._cluster_dir = None

    def get_redis_memory_info(self):
        """Get Redis memory usage info"""
        memory_info = {}
        for i, client in enumerate(self.redis_clients):
            try:
                info = client.info("memory")
                used_memory = info["used_memory"] / 1024 / 1024  # MB
                used_memory_human = info["used_memory_human"]
                memory_info[f"redis_{self.ports[i]}"] = {
                    "used_memory_mb": used_memory,
                    "used_memory_human": used_memory_human,
                }
            except Exception as e:
                memory_info[f"redis_{self.ports[i]}"] = {"error": str(e)}
        return memory_info


def _is_ip_address(value: str) -> bool:
    with contextlib.suppress(ValueError):
        ipaddress.ip_address(value)
        return True
    return False


def _get_node_ip_address() -> str:
    with (
        contextlib.suppress(Exception),
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock,
    ):
        sock.connect(("8.8.8.8", 80))
        return str(sock.getsockname()[0])
    with contextlib.suppress(Exception):
        node_ip = ray.util.get_node_ip_address()
        if _is_ip_address(node_ip):
            return node_ip
    with contextlib.suppress(Exception):
        node_ip = ray.util.get_node_ip_address()
        if _is_ip_address(node_ip):
            return node_ip
    return socket.gethostbyname(socket.gethostname())


class _RayRedisNodeActor:
    def __init__(self) -> None:
        self._manager: RedisClusterManager | None = None
        self._node_host: str | None = None

    def start(
        self,
        ports: list[int],
        max_memory_per_port: float,
        log_dir: str | None,
        cluster_enabled: bool,
        log_startup: bool,
    ) -> dict[str, Any]:
        node_ip = _get_node_ip_address()
        self._node_host = node_ip
        redis_host = node_ip if cluster_enabled else "127.0.0.1"
        self._manager = RedisClusterManager(
            ports=ports,
            max_memory_per_port=max_memory_per_port,
            host=redis_host,
            log_dir=log_dir,
            cluster_enabled=cluster_enabled,
            log_startup=log_startup,
        )
        if not self._manager.start_redis_cluster():
            failure = self._manager._last_startup_failure
            if isinstance(failure, RedisClusterStartupError):
                raise failure
            raise RedisClusterStartupError(f"Failed to start Redis on Ray node {node_ip}: {failure}")  # noqa: TRY003
        return {"host": redis_host, "node_host": node_ip, "ports": ports}

    def get_db_sizes(self) -> dict[str, Any]:
        if self._manager is None:
            return {"host": _get_node_ip_address(), "dbsizes": {}}
        dbsizes = {}
        for port, client in zip(self._manager.ports, self._manager.redis_clients):
            with contextlib.suppress(Exception):
                dbsizes[str(port)] = int(client.dbsize())
        return {
            "host": self._node_host or self._manager.host,
            "client_host": self._manager.host,
            "dbsizes": dbsizes,
        }

    def stop(self) -> None:
        if self._manager is not None:
            self._manager.stop_redis_cluster()
            self._manager = None


class RayRedisClusterManager:
    def __init__(self, redis_cfg: Any, *, log_dir: str | Path | None = None) -> None:
        self.redis_cfg = redis_cfg
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self._actors: list[Any] = []

    def start(self, *, cluster_enabled: bool | None = None) -> tuple[str, list[int], int]:  # noqa: C901
        if not ray.is_initialized():
            raise RedisClusterStartupError("Ray must be initialized before starting Ray-managed Redis.")  # noqa: TRY003

        alive_nodes = [node for node in ray.nodes() if node.get("Alive")]
        if not alive_nodes:
            raise RedisClusterStartupError("No alive Ray nodes found for Redis cache.")  # noqa: TRY003

        if cluster_enabled is None:
            cluster_enabled = len(alive_nodes) > 1

        startup_timeout_s = float(getattr(self.redis_cfg, "redis_cluster_startup_timeout_s", 30.0))
        if cluster_enabled and startup_timeout_s <= 0:
            raise RedisClusterStartupError("redis_cluster_startup_timeout_s must be greater than 0")  # noqa: TRY003

        ports = [int(port) for port in self.redis_cfg.redis_cluster_ports]
        total_redis_shards = len(ports) * len(alive_nodes)
        if cluster_enabled and total_redis_shards < 3:
            raise RedisClusterStartupError(  # noqa: TRY003
                "Redis Cluster requires at least 3 Redis server nodes/shards, "
                f"got {total_redis_shards} shard(s) from {len(alive_nodes)} Ray node(s)."
            )
        redis_cli_path = shutil.which("redis-cli") if cluster_enabled else None
        if cluster_enabled and redis_cli_path is None:
            raise RedisClusterStartupError(  # noqa: TRY003
                "redis-cli command not found in PATH; install Redis before enabling Redis cache."
            )

        memory_shard_count = total_redis_shards if cluster_enabled else len(ports)
        max_memory_per_port = float(self.redis_cfg.redis_cache_max_memory) / memory_shard_count
        remote_actor_cls = ray.remote(num_cpus=0)(_RayRedisNodeActor)
        start_refs = []

        try:
            for node in alive_nodes:
                node_id = node.get("NodeID")
                if not node_id:
                    raise RedisClusterStartupError(f"Ray node is missing NodeID: {node!r}")  # noqa: TRY003, TRY301
                actor = remote_actor_cls.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
                ).remote()
                self._actors.append(actor)
                start_refs.append(
                    actor.start.remote(
                        ports,
                        max_memory_per_port,
                        str(self.log_dir) if self.log_dir else None,
                        cluster_enabled,
                        False,
                    )
                )

            redis_nodes = ray.get(start_refs)
            self._log_started_nodes(redis_nodes)
            startup_host = str(redis_nodes[0]["host"])
            startup_ports = [int(port) for port in redis_nodes[0]["ports"]]
            if cluster_enabled:
                self._validate_node_hosts(redis_nodes)
                self._create_cluster(redis_cli_path, redis_nodes, timeout_s=startup_timeout_s)
                self._check_cluster(startup_host, startup_ports, timeout_s=startup_timeout_s)
        except Exception:
            self.stop()
            raise
        world_size = len(redis_nodes) if cluster_enabled else 1
        return startup_host, startup_ports, world_size

    @staticmethod
    def _log_started_nodes(redis_nodes: list[dict[str, Any]]) -> None:
        for node in redis_nodes:
            for i, port in enumerate(node["ports"]):
                print(f"Redis shard {i} started on port {port}", flush=True)

    @staticmethod
    def _validate_node_hosts(redis_nodes: list[dict[str, Any]]) -> None:
        if len(redis_nodes) <= 1:
            return
        loopback_hosts = {"127.0.0.1", "localhost", "::1"}
        bad_hosts = [str(node["host"]) for node in redis_nodes if str(node["host"]) in loopback_hosts]
        if bad_hosts:
            raise RedisClusterStartupError(  # noqa: TRY003
                f"Ray-managed Redis received loopback host(s) for a multi-node cluster: {bad_hosts!r}."
            )

    @staticmethod
    def _create_cluster(
        redis_cli_path: str,
        redis_nodes: list[dict[str, Any]],
        *,
        timeout_s: float = 30.0,
    ) -> None:
        addresses = [f"{node['host']}:{port}" for node in redis_nodes for port in node["ports"]]
        try:
            result = subprocess.run(
                [
                    redis_cli_path,
                    "--cluster",
                    "create",
                    *addresses,
                    "--cluster-replicas",
                    "0",
                    "--cluster-yes",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise RedisClusterStartupError(  # noqa: TRY003
                f"redis-cli cluster create timed out after {timeout_s}s"
            ) from exc
        if result.returncode != 0:
            raise RedisClusterStartupError(  # noqa: TRY003
                f"redis-cli cluster create failed with code {result.returncode}: {result.stderr or result.stdout}"
            )

    @staticmethod
    def _check_cluster(startup_host: str, startup_ports: list[int], *, timeout_s: float = 30.0) -> None:
        if not wait_for_redis_cluster(startup_host, startup_ports, timeout_s=timeout_s):
            raise RedisClusterStartupError(  # noqa: TRY003
                f"Redis Cluster did not become ready at {startup_host}:{startup_ports}"
            )

    def get_db_sizes(self) -> list[dict[str, Any]]:
        dbsize_refs = [actor.get_db_sizes.remote() for actor in self._actors]
        return list(ray.get(dbsize_refs)) if dbsize_refs else []

    def stop(self) -> None:
        with contextlib.suppress(Exception):
            print(f"Ray Redis dbsize by node: {self.get_db_sizes()}", flush=True)
        for actor in self._actors:
            with contextlib.suppress(Exception):
                ray.get(actor.stop.remote())
            with contextlib.suppress(Exception):
                ray.kill(actor, no_restart=True)
        self._actors.clear()
