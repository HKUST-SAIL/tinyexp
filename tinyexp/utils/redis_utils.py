from __future__ import annotations

import contextlib
import shutil
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import redis


class RedisClusterConfigError(ValueError):
    """Invalid configuration for RedisClusterManager."""


class EmptyRedisPortsError(RedisClusterConfigError):
    def __init__(self) -> None:
        super().__init__("ports must not be empty")


class DuplicateRedisPortsError(RedisClusterConfigError):
    def __init__(self, ports: Sequence[int]) -> None:
        self.ports = list(ports)
        super().__init__(f"ports must be unique, got {self.ports!r}")


class InvalidRedisPortError(RedisClusterConfigError):
    def __init__(self, port: int) -> None:
        self.port = port
        super().__init__(f"Invalid port {port!r}, expected 1..65535")


class InvalidRedisMaxMemoryError(RedisClusterConfigError):
    def __init__(self, max_memory_per_port_gb: float) -> None:
        self.max_memory_per_port_gb = max_memory_per_port_gb
        super().__init__(f"max_memory_per_port must be > 0 GB, got {max_memory_per_port_gb!r}")


class RedisShardStartupError(RuntimeError):
    def __init__(self, port: int, last_error: Exception | None) -> None:
        self.port = port
        self.last_error = last_error
        super().__init__(f"Redis shard on port {port} failed to start: {last_error}")


class RedisClusterManager:
    def __init__(
        self,
        ports: Sequence[int],
        max_memory_per_port: float,
        *,
        host: str = "127.0.0.1",
        startup_timeout_s: float = 15.0,
        log_dir: str | Path | None = None,
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
        """
        normalized_ports = [int(p) for p in ports]
        if not normalized_ports:
            raise EmptyRedisPortsError
        if len(set(normalized_ports)) != len(normalized_ports):
            raise DuplicateRedisPortsError(normalized_ports)
        for port in normalized_ports:
            if port <= 0 or port > 65535:
                raise InvalidRedisPortError(port)

        if max_memory_per_port <= 0:
            raise InvalidRedisMaxMemoryError(max_memory_per_port)

        self.redis_processes: list[subprocess.Popen[Any]] = []
        self.redis_clients: list[redis.Redis] = []
        self._log_files: list[Any] = []

        self.host = host
        self.ports = normalized_ports
        self.max_memory_per_port_gb = float(max_memory_per_port)
        self.max_memory_per_port_bytes = self._gb_to_bytes(self.max_memory_per_port_gb)
        self.startup_timeout_s = float(startup_timeout_s)
        self.log_dir = Path(log_dir) if log_dir is not None else None

    @staticmethod
    def _gb_to_bytes(gb: float) -> int:
        # Use binary GB (GiB) to match common memory accounting: 1 GB = 1024^3 bytes.
        return max(1, int(gb * (1024**3)))

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.stop_redis_cluster()

    def __enter__(self) -> RedisClusterManager:
        self.start_redis_cluster()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        self.stop_redis_cluster()

    def _wait_until_healthy(self, client: redis.Redis, *, port: int) -> None:
        deadline = time.monotonic() + self.startup_timeout_s
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                client.ping()
            except redis.exceptions.RedisError as e:
                last_error = e
                time.sleep(0.1)
            else:
                return
        raise RedisShardStartupError(port=port, last_error=last_error)

    def start_redis_cluster(self) -> bool:
        """
        Start multiple Redis server instances

        Returns:
            bool: True if all Redis servers started successfully, False otherwise.
        """
        self.stop_redis_cluster()

        redis_server_path = shutil.which("redis-server")
        if redis_server_path is None:
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
                    [
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
                    ],
                    stdout=stdout,
                    stderr=stderr,
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
                self._wait_until_healthy(redis_client, port=port)
                self.redis_clients.append(redis_client)

                print(f"Redis shard {i} started on port {port}")

        except Exception as e:
            print(f"Failed to start Redis cluster: {e}")
            self.stop_redis_cluster()
            print(e)
            return False
        else:
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
