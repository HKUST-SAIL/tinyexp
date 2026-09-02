#!/usr/bin/env python3
"""Run a command with a Redis cache lifecycle owned by this wrapper.

The wrapper reads ``redis_cfg`` overrides from the target command, starts
standalone Redis for single-node jobs or a Redis Cluster for multi-node jobs,
injects the final Redis connection settings back into the child command as Hydra
overrides, waits for the child command, and then stops Redis processes it owns.
In multi-node mode, the HTTP rendezvous is used only during startup; each
wrapper cleans up its locally owned Redis as soon as its child command exits.

For multi-node jobs, pass ``--node-count``, ``--node-rank``, and ``--head-addr``
or provide matching StepFun env/Hydra values through ``NODE_RANK`` and
``redis_cfg``. Rank 0 hosts the HTTP rendezvous server on ``0.0.0.0``;
all ranks register through ``--head-addr``.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

from omegaconf import OmegaConf

from tinyexp.exp_mixins import RedisCfgMixin
from tinyexp.utils.redis_utils import RedisClusterManager, wait_for_redis_cluster

REDIS_RENDEZVOUS_PORT = 26379
REDIS_RENDEZVOUS_TIMEOUT_S = 600
_REDIS_CONNECTION_OVERRIDE_PREFIXES = (
    "redis_cfg.redis_cluster_host=",
    "redis_cfg.redis_cluster_ports=",
    "redis_cfg.redis_rendezvous_world_size=",
)

CHILD_PROCESS: subprocess.Popen[Any] | None = None


class _RedisLifecycle:
    """Own Redis resources created by one wrapper invocation."""

    def __init__(self) -> None:
        self._managers: list[RedisClusterManager] = []
        self._rendezvous_server: ThreadingHTTPServer | None = None

    @property
    def started_nodes(self) -> list[tuple[str, int]]:
        return [(manager.host, port) for manager in self._managers for port in manager.ports]

    def start_nodes(
        self,
        *,
        host: str,
        ports: list[int],
        max_memory_per_port: float,
        cluster_enabled: bool,
    ) -> bool:
        manager = RedisClusterManager(
            ports=ports,
            max_memory_per_port=max_memory_per_port,
            host=host,
            cluster_enabled=cluster_enabled,
        )
        if not manager.start_redis_cluster():
            return False
        self._managers.append(manager)
        return True

    def own_rendezvous_server(self, server: ThreadingHTTPServer) -> None:
        self._rendezvous_server = server

    def close(self) -> None:
        if self._rendezvous_server is not None:
            with suppress(Exception):
                self._rendezvous_server.shutdown()
            with suppress(Exception):
                self._rendezvous_server.server_close()
            self._rendezvous_server = None

        for manager in reversed(self._managers):
            with suppress(Exception):
                manager.stop_redis_cluster()
        self._managers.clear()


REDIS_LIFECYCLE: _RedisLifecycle | None = None


def uint(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer: {value}") from exc  # noqa: TRY003
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"invalid unsigned integer: {value}")  # noqa: TRY003
    return parsed


def positive_int(value: str) -> int:
    parsed = uint(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1: {value}")  # noqa: TRY003
    return parsed


def env_uint(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed >= 0 else default


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tinyexp-run-with-redis",
        description="Start Redis, inject redis_cfg overrides, then run <command>.",
    )
    parser.add_argument(
        "--node-count",
        type=positive_int,
        default=None,
        help="Total Redis rendezvous nodes. Default: redis_cfg.redis_rendezvous_world_size.",
    )
    parser.add_argument(
        "--node-rank",
        type=uint,
        default=env_uint("NODE_RANK", 0),
        help="Current node rank. Rank 0 hosts rendezvous. Default: NODE_RANK or 0.",
    )
    parser.add_argument(
        "--head-addr",
        default="",
        help="Rendezvous address all nodes use for multi-node Redis. Default: redis_cfg.redis_cluster_host.",
    )
    parser.add_argument(
        "--rendezvous-port",
        type=positive_int,
        default=REDIS_RENDEZVOUS_PORT,
        help=f"HTTP rendezvous port. Default: {REDIS_RENDEZVOUS_PORT}.",
    )
    parser.add_argument(
        "--wait-timeout",
        type=positive_int,
        default=REDIS_RENDEZVOUS_TIMEOUT_S,
        help=f"Startup wait timeout. Default: {REDIS_RENDEZVOUS_TIMEOUT_S}.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run after --.")
    args = parser.parse_args(argv)
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("missing command.")
    return args


def _load_python_file(module_path: Path):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(f"_tinyexp_run_with_redis_{module_path.stem}", module_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_experiment_module(argv: list[str]):  # type: ignore[no-untyped-def]
    if len(argv) >= 3 and Path(argv[0]).name.startswith("python") and argv[1] == "-m":
        return importlib.import_module(argv[2])

    for arg in argv:
        module_path = Path(arg)
        if module_path.suffix == ".py" and module_path.is_file():
            return _load_python_file(module_path.resolve())

    return None


def _restore_signal_handlers(previous_handlers: dict[int, Any]) -> None:
    for signum, previous_handler in previous_handlers.items():
        signal.signal(signum, previous_handler)


def main(argv: list[str]) -> int:
    global CHILD_PROCESS, REDIS_LIFECYCLE

    args = parse_args(argv)
    redis_cfg = build_redis_cfg(args.command)
    configured_ports = [int(port) for port in redis_cfg.redis_cluster_ports]
    if not configured_ports:
        print("redis_cfg.redis_cluster_ports must not be empty", file=sys.stderr)
        return 2

    world_size = args.node_count or int(redis_cfg.redis_rendezvous_world_size)
    if world_size < 1:
        print("redis_cfg.redis_rendezvous_world_size must be >= 1", file=sys.stderr)
        return 2
    # node_rank only matters for multi-node cluster mode; single-node local Redis
    # (world_size == 1) ignores it, so don't reject an inherited NODE_RANK env here.
    if world_size > 1 and args.node_rank >= world_size:
        print(
            f"--node-rank must be < --node-count, got {args.node_rank} >= {world_size}",
            file=sys.stderr,
        )
        return 2

    head_addr = args.head_addr or str(redis_cfg.redis_cluster_host)
    if world_size > 1 and head_addr in {"", "127.0.0.1", "localhost", "::1"}:
        print("--head-addr is required for multi-node Redis jobs", file=sys.stderr)
        return 2

    lifecycle = _RedisLifecycle()
    REDIS_LIFECYCLE = lifecycle
    previous_signal_handlers = {
        signal.SIGTERM: signal.getsignal(signal.SIGTERM),
        signal.SIGINT: signal.getsignal(signal.SIGINT),
    }
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    startup_node: tuple[str, list[int]] | None = None
    try:
        redis_status = True
        if redis_cfg.redis_cache_enabled and world_size > 1:
            startup_node = start_rendezvous_redis_cluster(
                redis_cfg,
                world_size=world_size,
                node_rank=args.node_rank,
                head_addr=head_addr,
                rendezvous_port=args.rendezvous_port,
                timeout_s=args.wait_timeout,
                lifecycle=lifecycle,
            )
            redis_status = startup_node is not None
        elif redis_cfg.redis_cache_enabled:
            startup_node = start_local_redis(redis_cfg, lifecycle)
            redis_status = startup_node is not None

        print(f"Redis status:\033[32m{redis_status}\033[0m", flush=True)
        if not redis_status:
            return 1

        command = list(args.command)
        if startup_node is not None:
            startup_host, startup_ports = startup_node
            overrides = [
                f"redis_cfg.redis_cluster_host={startup_host}",
                f"redis_cfg.redis_cluster_ports=[{','.join(str(port) for port in startup_ports)}]",
                f"redis_cfg.redis_rendezvous_world_size={world_size}",
            ]
            command = [arg for arg in command if not arg.startswith(_REDIS_CONNECTION_OVERRIDE_PREFIXES)]
            command = [*command, *overrides]

        print(f"Running command: {shlex.join(command)}", flush=True)
        CHILD_PROCESS = subprocess.Popen(  # noqa: S603
            command,
            env=os.environ.copy(),
            start_new_session=True,
        )
        exit_code = CHILD_PROCESS.wait()
        return exit_code
    finally:
        CHILD_PROCESS = None
        lifecycle.close()
        REDIS_LIFECYCLE = None
        _restore_signal_handlers(previous_signal_handlers)


def build_redis_cfg(argv: list[str]) -> Any:
    exp: Any | None = None
    module = _load_experiment_module(argv)
    if module is not None:
        exp_classes = [
            obj
            for obj in vars(module).values()
            if inspect.isclass(obj)
            and obj.__module__ == module.__name__
            and issubclass(obj, RedisCfgMixin)
            and obj is not RedisCfgMixin
        ]
        if len(exp_classes) == 1:
            exp = cast(Any, exp_classes[0]())
            exp.exp_class = f"{exp_classes[0].__module__}.{exp_classes[0].__qualname__}"

    overrides = [arg for arg in argv if arg.startswith("redis_cfg.")]
    if exp is not None:
        if overrides:
            exp.set_cfg(OmegaConf.from_dotlist(overrides))
        return exp.redis_cfg

    print(
        "Could not infer experiment config; using RedisCfgMixin defaults",
        file=sys.stderr,
    )
    cfg = OmegaConf.structured(RedisCfgMixin())
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg.redis_cfg


def start_local_redis(redis_cfg: Any, lifecycle: _RedisLifecycle) -> tuple[str, list[int]] | None:
    host = str(redis_cfg.redis_cluster_host)
    ports = [int(port) for port in redis_cfg.redis_cluster_ports]
    max_memory_per_port = float(redis_cfg.redis_cache_max_memory) / len(ports)
    if not lifecycle.start_nodes(
        host=host,
        ports=ports,
        max_memory_per_port=max_memory_per_port,
        cluster_enabled=False,
    ):
        return None
    return host, ports


def start_rendezvous_redis_cluster(
    redis_cfg: Any,
    *,
    world_size: int,
    node_rank: int,
    head_addr: str,
    rendezvous_port: int,
    timeout_s: int,
    lifecycle: _RedisLifecycle,
) -> tuple[str, list[int]] | None:
    ports = [int(port) for port in redis_cfg.redis_cluster_ports]
    node_count = world_size * len(ports)
    if node_count < 3:
        print("Redis Cluster requires at least 3 master nodes", file=sys.stderr)
        return None
    redis_cli_path = shutil.which("redis-cli") if node_rank == 0 else None
    if node_rank == 0 and redis_cli_path is None:
        print(
            "redis-cli command not found in PATH; install Redis before enabling Redis Cluster",
            file=sys.stderr,
        )
        return None

    node_host = resolve_node_host(head_addr, rendezvous_port)
    if node_host in {"127.0.0.1", "localhost", "::1"}:
        print(
            f"Redis Cluster node host must not be loopback in multi-node mode: {node_host}",
            file=sys.stderr,
        )
        return None

    max_memory_per_port = float(redis_cfg.redis_cache_max_memory) / node_count
    if not lifecycle.start_nodes(
        host=node_host,
        ports=ports,
        max_memory_per_port=max_memory_per_port,
        cluster_enabled=True,
    ):
        return None

    server = (
        start_rendezvous_server(
            head_addr,
            rendezvous_port,
            world_size,
            redis_cli_path,
            timeout_s=timeout_s,
        )
        if node_rank == 0
        else None
    )
    if node_rank == 0:
        if server is None:
            return None
        lifecycle.own_rendezvous_server(server)

    return register_redis_node(
        head_addr=head_addr,
        rendezvous_port=rendezvous_port,
        node_host=node_host,
        ports=ports,
        timeout_s=timeout_s,
    )


def resolve_node_host(head_addr: str, rendezvous_port: int) -> str:
    for env_name in ("SOCKET_IP", "POD_IP"):
        value = os.environ.get(env_name)
        if value:
            return value

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock, suppress(OSError):
        sock.connect((head_addr, rendezvous_port))
        value = sock.getsockname()[0]
        if value:
            return value

    with suppress(OSError):
        value = socket.gethostbyname(socket.gethostname())
        if value:
            return value
    return "127.0.0.1"


def start_rendezvous_server(  # noqa: C901
    head_addr: str,
    rendezvous_port: int,
    world_size: int,
    redis_cli_path: str | None,
    timeout_s: float = REDIS_RENDEZVOUS_TIMEOUT_S,
) -> ThreadingHTTPServer | None:
    if timeout_s <= 0:
        raise ValueError("rendezvous timeout must be greater than 0")  # noqa: TRY003

    nodes: dict[str, tuple[str, list[int]]] = {}
    lock = threading.Lock()
    ready: dict[str, Any] = {}
    error = ""
    creation_in_progress = False
    startup_deadline = time.monotonic() + timeout_s

    def response_locked() -> dict[str, Any]:
        if error:
            return {"status": "error", "message": error}
        if ready:
            return {"status": "ready", **ready}
        return {
            "status": "waiting",
            "registered": len(nodes),
            "world_size": world_size,
        }

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: C901
            nonlocal creation_in_progress, error
            if self.path != "/register":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                host = str(payload["host"])
                worker_ports = [int(port) for port in payload["ports"]]
                if not host or not worker_ports:
                    self.send_error(400, "invalid registration payload")
                    return
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                self.send_error(400, f"invalid registration payload: {exc}")
                return

            node_key = f"{host}:{','.join(str(port) for port in worker_ports)}"
            creation_request: tuple[list[str], str, list[int], float] | None = None
            response: dict[str, Any] | None = None

            with lock:
                nodes[node_key] = (host, worker_ports)
                if error or ready or creation_in_progress or len(nodes) < world_size:
                    response = response_locked()
                else:
                    remaining = startup_deadline - time.monotonic()
                    if remaining <= 0:
                        error = f"Redis Cluster startup timed out after {timeout_s}s"
                        response = response_locked()
                    else:
                        creation_in_progress = True
                        startup_host, startup_ports = next(iter(nodes.values()))
                        addresses = [
                            f"{node_host}:{port}" for node_host, node_ports in nodes.values() for port in node_ports
                        ]
                        creation_request = (addresses, startup_host, startup_ports, remaining)

            if creation_request is not None:
                addresses, startup_host, startup_ports, command_timeout = creation_request
                creation_error: str | None = None
                creation_result: tuple[str, list[int]] | None = None
                create_cmd = [
                    redis_cli_path or "redis-cli",
                    "--cluster",
                    "create",
                    *addresses,
                    "--cluster-replicas",
                    "0",
                    "--cluster-yes",
                ]
                try:
                    result = subprocess.run(  # noqa: S603
                        create_cmd,
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=command_timeout,
                    )
                    if result.returncode != 0:
                        creation_error = f"redis-cli cluster create failed: {result.stderr or result.stdout}"
                    else:
                        readiness_timeout = startup_deadline - time.monotonic()
                        if readiness_timeout <= 0:
                            creation_error = f"Redis Cluster startup timed out after {timeout_s}s"
                        elif wait_for_redis_cluster(
                            startup_host,
                            startup_ports,
                            timeout_s=readiness_timeout,
                        ):
                            creation_result = (startup_host, startup_ports)
                        else:
                            creation_error = f"Redis Cluster did not become ready at {startup_host}:{startup_ports}"
                except subprocess.TimeoutExpired as exc:
                    creation_error = f"redis-cli cluster create timed out after {command_timeout:.3f}s"
                    print(f"{creation_error}: {exc}", file=sys.stderr)
                except Exception as exc:
                    creation_error = f"Redis Cluster startup failed: {exc}"

                with lock:
                    creation_in_progress = False
                    if creation_result is not None and creation_error is None:
                        ready.update(
                            {
                                "startup_host": creation_result[0],
                                "startup_ports": creation_result[1],
                            }
                        )
                    else:
                        error = creation_error or "Redis Cluster startup failed"
                    response = response_locked()

            if response is None:
                response = {"status": "error", "message": "Redis Cluster startup did not produce a response"}
            self.write_json(response)

        def write_json(self, response: dict[str, Any]) -> None:
            body = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            with suppress(BrokenPipeError, ConnectionResetError):
                self.wfile.write(body)

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    try:
        server = ThreadingHTTPServer(("0.0.0.0", rendezvous_port), Handler)  # noqa: S104
    except OSError as exc:
        print(
            f"Redis rendezvous failed to listen on 0.0.0.0:{rendezvous_port}: {exc}",
            file=sys.stderr,
        )
        return None

    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(
        f"Redis rendezvous master listening on 0.0.0.0:{rendezvous_port} for {head_addr}",
        flush=True,
    )
    return server


def register_redis_node(
    *,
    head_addr: str,
    rendezvous_port: int,
    node_host: str,
    ports: list[int],
    timeout_s: int,
) -> tuple[str, list[int]] | None:
    url = f"http://{head_addr}:{rendezvous_port}/register"
    payload = json.dumps({"host": node_host, "ports": ports}).encode()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            request = urllib.request.Request(  # noqa: S310
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=min(5.0, remaining)) as response:  # noqa: S310
                result = json.loads(response.read())
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
            continue

        if result["status"] == "ready":
            startup_ports = [int(port) for port in result["startup_ports"]]
            print(
                f"Redis Cluster startup node: {result['startup_host']}:{startup_ports[0]}",
                flush=True,
            )
            return str(result["startup_host"]), startup_ports
        if result["status"] == "error":
            print(f"Redis rendezvous failed: {result['message']}", file=sys.stderr)
            return None
        print(
            f"Redis rendezvous waiting: {result['registered']}/{result['world_size']} workers registered",
            flush=True,
        )
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))

    print(f"Redis rendezvous timed out after {timeout_s}s", file=sys.stderr)
    return None


def signal_process_group(process: subprocess.Popen[Any], signum: int) -> None:
    pid = getattr(process, "pid", None)
    killpg = getattr(os, "killpg", None)
    if pid is not None and killpg is not None:
        with suppress(ProcessLookupError, PermissionError):
            killpg(int(pid), signum)
            return
    with suppress(ProcessLookupError, PermissionError):
        process.send_signal(signum)


def kill_process_group(process: subprocess.Popen[Any]) -> None:
    pid = getattr(process, "pid", None)
    killpg = getattr(os, "killpg", None)
    if pid is not None and killpg is not None:
        with suppress(ProcessLookupError, PermissionError):
            killpg(int(pid), signal.SIGKILL)
            return
    with suppress(ProcessLookupError, PermissionError):
        process.kill()


def stop_child_process(process: subprocess.Popen[Any], signum: int, timeout_s: int = 30) -> None:
    if process.poll() is not None:
        return
    signal_process_group(process, signum)
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        if process.poll() is None:
            kill_process_group(process)
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=5)


def handle_signal(signum: int, _frame: object) -> None:
    if CHILD_PROCESS is not None:
        stop_child_process(CHILD_PROCESS, signum)
    if REDIS_LIFECYCLE is not None:
        REDIS_LIFECYCLE.close()
    raise SystemExit(128 + signum)


def cli() -> None:
    raise SystemExit(main(sys.argv[1:]))


if __name__ == "__main__":
    cli()
