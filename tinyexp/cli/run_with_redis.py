#!/usr/bin/env python3
"""Run a command with a Redis cache lifecycle owned by this wrapper.

The wrapper reads ``redis_cache_cfg`` overrides from the target command, starts
standalone Redis for single-node jobs or a Redis Cluster for multi-node jobs,
injects the final Redis connection settings back into the child command as Hydra
overrides, waits for the child command, and then stops Redis processes it owns.

For multi-node jobs, pass ``--node-count``, ``--node-rank``, and ``--head-addr``
or provide matching StepFun env/Hydra values through ``NODE_RANK`` and
``redis_cache_cfg``. Rank 0 hosts the HTTP rendezvous server on ``0.0.0.0``;
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

import redis
from omegaconf import OmegaConf

from tinyexp.exp_mixins import RedisCfgMixin

REDIS_RENDEZVOUS_PORT = 26379
REDIS_RENDEZVOUS_TIMEOUT_S = 600
_REDIS_CONNECTION_OVERRIDE_PREFIXES = (
    "redis_cache_cfg.redis_cluster_host=",
    "redis_cache_cfg.redis_cluster_ports=",
    "redis_cache_cfg.redis_rendezvous_world_size=",
)

STARTED_NODES: list[tuple[str, int]] = []
CHILD_PROCESS: subprocess.Popen[Any] | None = None


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
        description="Start Redis, inject redis_cache_cfg overrides, then run <command>.",
    )
    parser.add_argument(
        "--node-count",
        type=positive_int,
        default=None,
        help="Total Redis rendezvous nodes. Default: redis_cache_cfg.redis_rendezvous_world_size.",
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
        help="Rendezvous address all nodes use for multi-node Redis. Default: redis_cache_cfg.redis_cluster_host.",
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


def main(argv: list[str]) -> int:
    global CHILD_PROCESS, STARTED_NODES

    args = parse_args(argv)
    redis_cache_cfg = build_redis_cache_cfg(args.command)
    configured_ports = [int(port) for port in redis_cache_cfg.redis_cluster_ports]
    if not configured_ports:
        print("redis_cache_cfg.redis_cluster_ports must not be empty", file=sys.stderr)
        return 2

    world_size = args.node_count or int(redis_cache_cfg.redis_rendezvous_world_size)
    if world_size < 1:
        print("redis_cache_cfg.redis_rendezvous_world_size must be >= 1", file=sys.stderr)
        return 2
    if args.node_rank >= world_size:
        print(
            f"--node-rank must be < --node-count, got {args.node_rank} >= {world_size}",
            file=sys.stderr,
        )
        return 2

    head_addr = args.head_addr or str(redis_cache_cfg.redis_cluster_host)
    if world_size > 1 and head_addr in {"", "127.0.0.1", "localhost", "::1"}:
        print("--head-addr is required for multi-node Redis jobs", file=sys.stderr)
        return 2

    STARTED_NODES = []
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    startup_node: tuple[str, list[int]] | None = None
    try:
        redis_status = True
        if redis_cache_cfg.redis_cache_enabled and world_size > 1:
            startup_node = start_rendezvous_redis_cluster(
                redis_cache_cfg,
                world_size=world_size,
                node_rank=args.node_rank,
                head_addr=head_addr,
                rendezvous_port=args.rendezvous_port,
                timeout_s=args.wait_timeout,
                started_nodes=STARTED_NODES,
            )
            redis_status = startup_node is not None
        elif redis_cache_cfg.redis_cache_enabled:
            startup_node = start_local_redis(redis_cache_cfg, STARTED_NODES)
            redis_status = startup_node is not None

        print(f"Redis status:\033[32m{redis_status}\033[0m", flush=True)
        if not redis_status:
            return 1

        command = list(args.command)
        if startup_node is not None:
            startup_host, startup_ports = startup_node
            overrides = [
                f"redis_cache_cfg.redis_cluster_host={startup_host}",
                f"redis_cache_cfg.redis_cluster_ports=[{','.join(str(port) for port in startup_ports)}]",
                f"redis_cache_cfg.redis_rendezvous_world_size={world_size}",
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
        if startup_node is not None and world_size > 1:
            wait_for_rendezvous_finish(
                head_addr=head_addr,
                rendezvous_port=args.rendezvous_port,
                node_id=redis_node_id(STARTED_NODES),
                timeout_s=args.wait_timeout,
            )
        return exit_code
    finally:
        CHILD_PROCESS = None
        cleanup_started_nodes(STARTED_NODES)
        STARTED_NODES = []


def build_redis_cache_cfg(argv: list[str]) -> Any:
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

    overrides = [arg for arg in argv if arg.startswith("redis_cache_cfg.")]
    if exp is not None:
        if overrides:
            exp.set_cfg(OmegaConf.from_dotlist(overrides))
        return exp.redis_cache_cfg

    print(
        "Could not infer experiment config; using RedisCfgMixin defaults",
        file=sys.stderr,
    )
    cfg = OmegaConf.structured(RedisCfgMixin())
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg.redis_cache_cfg


def start_local_redis(redis_cache_cfg: Any, started_nodes: list[tuple[str, int]]) -> tuple[str, list[int]] | None:
    host = str(redis_cache_cfg.redis_cluster_host)
    ports = [int(port) for port in redis_cache_cfg.redis_cluster_ports]
    max_memory_bytes = max(1, int((float(redis_cache_cfg.redis_cache_max_memory) / len(ports)) * (1024**3)))
    local_started_nodes: list[tuple[str, int]] = []
    for port in ports:
        if not start_redis_node(
            host=host,
            port=port,
            max_memory_bytes=max_memory_bytes,
            cluster_enabled=False,
        ):
            cleanup_started_nodes(local_started_nodes)
            return None
        local_started_nodes.append((host, port))
        print(f"Redis server started on {host}:{port}", flush=True)
    started_nodes.extend(local_started_nodes)
    return host, ports


def start_rendezvous_redis_cluster(
    redis_cache_cfg: Any,
    *,
    world_size: int,
    node_rank: int,
    head_addr: str,
    rendezvous_port: int,
    timeout_s: int,
    started_nodes: list[tuple[str, int]],
) -> tuple[str, list[int]] | None:
    ports = [int(port) for port in redis_cache_cfg.redis_cluster_ports]
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

    max_memory_bytes = max(1, int((float(redis_cache_cfg.redis_cache_max_memory) / node_count) * (1024**3)))
    local_started_nodes: list[tuple[str, int]] = []
    for port in ports:
        if not start_redis_node(
            host=node_host,
            port=port,
            max_memory_bytes=max_memory_bytes,
            cluster_enabled=True,
        ):
            cleanup_started_nodes(local_started_nodes)
            return None
        local_started_nodes.append((node_host, port))
        print(f"Redis Cluster node started on {node_host}:{port}", flush=True)
    started_nodes.extend(local_started_nodes)

    server = start_rendezvous_server(head_addr, rendezvous_port, world_size, redis_cli_path) if node_rank == 0 else None
    if node_rank == 0 and server is None:
        return None

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
    head_addr: str, rendezvous_port: int, world_size: int, redis_cli_path: str | None
) -> ThreadingHTTPServer | None:
    nodes: dict[str, tuple[str, list[int]]] = {}
    finished_nodes: set[str] = set()
    lock = threading.Lock()
    ready: dict[str, Any] = {}
    error = ""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            nonlocal error
            if self.path == "/finish":
                self.handle_finish()
                return
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

            with lock:
                nodes[f"{host}:{','.join(str(port) for port in worker_ports)}"] = (
                    host,
                    worker_ports,
                )
                if not ready and not error and len(nodes) >= world_size:
                    addresses = [
                        f"{node_host}:{port}" for node_host, node_ports in nodes.values() for port in node_ports
                    ]
                    create_cmd = [
                        redis_cli_path or "redis-cli",
                        "--cluster",
                        "create",
                        *addresses,
                        "--cluster-replicas",
                        "0",
                        "--cluster-yes",
                    ]
                    result = subprocess.run(create_cmd, check=False, capture_output=True, text=True)  # noqa: S603
                    if result.returncode != 0:
                        error = f"redis-cli cluster create failed: {result.stderr or result.stdout}"
                    else:
                        startup_host, startup_ports = next(iter(nodes.values()))
                        if wait_for_redis_cluster(startup_host, startup_ports):
                            ready.update(
                                {
                                    "startup_host": startup_host,
                                    "startup_ports": startup_ports,
                                }
                            )
                        else:
                            error = f"Redis Cluster did not become ready at {startup_host}:{startup_ports}"

                if error:
                    response: dict[str, Any] = {"status": "error", "message": error}
                elif ready:
                    response = {"status": "ready", **ready}
                else:
                    response = {
                        "status": "waiting",
                        "registered": len(nodes),
                        "world_size": world_size,
                    }

            self.write_json(response)

        def handle_finish(self) -> None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                node_id = str(payload["node_id"])
                if not node_id:
                    self.send_error(400, "invalid finish payload")
                    return
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                self.send_error(400, f"invalid finish payload: {exc}")
                return

            with lock:
                finished_nodes.add(node_id)
                if len(finished_nodes) >= world_size:
                    response = {
                        "status": "done",
                        "finished": len(finished_nodes),
                        "world_size": world_size,
                    }
                else:
                    response = {
                        "status": "waiting",
                        "finished": len(finished_nodes),
                        "world_size": world_size,
                    }
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
        try:
            request = urllib.request.Request(  # noqa: S310
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310
                result = json.loads(response.read())
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            time.sleep(1)
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
        time.sleep(1)

    print(f"Redis rendezvous timed out after {timeout_s}s", file=sys.stderr)
    return None


def redis_node_id(started_nodes: list[tuple[str, int]]) -> str:
    return ";".join(f"{host}:{port}" for host, port in sorted(started_nodes))


def wait_for_rendezvous_finish(*, head_addr: str, rendezvous_port: int, node_id: str, timeout_s: int) -> bool:
    if not node_id:
        return True
    url = f"http://{head_addr}:{rendezvous_port}/finish"
    payload = json.dumps({"node_id": node_id}).encode()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            request = urllib.request.Request(  # noqa: S310
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310
                result = json.loads(response.read())
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            time.sleep(1)
            continue
        if result["status"] == "done":
            print("Redis rendezvous finish barrier complete", flush=True)
            return True
        print(
            f"Redis rendezvous finish waiting: {result['finished']}/{result['world_size']} workers finished",
            flush=True,
        )
        time.sleep(1)
    print(f"Redis rendezvous finish barrier timed out after {timeout_s}s", file=sys.stderr)
    return False


def wait_for_redis_cluster(startup_host: str, startup_ports: list[int], timeout_s: int = 30) -> bool:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        cluster = redis.RedisCluster(
            startup_nodes=[redis.cluster.ClusterNode(startup_host, int(port)) for port in startup_ports],
            decode_responses=False,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        try:
            cluster.ping()
            ready = True
            for index in range(16):
                key = f"tinyexp:redis-cluster-readiness:{index}"
                cluster.set(key, "ok", ex=60)
                if cluster.get(key) != b"ok":
                    ready = False
                    break
        except redis.exceptions.RedisError as exc:
            last_error = exc
            time.sleep(1)
        else:
            if ready:
                return True
            last_error = RuntimeError("readiness value mismatch")
            time.sleep(1)
        finally:
            with suppress(Exception):
                cluster.close()
    print(f"Redis Cluster readiness check failed: {last_error}", file=sys.stderr)
    return False


def start_redis_node(*, host: str, port: int, max_memory_bytes: int, cluster_enabled: bool) -> bool:
    shutdown_redis(host, port)
    pidfile = redis_pidfile(port)
    log_file = Path(f"/tmp/tinyexp-redis-{int(port)}.log")  # noqa: S108
    if cluster_enabled:
        with suppress(FileNotFoundError):
            Path(f"/tmp/tinyexp-redis-{int(port)}.nodes.conf").unlink()  # noqa: S108

    command = [
        "redis-server",
        "--bind",
        host,
        "--protected-mode",
        "no",
        "--port",
        str(int(port)),
        "--daemonize",
        "no",
        "--dir",
        "/tmp",  # noqa: S108
        "--save",
        "",
        "--appendonly",
        "no",
        "--maxmemory",
        str(int(max_memory_bytes)),
    ]
    if cluster_enabled:
        command.extend(
            [
                "--cluster-enabled",
                "yes",
                "--cluster-config-file",
                f"tinyexp-redis-{int(port)}.nodes.conf",
                "--cluster-node-timeout",
                "5000",
                "--cluster-announce-ip",
                host,
                "--cluster-announce-port",
                str(int(port)),
                "--cluster-announce-bus-port",
                str(int(port) + 10000),
            ]
        )

    try:
        with log_file.open("ab") as output:
            process = subprocess.Popen(  # noqa: S603
                command,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        pidfile.write_text(str(process.pid))
    except OSError as exc:
        print(f"failed to start redis-server on {host}:{port}: {exc}", file=sys.stderr)
        return False

    if wait_for_redis(host, port):
        return True
    kill_pidfile(pidfile)
    return False


def cleanup_started_nodes(started_nodes: list[tuple[str, int]]) -> None:
    for host, port in started_nodes:
        shutdown_redis(host, int(port))
        kill_matching_redis(host, int(port))


def shutdown_redis(host: str, port: int) -> None:
    client = redis.Redis(host=host, port=int(port), socket_connect_timeout=0.5, socket_timeout=0.5)
    with suppress(redis.exceptions.RedisError):
        client.shutdown(nosave=True)
    with suppress(Exception):
        client.close()
    kill_pidfile(redis_pidfile(port))


def redis_pidfile(port: int) -> Path:
    return Path(f"/tmp/tinyexp-redis-{int(port)}.pid")  # noqa: S108


def kill_pidfile(pidfile: Path) -> None:
    try:
        pid = int(pidfile.read_text().strip())
    except (FileNotFoundError, ValueError):
        pid = 0
    if pid:
        with suppress(ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGKILL)
    with suppress(FileNotFoundError):
        pidfile.unlink()


def kill_matching_redis(host: str, port: int) -> None:
    with suppress(subprocess.TimeoutExpired):
        result = subprocess.run(  # noqa: S603
            ["ps", "-eo", "pid=,args="],  # noqa: S607
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
        )
        target = f"redis-server {host}:{int(port)}"
        for line in result.stdout.splitlines():
            if target not in line:
                continue
            with suppress(ValueError, ProcessLookupError, PermissionError):
                os.kill(int(line.split(None, 1)[0]), signal.SIGKILL)


def wait_for_redis(host: str, port: int) -> bool:
    client = redis.StrictRedis(host=host, port=int(port), socket_connect_timeout=1, socket_timeout=1)
    deadline = time.monotonic() + 15
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            client.ping()
        except redis.exceptions.RedisError as exc:
            last_error = exc
            time.sleep(0.2)
        else:
            client.close()
            return True
    client.close()
    print(f"Redis node {host}:{port} failed health check: {last_error}", file=sys.stderr)
    return False


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
    cleanup_started_nodes(STARTED_NODES)
    raise SystemExit(128 + signum)


def cli() -> None:
    raise SystemExit(main(sys.argv[1:]))


if __name__ == "__main__":
    cli()
