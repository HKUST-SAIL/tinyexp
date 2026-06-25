from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import os
import shlex
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
from typing import Any

import redis
from omegaconf import OmegaConf

from tinyexp import RedisCfgMixin

REDIS_RENDEZVOUS_PORT = 26379
REDIS_RENDEZVOUS_TIMEOUT_S = 600


def main(argv: list[str]) -> int:  # noqa: C901
    if argv[:1] == ["--"]:
        argv = argv[1:]
    if not argv:
        print("Usage: python scripts/run_with_redis.py -- <command> [args...]", file=sys.stderr)
        return 2

    redis_cache_cfg = build_redis_cache_cfg(argv)
    configured_ports = [int(port) for port in redis_cache_cfg.redis_cluster_ports]
    if not configured_ports:
        print("redis_cache_cfg.redis_cluster_ports must not be empty", file=sys.stderr)
        return 2

    started_nodes: list[tuple[str, int]] = []
    env = os.environ.copy()

    try:
        redis_status = True
        world_size = int(redis_cache_cfg.redis_rendezvous_world_size)
        if world_size < 1:
            print("redis_cache_cfg.redis_rendezvous_world_size must be >= 1", file=sys.stderr)
            return 2

        if redis_cache_cfg.redis_cache_enabled and world_size > 1:
            redis_status = start_rendezvous_redis_cluster(redis_cache_cfg, world_size, started_nodes, env)
        elif redis_cache_cfg.redis_cache_enabled:
            redis_status = start_local_redis(redis_cache_cfg, started_nodes, env)

        print(f"Redis status:\033[32m{redis_status}\033[0m", flush=True)
        if not redis_status:
            return 1

        redis_host = env.get("TINYEXP_REDIS_CLUSTER_HOST")
        redis_ports = env.get("TINYEXP_REDIS_CLUSTER_PORTS")
        if redis_host and redis_ports:
            overrides = [
                f"redis_cache_cfg.redis_cluster_host={redis_host}",
                f"redis_cache_cfg.redis_cluster_ports=[{redis_ports}]",
            ]
            insert_at = len(argv)
            app_arg_start = 3 if len(argv) >= 3 and argv[1] == "-m" else 2
            for index, arg in enumerate(argv[app_arg_start:], start=app_arg_start):
                if arg.startswith("--"):
                    insert_at = index
                    break
            argv = [*argv[:insert_at], *overrides, *argv[insert_at:]]

        print(f"Running command: {shlex.join(argv)}", flush=True)
        process = subprocess.Popen(argv, env=env)  # noqa: S603
        try:
            return process.wait()
        except KeyboardInterrupt:
            process.send_signal(signal.SIGINT)
            return process.wait()
    finally:
        cleanup_started_nodes(started_nodes)


def build_redis_cache_cfg(argv: list[str]) -> Any:
    exp = None
    module = None
    if len(argv) >= 2 and Path(argv[0]).name.startswith("python"):
        if argv[1] == "-m" and len(argv) >= 3:
            module = importlib.import_module(argv[2])
        elif argv[1].endswith(".py"):
            module_path = Path(argv[1]).resolve()
            spec = importlib.util.spec_from_file_location(f"_tinyexp_run_with_redis_{module_path.stem}", module_path)
            if spec is not None and spec.loader is not None:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
    if module is not None:
        exp_classes = [
            obj
            for obj in vars(module).values()
            if inspect.isclass(obj) and issubclass(obj, RedisCfgMixin) and obj is not RedisCfgMixin
        ]
        if len(exp_classes) == 1:
            exp = exp_classes[0]()
            exp.exp_class = f"{exp_classes[0].__module__}.{exp_classes[0].__qualname__}"

    if exp is None:
        print("Could not infer experiment config; using RedisCfgMixin defaults", file=sys.stderr)
        return RedisCfgMixin().redis_cache_cfg

    overrides = [arg for arg in argv[1:] if arg.startswith("redis_cache_cfg.")]
    if overrides:
        exp.set_cfg(OmegaConf.from_dotlist(overrides))
    return exp.redis_cache_cfg


def start_local_redis(redis_cache_cfg: Any, started_nodes: list[tuple[str, int]], env: dict[str, str]) -> bool:
    host = redis_cache_cfg.redis_cluster_host
    ports = [int(port) for port in redis_cache_cfg.redis_cluster_ports]
    max_memory_bytes = max(1, int((redis_cache_cfg.redis_cache_max_memory / len(ports)) * (1024**3)))
    local_started_nodes: list[tuple[str, int]] = []
    for port in ports:
        if not start_redis_node(host=host, port=port, max_memory_bytes=max_memory_bytes, cluster_enabled=False):
            cleanup_started_nodes(local_started_nodes)
            return False
        local_started_nodes.append((host, port))
        print(f"Redis server started on {host}:{port}", flush=True)
    started_nodes.extend(local_started_nodes)
    env["TINYEXP_REDIS_CLUSTER_HOST"] = host
    env["TINYEXP_REDIS_CLUSTER_PORTS"] = ",".join(str(port) for port in ports)
    return True


def start_rendezvous_redis_cluster(  # noqa: C901
    redis_cache_cfg: Any, world_size: int, started_nodes: list[tuple[str, int]], env: dict[str, str]
) -> bool:
    ports = [int(port) for port in redis_cache_cfg.redis_cluster_ports]
    node_count = world_size * len(ports)
    if node_count < 3:
        print("Redis Cluster requires at least 3 master nodes", file=sys.stderr)
        return False

    rendezvous_host = redis_cache_cfg.redis_cluster_host
    if rendezvous_host == "127.0.0.1":
        print(
            "redis_cache_cfg.redis_cluster_host must be set to the master host when rendezvous world size > 1",
            file=sys.stderr,
        )
        return False
    rendezvous_port = REDIS_RENDEZVOUS_PORT
    timeout_s = REDIS_RENDEZVOUS_TIMEOUT_S

    node_host = os.environ.get("POD_IP")
    if node_host is None:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock, suppress(OSError):
            sock.connect((rendezvous_host, rendezvous_port))
            node_host = sock.getsockname()[0]
    if not node_host or node_host.startswith("127."):
        with suppress(OSError):
            node_host = socket.gethostbyname(socket.gethostname())
    if not node_host:
        node_host = "127.0.0.1"

    max_memory_bytes = max(1, int((redis_cache_cfg.redis_cache_max_memory / node_count) * (1024**3)))
    local_started_nodes: list[tuple[str, int]] = []
    for port in ports:
        if not start_redis_node(host=node_host, port=port, max_memory_bytes=max_memory_bytes, cluster_enabled=True):
            cleanup_started_nodes(local_started_nodes)
            return False
        local_started_nodes.append((node_host, port))
        print(f"Redis Cluster node started on {node_host}:{port}", flush=True)
    started_nodes.extend(local_started_nodes)

    nodes: dict[str, tuple[str, list[int]]] = {}
    lock = threading.Lock()
    ready: dict[str, Any] = {}
    error = ""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            nonlocal error
            if self.path != "/register":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            host = payload["host"]
            worker_ports = [int(port) for port in payload["ports"]]
            with lock:
                nodes[f"{host}:{','.join(str(port) for port in worker_ports)}"] = (host, worker_ports)
                if not ready and not error and len(nodes) >= world_size:
                    addresses = [
                        f"{node_host}:{port}" for node_host, node_ports in nodes.values() for port in node_ports
                    ]
                    create_cmd = [
                        "redis-cli",
                        "--cluster",
                        "create",
                        *addresses,
                        "--cluster-replicas",
                        "0",
                        "--cluster-yes",
                    ]
                    if subprocess.run(create_cmd, check=False).returncode != 0:  # noqa: S603
                        error = f"redis-cli cluster create failed for {addresses}"
                    else:
                        startup_host, startup_ports = next(iter(nodes.values()))
                        ready.update({"startup_host": startup_host, "startup_ports": startup_ports})
                if error:
                    response = {"status": "error", "message": error}
                elif ready:
                    response = {"status": "ready", **ready}
                else:
                    response = {"status": "waiting", "registered": len(nodes), "world_size": world_size}
            body = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    server = None
    with suppress(OSError):
        server = ThreadingHTTPServer((rendezvous_host, rendezvous_port), Handler)
    if server is not None:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"Redis rendezvous master listening on {rendezvous_host}:{rendezvous_port}", flush=True)

    url = f"http://{rendezvous_host}:{rendezvous_port}/register"
    payload = json.dumps({"host": node_host, "ports": ports}).encode()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            request = urllib.request.Request(  # noqa: S310
                url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
            )
            with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310
                result = json.loads(response.read())
        except (OSError, urllib.error.URLError, TimeoutError):
            time.sleep(1)
            continue
        if result["status"] == "ready":
            env["TINYEXP_REDIS_CLUSTER_HOST"] = result["startup_host"]
            env["TINYEXP_REDIS_CLUSTER_PORTS"] = ",".join(str(port) for port in result["startup_ports"])
            print(f"Redis Cluster startup node: {result['startup_host']}:{result['startup_ports'][0]}", flush=True)
            return True
        if result["status"] == "error":
            print(f"Redis rendezvous failed: {result['message']}", file=sys.stderr)
            return False
        print(f"Redis rendezvous waiting: {result['registered']}/{result['world_size']} workers registered", flush=True)
        time.sleep(1)
    print(f"Redis rendezvous timed out after {timeout_s}s", file=sys.stderr)
    return False


def start_redis_node(*, host: str, port: int, max_memory_bytes: int, cluster_enabled: bool) -> bool:
    shutdown_redis(host, port)
    command_parts = [
        f"rm -f /tmp/tinyexp-redis-{int(port)}.nodes.conf;" if cluster_enabled else "",
        "nohup redis-server",
        f"--bind {shlex.quote(host)}",
        "--protected-mode no",
        f"--port {int(port)}",
        "--daemonize no",
        "--dir /tmp",
        "--save ''",
        "--appendonly no",
        f"--maxmemory {max_memory_bytes}",
    ]
    if cluster_enabled:
        command_parts.extend(
            [
                "--cluster-enabled yes",
                f"--cluster-config-file tinyexp-redis-{int(port)}.nodes.conf",
                "--cluster-node-timeout 5000",
                f"--cluster-announce-ip {shlex.quote(host)}",
                f"--cluster-announce-port {int(port)}",
                f"--cluster-announce-bus-port {int(port) + 10000}",
            ]
        )
    command_parts.append(f"> /tmp/tinyexp-redis-{int(port)}.log 2>&1 & echo $! > /tmp/tinyexp-redis-{int(port)}.pid")
    shell_cmd = " ".join(part for part in command_parts if part)
    if subprocess.run(["bash", "-lc", shell_cmd], check=False).returncode != 0:  # noqa: S603,S607
        return False
    return wait_for_redis(host, port)


def cleanup_started_nodes(started_nodes: list[tuple[str, int]]) -> None:
    nodes_by_host: dict[str, list[int]] = {}
    for host, port in started_nodes:
        nodes_by_host.setdefault(host, []).append(port)
    for host, ports in nodes_by_host.items():
        cleanup_code = "\n".join(
            [
                "import os, signal, subprocess",
                f"host = {host!r}",
                f"ports = {sorted({int(port) for port in ports})!r}",
                "for port in ports:",
                "    pidfile = f'/tmp/tinyexp-redis-{port}.pid'",
                "    try:",
                "        with open(pidfile) as f:",
                "            os.kill(int(f.read().strip()), signal.SIGKILL)",
                "    except Exception:",
                "        pass",
                "    try:",
                "        os.remove(pidfile)",
                "    except FileNotFoundError:",
                "        pass",
                "out = subprocess.run(['ps', '-eo', 'pid=,args='], text=True, stdout=subprocess.PIPE, check=False).stdout",
                "targets = [f'redis-server {host}:{port}' for port in ports]",
                "for line in out.splitlines():",
                "    if any(target in line for target in targets):",
                "        try:",
                "            os.kill(int(line.split(None, 1)[0]), signal.SIGKILL)",
                "        except Exception:",
                "            pass",
            ]
        )
        with suppress(subprocess.TimeoutExpired):
            subprocess.run([sys.executable, "-c", cleanup_code], check=False, timeout=10)  # noqa: S603


def shutdown_redis(host: str, port: int) -> None:
    pidfile = f"/tmp/tinyexp-redis-{int(port)}.pid"  # noqa: S108
    shutdown_code = (
        "import redis; "
        f"c=redis.Redis(host={host!r}, port={int(port)}, socket_connect_timeout=0.5, socket_timeout=0.5); "
        "\ntry:\n c.shutdown(nosave=True)\nexcept redis.exceptions.RedisError:\n pass\nfinally:\n c.close()"
    )
    with suppress(subprocess.TimeoutExpired):
        subprocess.run([sys.executable, "-c", shutdown_code], check=False, timeout=2)  # noqa: S603
    kill_cmd = f"test -f {pidfile} && kill -9 $(cat {pidfile}) >/dev/null 2>&1 || true; rm -f {pidfile}"
    with suppress(subprocess.TimeoutExpired):
        subprocess.run(["bash", "-lc", kill_cmd], check=False, timeout=5)  # noqa: S603,S607


def wait_for_redis(host: str, port: int) -> bool:
    client = redis.StrictRedis(host=host, port=port, socket_connect_timeout=1, socket_timeout=1)
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


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
