from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import shlex
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path
from types import ModuleType
from typing import Any

import redis
from omegaconf import OmegaConf

from tinyexp import RedisCfgMixin


def main(argv: list[str]) -> int:  # noqa: C901
    node_specs: list[str] = []
    while argv[:1] == ["--cluster-node"]:
        if len(argv) < 2:
            print("--cluster-node requires ssh_target:host[:port[,port...]]", file=sys.stderr)
            return 2
        node_specs.append(argv[1])
        argv = argv[2:]
    if not node_specs:
        node_specs = [item for item in os.environ.get("TINYEXP_REDIS_CLUSTER_NODES", "").split(";") if item]
    if argv[:1] == ["--"]:
        argv = argv[1:]
    if not argv:
        print(
            "Usage: python scripts/run_with_redis.py "
            "[--cluster-node ssh_target:host[:port[,port...]]] -- <command> [args...]",
            file=sys.stderr,
        )
        return 2

    redis_cache_cfg = build_redis_cache_cfg(argv)
    configured_ports = [int(port) for port in redis_cache_cfg.redis_cluster_ports]
    if not configured_ports:
        print("redis_cache_cfg.redis_cluster_ports must not be empty", file=sys.stderr)
        return 2
    nodes: list[tuple[str, str, list[int]]] = []
    started_nodes: list[tuple[str, str, int]] = []
    env = os.environ.copy()

    try:
        for spec in node_specs:
            parts = spec.split(":", maxsplit=2)
            if len(parts) not in {2, 3}:
                print(f"Invalid --cluster-node spec: {spec}", file=sys.stderr)
                return 2
            ssh_target, host = parts[:2]
            ports = [int(port) for port in parts[2].split(",") if port] if len(parts) == 3 else configured_ports
            nodes.append((ssh_target, host, ports))

        redis_status = True
        if redis_cache_cfg.redis_cache_enabled and nodes:
            redis_status = start_redis_cluster(nodes, redis_cache_cfg.redis_cache_max_memory, started_nodes, env)
        elif redis_cache_cfg.redis_cache_enabled:
            redis_status = start_local_redis(redis_cache_cfg, started_nodes, env)

        print(f"Redis status:\033[32m{redis_status}\033[0m", flush=True)
        if not redis_status:
            return 1

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
    exp = load_exp_from_command(argv)
    if exp is None:
        print("Could not infer experiment config; using RedisCfgMixin defaults", file=sys.stderr)
        return RedisCfgMixin().redis_cache_cfg
    overrides = [arg for arg in argv[1:] if arg.startswith("redis_cache_cfg.")]
    if overrides:
        exp.set_cfg(OmegaConf.from_dotlist(overrides))
    return exp.redis_cache_cfg


def load_exp_from_command(argv: list[str]) -> Any | None:
    if len(argv) < 2 or not Path(argv[0]).name.startswith("python"):
        return None
    if argv[1] == "-m" and len(argv) >= 3:
        module = importlib.import_module(argv[2])
    elif argv[1].endswith(".py"):
        module = import_module_from_path(argv[1])
    else:
        return None
    if module is None:
        return None

    exp_classes = [
        obj
        for obj in vars(module).values()
        if inspect.isclass(obj) and issubclass(obj, RedisCfgMixin) and obj is not RedisCfgMixin
    ]
    if len(exp_classes) != 1:
        return None
    exp = exp_classes[0]()
    exp.exp_class = f"{exp_classes[0].__module__}.{exp_classes[0].__qualname__}"
    return exp


def import_module_from_path(path: str) -> ModuleType | None:
    module_path = Path(path).resolve()
    module_name = f"_tinyexp_run_with_redis_{module_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def start_local_redis(redis_cache_cfg: Any, started_nodes: list[tuple[str, str, int]], env: dict[str, str]) -> bool:
    host = redis_cache_cfg.redis_cluster_host
    ports = [int(port) for port in redis_cache_cfg.redis_cluster_ports]
    max_memory_bytes = max(1, int((redis_cache_cfg.redis_cache_max_memory / len(ports)) * (1024**3)))
    local_started_nodes: list[tuple[str, str, int]] = []
    for port in ports:
        shutdown_redis("", host, port)
        shell_cmd = " ".join(
            [
                "nohup redis-server",
                f"--bind {shlex.quote(host)}",
                "--protected-mode no",
                f"--port {port}",
                "--daemonize no",
                "--save ''",
                "--appendonly no",
                f"--maxmemory {max_memory_bytes}",
                f"> /tmp/tinyexp-redis-{port}.log 2>&1 & echo $! > /tmp/tinyexp-redis-{port}.pid",
            ]
        )
        if subprocess.run(["bash", "-lc", shell_cmd], check=False).returncode != 0:  # noqa: S603,S607
            cleanup_started_nodes(local_started_nodes)
            return False
        if not wait_for_redis(host, port):
            cleanup_started_nodes(local_started_nodes)
            return False
        local_started_nodes.append(("", host, port))
        print(f"Redis server started on {host}:{port}", flush=True)
    started_nodes.extend(local_started_nodes)
    env["TINYEXP_REDIS_CLUSTER_HOST"] = host
    env["TINYEXP_REDIS_CLUSTER_PORTS"] = ",".join(str(port) for port in ports)
    return True


def start_redis_cluster(
    nodes: list[tuple[str, str, list[int]]],
    max_memory_gb: int,
    started_nodes: list[tuple[str, str, int]],
    env: dict[str, str],
) -> bool:
    node_count = sum(len(ports) for _, _, ports in nodes)
    if node_count < 3:
        print("Redis Cluster requires at least 3 master nodes", file=sys.stderr)
        return False

    max_memory_bytes = max(1, int((max_memory_gb / node_count) * (1024**3)))
    for ssh_target, host, ports in nodes:
        for port in ports:
            shutdown_redis(ssh_target, host, port)
            shell_cmd = " ".join(
                [
                    f"rm -f /tmp/tinyexp-redis-{int(port)}.nodes.conf;",
                    "nohup redis-server",
                    f"--bind {shlex.quote(host)}",
                    "--protected-mode no",
                    f"--port {int(port)}",
                    "--daemonize no",
                    "--dir /tmp",
                    "--save ''",
                    "--appendonly no",
                    f"--maxmemory {max_memory_bytes}",
                    "--cluster-enabled yes",
                    f"--cluster-config-file tinyexp-redis-{int(port)}.nodes.conf",
                    "--cluster-node-timeout 5000",
                    f"--cluster-announce-ip {shlex.quote(host)}",
                    f"--cluster-announce-port {int(port)}",
                    f"--cluster-announce-bus-port {int(port) + 10000}",
                    f"> /tmp/tinyexp-redis-{int(port)}.log 2>&1 & echo $! > /tmp/tinyexp-redis-{int(port)}.pid",
                ]
            )
            if ssh_target:
                result = subprocess.run(["ssh", ssh_target, shell_cmd], check=False)  # noqa: S603,S607
            else:
                result = subprocess.run(["bash", "-lc", shell_cmd], check=False)  # noqa: S603,S607
            if result.returncode != 0 or not wait_for_redis(host, port):
                return False
            started_nodes.append((ssh_target, host, port))
            print(f"Redis Cluster node started on {host}:{port}", flush=True)

    addresses = [f"{host}:{port}" for _, host, ports in nodes for port in ports]
    create_cmd = ["redis-cli", "--cluster", "create", *addresses, "--cluster-replicas", "0", "--cluster-yes"]
    if subprocess.run(create_cmd, check=False).returncode != 0:  # noqa: S603
        return False

    startup_host, startup_port = addresses[0].split(":", maxsplit=1)
    env["TINYEXP_REDIS_CLUSTER_HOST"] = startup_host
    env["TINYEXP_REDIS_CLUSTER_PORTS"] = ",".join(
        [startup_port, *[str(port) for _, _, ports in nodes for port in ports if str(port) != startup_port]]
    )
    print(f"Redis Cluster startup node: {startup_host}:{startup_port}", flush=True)
    return True


def cleanup_started_nodes(started_nodes: list[tuple[str, str, int]]) -> None:
    nodes_by_target: dict[tuple[str, str], list[int]] = {}
    for ssh_target, host, port in started_nodes:
        nodes_by_target.setdefault((ssh_target, host), []).append(port)
    for (ssh_target, host), ports in nodes_by_target.items():
        cleanup_redis_nodes(ssh_target, host, ports)


def cleanup_redis_nodes(ssh_target: str, host: str, ports: list[int]) -> None:
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
    if ssh_target:
        shell_cmd = f"python3 -c {shlex.quote(cleanup_code)} || python -c {shlex.quote(cleanup_code)}"
        with suppress(subprocess.TimeoutExpired):
            subprocess.run(["ssh", ssh_target, shell_cmd], check=False, timeout=10)  # noqa: S603,S607
    else:
        with suppress(subprocess.TimeoutExpired):
            subprocess.run([sys.executable, "-c", cleanup_code], check=False, timeout=10)  # noqa: S603


def shutdown_redis(ssh_target: str, host: str, port: int) -> None:
    pidfile = f"/tmp/tinyexp-redis-{int(port)}.pid"  # noqa: S108
    if ssh_target:
        shell_cmd = " ".join(
            [
                f"timeout 2 redis-cli -h {shlex.quote(host)} -p {int(port)} shutdown nosave >/dev/null 2>&1 || true;",
                f"test -f {pidfile} && kill -9 $(cat {pidfile}) >/dev/null 2>&1 || true;",
                f"rm -f {pidfile};",
                "ps -eo pid=,comm=,args= |",
                f"awk -v addr={shlex.quote(f'{host}:{int(port)}')} '$2 == \"redis-server\" && index($0, addr) {{print $1}}' |",
                "xargs -r kill -9",
            ]
        )
        with suppress(subprocess.TimeoutExpired):
            subprocess.run(["ssh", ssh_target, shell_cmd], check=False, timeout=5)  # noqa: S603,S607
        return

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
