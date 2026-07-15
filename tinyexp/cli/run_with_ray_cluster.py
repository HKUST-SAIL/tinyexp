#!/usr/bin/env python3
"""Start a static Ray cluster, then run a command on node-rank 0."""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

RAY_BIN: str | None = None
RAY_ROLE = ""

ALIVE_NODE_COUNT_CODE = """
import os
import ray

ray.init(address=os.environ["RAY_ADDRESS"], logging_level="ERROR", log_to_driver=False)
try:
    print(sum(1 for node in ray.nodes() if node.get("Alive")))
finally:
    ray.shutdown()
"""


def log(message: str) -> None:
    print(f"Ray Cluster: {message}", file=sys.stderr, flush=True)


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


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tinyexp-run-with-ray-cluster",
        description="Start a static Ray cluster, then run <command> on node-rank 0.",
    )
    parser.add_argument(
        "--node-count",
        type=positive_int,
        default=1,
        help="Total cluster nodes. Default: 1.",
    )
    parser.add_argument("--node-rank", type=uint, default=0, help="Current node rank. Default: 0.")
    parser.add_argument("--head-addr", default="", help="Address workers use to reach Ray head.")
    parser.add_argument(
        "--head-node-ip",
        default="",
        help="Node IP passed to ray start --head. Default: head addr.",
    )
    parser.add_argument("--ray-port", type=uint, default=6379, help="Ray head port. Default: 6379.")
    parser.add_argument(
        "--ray-bin",
        default="",
        help="Ray executable. Default: ray, then ./.venv/bin/ray.",
    )
    parser.add_argument(
        "--python-bin",
        default="",
        help="Python executable for readiness checks. Default: ./.venv/bin/python, python3, python.",
    )
    parser.add_argument(
        "--wait-timeout",
        type=uint,
        default=600,
        help="Startup wait timeout. Default: 600.",
    )
    parser.add_argument(
        "--worker-poll-interval",
        type=uint,
        default=10,
        help="Worker polling interval. Default: 10.",
    )
    parser.add_argument(
        "--include-dashboard",
        default="false",
        help="Enable Ray dashboard. Default: false.",
    )
    parser.add_argument(
        "--dashboard-port",
        type=uint,
        default=8265,
        help="Ray dashboard port. Default: 8265.",
    )
    parser.add_argument(
        "--metrics-port",
        type=uint,
        default=8080,
        help="Ray metrics export port. Default: 8080.",
    )
    parser.add_argument(
        "--client-port",
        type=uint,
        default=None,
        help="Optional Ray Client server port.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run after --.")
    args = parser.parse_args(argv)
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("missing command.")
    if args.node_count > 1 and not args.head_addr:
        parser.error("--head-addr is required for multi-node Ray jobs.")
    return args


def find_executable(value: str, fallbacks: list[str], error_message: str) -> str:
    candidates = [value] if value else fallbacks
    for candidate in candidates:
        found = shutil.which(candidate)
        if found:
            return found
        path = Path(candidate)
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    print(error_message, file=sys.stderr)
    raise SystemExit(1)


def prepend_no_proxy(env: dict[str, str], head_addr: str) -> None:
    prefix = f"{head_addr},127.0.0.1,localhost"
    env["NO_PROXY"] = f"{prefix},{env.get('NO_PROXY', '')}"
    env["no_proxy"] = f"{prefix},{env.get('no_proxy', '')}"


def run_quiet(args: list[str], env: dict[str, str] | None = None) -> int:
    return subprocess.run(  # noqa: S603
        args,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode


def cleanup_ray() -> None:
    global RAY_BIN
    if RAY_BIN is None:
        return
    log(f"stopping Ray {RAY_ROLE or 'runtime'}")
    run_quiet([RAY_BIN, "stop", "--force"])
    RAY_BIN = None


def handle_sigterm(signum: int, _frame: object) -> None:
    cleanup_ray()
    raise SystemExit(128 + signum)


def ray_alive_count(python_bin: str, ray_address: str, env: dict[str, str]) -> int | None:
    check_env = {**env, "RAY_ADDRESS": ray_address}
    result = subprocess.run(  # noqa: S603
        [python_bin, "-c", ALIVE_NODE_COUNT_CODE],
        env=check_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    for line in reversed(result.stdout.splitlines()):
        stripped = line.strip()
        if stripped.isdigit():
            return int(stripped)
    return None


def wait_for_head(ray_bin: str, ray_address: str, wait_timeout: int) -> bool:
    deadline = time.monotonic() + wait_timeout
    while time.monotonic() < deadline:
        if run_quiet([ray_bin, "status", f"--address={ray_address}"]) == 0:
            log(f"Ray head is reachable at {ray_address}")
            return True
        log(f"waiting for Ray head at {ray_address}")
        time.sleep(2)
    return False


def run_head(
    args: argparse.Namespace,
    ray_bin: str,
    python_bin: str,
    ray_address: str,
    env: dict[str, str],
) -> int:
    ray_start_args = [
        ray_bin,
        "start",
        "--head",
        f"--node-ip-address={args.head_node_ip or args.head_addr}",
        f"--port={args.ray_port}",
        "--dashboard-host=0.0.0.0",
        f"--dashboard-port={args.dashboard_port}",
        f"--metrics-export-port={args.metrics_port}",
        f"--include-dashboard={args.include_dashboard}",
    ]
    if args.client_port is not None:
        ray_start_args.append(f"--ray-client-server-port={args.client_port}")

    log(f"starting Ray head on {args.head_node_ip or args.head_addr}:{args.ray_port}")
    start_result = subprocess.run(ray_start_args, env=env, check=False)  # noqa: S603
    if start_result.returncode != 0:
        return start_result.returncode

    deadline = time.monotonic() + args.wait_timeout
    alive_count: int | None = 0
    while time.monotonic() < deadline:
        alive_count = ray_alive_count(python_bin, ray_address, env)
        if alive_count is not None and alive_count >= args.node_count:
            log(f"Ray cluster has {alive_count}/{args.node_count} alive nodes")
            log(f"running command with RAY_ADDRESS={ray_address}: {' '.join(args.command)}")
            return subprocess.run(args.command, env=env, check=False).returncode  # noqa: S603
        log(f"waiting for Ray nodes: {alive_count or 0}/{args.node_count} alive")
        time.sleep(2)

    print(
        f"Error: timed out waiting for {args.node_count} Ray nodes at {ray_address}; "
        f"last alive count={alive_count or 0}",
        file=sys.stderr,
    )
    return 1


def run_worker(args: argparse.Namespace, ray_bin: str, ray_address: str) -> int:
    if not wait_for_head(ray_bin, ray_address, args.wait_timeout):
        print(f"Error: timed out waiting for Ray head at {ray_address}", file=sys.stderr)
        return 1

    log(f"starting Ray worker for {ray_address}")
    start_result = subprocess.run([ray_bin, "start", f"--address={ray_address}"], check=False)  # noqa: S603
    if start_result.returncode != 0:
        return start_result.returncode

    log("Ray worker joined; waiting for head shutdown")
    while run_quiet([ray_bin, "status", f"--address={ray_address}"]) == 0:
        time.sleep(args.worker_poll_interval)
    log("Ray head is no longer reachable; worker exiting")
    return 0


def main(argv: list[str]) -> int:
    global RAY_BIN, RAY_ROLE

    args = parse_args(argv)
    if args.node_count == 1:
        log("single-node job detected; running command unchanged.")
        os.execvp(args.command[0], args.command)  # noqa: S606

    ray_bin = find_executable(
        args.ray_bin,
        ["ray", "./.venv/bin/ray"],
        "Error: ray executable not found. Use --ray-bin.",
    )
    python_bin = find_executable(
        args.python_bin,
        ["./.venv/bin/python", "python3", "python"],
        "Error: python executable not found. Use --python-bin.",
    )

    ray_address = f"{args.head_addr}:{args.ray_port}"
    env = os.environ.copy()
    env["RAY_ADDRESS"] = ray_address
    env["RAY_USAGE_STATS_ENABLED"] = env.get("RAY_USAGE_STATS_ENABLED", "0")
    prepend_no_proxy(env, args.head_addr)

    run_quiet([ray_bin, "stop", "--force"])
    RAY_BIN = ray_bin
    signal.signal(signal.SIGTERM, handle_sigterm)

    try:
        if args.node_rank == 0:
            RAY_ROLE = "head"
            return run_head(args, ray_bin, python_bin, ray_address, env)
        RAY_ROLE = "worker"
        return run_worker(args, ray_bin, ray_address)
    finally:
        cleanup_ray()


def cli() -> None:
    raise SystemExit(main(sys.argv[1:]))


if __name__ == "__main__":
    cli()
