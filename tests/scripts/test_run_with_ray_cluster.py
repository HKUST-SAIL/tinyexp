from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run_command(args: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        args,
        cwd=ROOT,
        env={**os.environ, **(env or {})},
        text=True,
        capture_output=True,
        check=False,
    )


def test_run_with_ray_cluster_python_syntax() -> None:
    assert run_command([sys.executable, "-m", "py_compile", "scripts/run_with_ray_cluster.py"]).returncode == 0


def test_run_with_ray_cluster_single_node_passthrough_ignores_custom_env() -> None:
    result = run_command(
        [
            sys.executable,
            "scripts/run_with_ray_cluster.py",
            "--",
            sys.executable,
            "-c",
            "print('passthrough-ok')",
        ],
        env={"RAY_CLUSTER_NODE_COUNT": "bad"},
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "passthrough-ok"
    assert "single-node job detected" in result.stderr


def test_run_with_ray_cluster_multi_node_requires_head_addr() -> None:
    result = run_command(
        [
            sys.executable,
            "scripts/run_with_ray_cluster.py",
            "--node-count",
            "2",
            "--",
            "echo",
            "unused",
        ]
    )

    assert result.returncode == 2
    assert "--head-addr is required" in result.stderr
