from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import tinyexp.cli.run_with_ray_cluster as ray_cluster

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
    assert run_command([sys.executable, "-m", "py_compile", "tinyexp/cli/run_with_ray_cluster.py"]).returncode == 0


def test_run_quiet_returns_failure_when_ray_command_times_out() -> None:
    with patch.object(subprocess, "run", side_effect=subprocess.TimeoutExpired("ray", 5)):
        assert ray_cluster.run_quiet(["ray", "status"]) == 1


def test_project_scripts_define_ray_cluster_command() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text()

    assert 'tinyexp-run-with-ray-cluster = "tinyexp.cli.run_with_ray_cluster:cli"' in pyproject


def test_run_with_ray_cluster_single_node_passthrough_ignores_custom_env() -> None:
    result = run_command(
        [
            sys.executable,
            "-m",
            "tinyexp.cli.run_with_ray_cluster",
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
            "-m",
            "tinyexp.cli.run_with_ray_cluster",
            "--node-count",
            "2",
            "--",
            "echo",
            "unused",
        ]
    )

    assert result.returncode == 2
    assert "--head-addr is required" in result.stderr


def test_parse_args_rejects_node_rank_out_of_range() -> None:
    with pytest.raises(SystemExit) as exc_info:
        ray_cluster.parse_args(
            [
                "--node-count",
                "2",
                "--node-rank",
                "2",
                "--head-addr",
                "10.0.0.1",
                "--",
                "echo",
                "unused",
            ]
        )

    assert exc_info.value.code == 2


@pytest.mark.parametrize("ray_port", ["0", "65536"])
def test_parse_args_rejects_invalid_ray_port(ray_port: str) -> None:
    with pytest.raises(SystemExit) as exc_info:
        ray_cluster.parse_args(
            [
                "--node-count",
                "2",
                "--head-addr",
                "10.0.0.1",
                "--ray-port",
                ray_port,
                "--",
                "echo",
                "unused",
            ]
        )

    assert exc_info.value.code == 2


def test_parse_args_rejects_zero_wait_timeout() -> None:
    with pytest.raises(SystemExit) as exc_info:
        ray_cluster.parse_args(
            [
                "--node-count",
                "2",
                "--head-addr",
                "10.0.0.1",
                "--wait-timeout",
                "0",
                "--",
                "echo",
                "unused",
            ]
        )

    assert exc_info.value.code == 2


def test_ray_alive_count_passes_timeout_and_handles_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> None:
        captured["timeout"] = kwargs["timeout"]
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])  # type: ignore[arg-type]

    monkeypatch.setattr(ray_cluster.subprocess, "run", fake_run)

    assert ray_cluster.ray_alive_count("python", "10.0.0.1:6379", {}, timeout_s=1.25) is None
    assert captured["timeout"] == 1.25


def test_wait_for_head_uses_remaining_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = {"now": 0.0}
    timeouts: list[float] = []
    sleeps: list[float] = []

    def monotonic() -> float:
        return clock["now"]

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock["now"] += seconds

    def fake_run_quiet(command: list[str], *, timeout_s: float = 5.0) -> int:
        timeouts.append(timeout_s)
        return 1

    monkeypatch.setattr(
        ray_cluster,
        "time",
        SimpleNamespace(monotonic=monotonic, sleep=sleep),
    )
    monkeypatch.setattr(ray_cluster, "run_quiet", fake_run_quiet)

    assert ray_cluster.wait_for_head("ray", "10.0.0.1:6379", 3) is False
    assert timeouts == [3.0, 1.0]
    assert sleeps == [2, 1.0]


def test_run_head_uses_remaining_timeout_and_bounds_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = {"now": 0.0}
    timeouts: list[float] = []
    sleeps: list[float] = []

    def monotonic() -> float:
        return clock["now"]

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock["now"] += seconds

    def fake_alive_count(
        python_bin: str,
        ray_address: str,
        env: dict[str, str],
        *,
        timeout_s: float,
    ) -> None:
        timeouts.append(timeout_s)
        return None

    args = SimpleNamespace(
        node_count=2,
        head_addr="10.0.0.1",
        head_node_ip="",
        ray_port=6379,
        dashboard_port=8265,
        metrics_port=8080,
        include_dashboard="false",
        client_port=None,
        wait_timeout=3,
        command=["echo", "unused"],
    )

    monkeypatch.setattr(
        ray_cluster,
        "time",
        SimpleNamespace(monotonic=monotonic, sleep=sleep),
    )
    monkeypatch.setattr(ray_cluster, "ray_alive_count", fake_alive_count)
    monkeypatch.setattr(
        ray_cluster.subprocess,
        "run",
        lambda *command, **kwargs: SimpleNamespace(returncode=0),
    )

    assert ray_cluster.run_head(args, "ray", "python", "10.0.0.1:6379", {}) == 1
    assert timeouts == [3.0, 1.0]
    assert sleeps == [2, 1.0]
