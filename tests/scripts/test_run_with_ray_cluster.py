from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import tinyexp.cli.run_with_ray_cluster as ray_cluster

ROOT = Path(__file__).resolve().parents[2]


def test_run_with_ray_cluster_single_node_passthrough_ignores_custom_env() -> None:
    # A subprocess because the single-node path replaces the process (execvp).
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "tinyexp.cli.run_with_ray_cluster", "--", sys.executable, "-c", "print('ok')"],
        cwd=ROOT,
        env={**os.environ, "RAY_CLUSTER_NODE_COUNT": "bad"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "ok"
    assert "single-node job detected" in result.stderr


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--node-count", "2"], "--head-addr is required"),
        (["--node-count", "2", "--node-rank", "2", "--head-addr", "10.0.0.1"], "--node-rank must be < --node-count"),
        (["--node-count", "2", "--head-addr", "10.0.0.1", "--ray-port", "0"], "--ray-port"),
        (["--node-count", "2", "--head-addr", "10.0.0.1", "--ray-port", "65536"], "--ray-port"),
        (["--node-count", "2", "--head-addr", "10.0.0.1", "--wait-timeout", "0"], "--wait-timeout"),
    ],
)
def test_parse_args_rejects_invalid_arguments(
    capsys: pytest.CaptureFixture[str], args: list[str], message: str
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        ray_cluster.parse_args([*args, "--", "echo", "unused"])

    assert exc_info.value.code == 2
    assert message in capsys.readouterr().err


def test_ray_commands_time_out_instead_of_hanging(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> None:
        captured["timeout"] = kwargs["timeout"]
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])  # type: ignore[arg-type]

    monkeypatch.setattr(ray_cluster.subprocess, "run", fake_run)

    assert ray_cluster.run_quiet(["ray", "status"]) == 1
    assert ray_cluster.ray_alive_count("python", "10.0.0.1:6379", {}, timeout_s=1.25) is None
    assert captured["timeout"] == 1.25


@pytest.fixture
def fake_clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Replace ``time`` in the CLI with a clock that only advances on sleep; returns the recorded sleeps."""
    clock = {"now": 0.0}
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock["now"] += seconds

    monkeypatch.setattr(ray_cluster, "time", SimpleNamespace(monotonic=lambda: clock["now"], sleep=sleep))
    return sleeps


def test_wait_for_head_uses_remaining_timeout(monkeypatch: pytest.MonkeyPatch, fake_clock: list[float]) -> None:
    timeouts: list[float] = []

    def fake_run_quiet(command: list[str], *, timeout_s: float = 5.0) -> int:
        timeouts.append(timeout_s)
        return 1

    monkeypatch.setattr(ray_cluster, "run_quiet", fake_run_quiet)

    assert ray_cluster.wait_for_head("ray", "10.0.0.1:6379", 3) is False
    assert timeouts == [3.0, 1.0]
    assert fake_clock == [2, 1.0]


def test_run_head_uses_remaining_timeout_and_bounds_sleep(
    monkeypatch: pytest.MonkeyPatch, fake_clock: list[float]
) -> None:
    timeouts: list[float] = []

    def fake_alive_count(python_bin: str, ray_address: str, env: dict[str, str], *, timeout_s: float) -> None:
        timeouts.append(timeout_s)

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
    monkeypatch.setattr(ray_cluster, "ray_alive_count", fake_alive_count)
    monkeypatch.setattr(ray_cluster.subprocess, "run", lambda *command, **kwargs: SimpleNamespace(returncode=0))

    assert ray_cluster.run_head(args, "ray", "python", "10.0.0.1:6379", {}) == 1
    assert timeouts == [3.0, 1.0]
    assert fake_clock == [2, 1.0]
