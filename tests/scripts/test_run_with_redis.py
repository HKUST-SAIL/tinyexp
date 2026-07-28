from __future__ import annotations

import signal
import subprocess
import sys
from pathlib import Path

from tinyexp.cli import run_with_redis
from tinyexp.cli.run_with_redis import build_redis_cache_cfg, main, stop_child_process

ROOT = Path(__file__).resolve().parents[2]


def _write_demo_exp(tmp_path: Path) -> Path:
    exp_file = tmp_path / "demo_exp.py"
    exp_file.write_text(
        "from dataclasses import dataclass\n"
        "from tinyexp import TinyExp\n"
        "from tinyexp.exp_mixins import RedisCfgMixin\n"
        "@dataclass\n"
        "class DemoExp(TinyExp, RedisCfgMixin):\n"
        "    pass\n"
    )
    return exp_file


class _FinishedProcess:
    def __init__(self) -> None:
        self.pid = 12345
        self.returncode = 0

    def wait(self, timeout=None) -> int:  # type: ignore[no-untyped-def]
        return self.returncode

    def poll(self) -> int:
        return self.returncode


def test_run_with_redis_python_syntax() -> None:
    assert (
        subprocess.run(  # noqa: S603
            [sys.executable, "-m", "py_compile", "tinyexp/cli/run_with_redis.py"],
            cwd=ROOT,
            check=False,
        ).returncode
        == 0
    )


def test_project_scripts_define_redis_command() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text()

    assert 'tinyexp-run-with-redis = "tinyexp.cli.run_with_redis:cli"' in pyproject


def test_build_redis_cache_cfg_uses_exp_class_overrides(tmp_path: Path) -> None:
    exp_file = _write_demo_exp(tmp_path)

    redis_cache_cfg = build_redis_cache_cfg(
        [
            "python",
            str(exp_file),
            "redis_cache_cfg.redis_cluster_ports=[7002]",
            "redis_cache_cfg.redis_cache_enabled=false",
            "redis_cache_cfg.redis_rendezvous_world_size=2",
        ]
    )

    assert redis_cache_cfg.redis_cluster_ports == [7002]
    assert redis_cache_cfg.redis_cache_enabled is False
    assert redis_cache_cfg.redis_rendezvous_world_size == 2


def test_build_redis_cache_cfg_applies_overrides_to_default_config() -> None:
    redis_cache_cfg = build_redis_cache_cfg(
        [
            "python",
            "missing_exp.py",
            "redis_cache_cfg.redis_cluster_ports=[7002]",
            "redis_cache_cfg.redis_cache_enabled=false",
            "redis_cache_cfg.redis_rendezvous_world_size=2",
        ]
    )

    assert redis_cache_cfg.redis_cluster_ports == [7002]
    assert redis_cache_cfg.redis_cache_enabled is False
    assert redis_cache_cfg.redis_rendezvous_world_size == 2


def test_build_redis_cache_cfg_uses_local_exp_class_over_imported_base(tmp_path: Path, monkeypatch) -> None:
    base_file = tmp_path / "base_exp_module.py"
    base_file.write_text(
        "from dataclasses import dataclass, field\n"
        "from tinyexp import TinyExp\n"
        "from tinyexp.exp_mixins import RedisCfgMixin\n"
        "@dataclass\n"
        "class BaseExp(TinyExp, RedisCfgMixin):\n"
        "    @dataclass\n"
        "    class RedisCfg(RedisCfgMixin.RedisCfg):\n"
        "        redis_cache_max_memory: int = 111\n"
        "    redis_cache_cfg: RedisCfg = field(default_factory=RedisCfg)\n"
    )
    exp_file = tmp_path / "child_exp.py"
    exp_file.write_text(
        "from dataclasses import dataclass, field\n"
        "from base_exp_module import BaseExp as ImportedBaseExp\n"
        "@dataclass\n"
        "class ChildExp(ImportedBaseExp):\n"
        "    @dataclass\n"
        "    class RedisCfg(ImportedBaseExp.RedisCfg):\n"
        "        redis_cache_max_memory: int = 321\n"
        "    redis_cache_cfg: RedisCfg = field(default_factory=RedisCfg)\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    redis_cache_cfg = build_redis_cache_cfg(["accelerate", "launch", str(exp_file)])

    assert redis_cache_cfg.redis_cache_max_memory == 321


def test_rendezvous_mode_requires_non_local_master_host(tmp_path: Path) -> None:
    exp_file = _write_demo_exp(tmp_path)

    assert (
        main(
            [
                "python",
                str(exp_file),
                "redis_cache_cfg.redis_rendezvous_world_size=2",
                "redis_cache_cfg.redis_cluster_ports=[7010,7011]",
            ]
        )
        == 2
    )


def test_run_with_redis_appends_connection_overrides_for_torchrun_command(tmp_path: Path, monkeypatch) -> None:
    exp_file = _write_demo_exp(tmp_path)
    captured = {}

    def fake_start_local_redis(redis_cache_cfg, started_nodes):
        captured["startup_ports"] = list(redis_cache_cfg.redis_cluster_ports)
        return "127.0.0.1", [7012, 7013]

    def fake_popen(args, **kwargs):
        captured["argv"] = args
        captured["popen_kwargs"] = kwargs
        return _FinishedProcess()

    monkeypatch.setattr("tinyexp.cli.run_with_redis.start_local_redis", fake_start_local_redis)
    monkeypatch.setattr("tinyexp.cli.run_with_redis.subprocess.Popen", fake_popen)

    assert (
        main(
            [
                "torchrun",
                "--standalone",
                "--nnodes=1",
                "--nproc_per_node=2",
                str(exp_file),
                "redis_cache_cfg.redis_cluster_ports=[7012,7013]",
                "redis_cache_cfg.redis_cache_enabled=true",
                "redis_cache_cfg.redis_rendezvous_world_size=1",
                "output_root=/tmp/out",
            ]
        )
        == 0
    )

    assert captured["startup_ports"] == [7012, 7013]
    assert captured["argv"][-3:] == [
        "redis_cache_cfg.redis_cluster_host=127.0.0.1",
        "redis_cache_cfg.redis_cluster_ports=[7012,7013]",
        "redis_cache_cfg.redis_rendezvous_world_size=1",
    ]
    assert captured["argv"].count("redis_cache_cfg.redis_cluster_ports=[7012,7013]") == 1
    assert captured["argv"].count("redis_cache_cfg.redis_rendezvous_world_size=1") == 1
    assert captured["argv"].index("--nproc_per_node=2") < captured["argv"].index(str(exp_file))
    assert captured["popen_kwargs"]["start_new_session"] is True


def test_rendezvous_mode_uses_exp_ports(tmp_path: Path, monkeypatch) -> None:
    exp_file = _write_demo_exp(tmp_path)
    captured = {}

    def fake_start_rendezvous_redis_cluster(
        redis_cache_cfg,
        *,
        world_size,
        node_rank,
        head_addr,
        rendezvous_port,
        timeout_s,
        started_nodes,
    ):
        captured["world_size"] = world_size
        captured["node_rank"] = node_rank
        captured["head_addr"] = head_addr
        captured["rendezvous_port"] = rendezvous_port
        captured["timeout_s"] = timeout_s
        captured["ports"] = list(redis_cache_cfg.redis_cluster_ports)
        return "10.0.0.1", [7010, 7011]

    def fake_popen(args, **kwargs):
        captured["argv"] = args
        captured["popen_kwargs"] = kwargs
        return _FinishedProcess()

    monkeypatch.setattr(
        "tinyexp.cli.run_with_redis.start_rendezvous_redis_cluster",
        fake_start_rendezvous_redis_cluster,
    )
    monkeypatch.setattr("tinyexp.cli.run_with_redis.subprocess.Popen", fake_popen)

    assert (
        main(
            [
                "--node-count",
                "2",
                "--node-rank",
                "1",
                "--head-addr",
                "10.0.0.1",
                "--rendezvous-port",
                "26380",
                "--wait-timeout",
                "30",
                "--",
                "python",
                str(exp_file),
                "redis_cache_cfg.redis_cluster_host=old",
                "redis_cache_cfg.redis_cluster_ports=[7010,7011]",
                "redis_cache_cfg.redis_rendezvous_world_size=99",
            ]
        )
        == 0
    )
    assert captured["world_size"] == 2
    assert captured["node_rank"] == 1
    assert captured["head_addr"] == "10.0.0.1"
    assert captured["rendezvous_port"] == 26380
    assert captured["timeout_s"] == 30
    assert captured["ports"] == [7010, 7011]
    assert captured["argv"].count("redis_cache_cfg.redis_cluster_host=10.0.0.1") == 1
    assert captured["argv"].count("redis_cache_cfg.redis_cluster_ports=[7010,7011]") == 1
    assert captured["argv"].count("redis_cache_cfg.redis_rendezvous_world_size=2") == 1
    assert "redis_cache_cfg.redis_cluster_host=old" not in captured["argv"]
    assert "redis_cache_cfg.redis_rendezvous_world_size=99" not in captured["argv"]
    assert captured["popen_kwargs"]["start_new_session"] is True


def test_stop_child_process_signals_process_group_and_kills_on_timeout(
    monkeypatch,
) -> None:
    calls = []

    class HangingProcess:
        pid = 24680

        def poll(self):  # type: ignore[no-untyped-def]
            return None

        def wait(self, timeout=None):  # type: ignore[no-untyped-def]
            raise subprocess.TimeoutExpired("child", timeout)

        def send_signal(self, signum):  # type: ignore[no-untyped-def]
            calls.append(("send_signal", signum))

        def kill(self):  # type: ignore[no-untyped-def]
            calls.append(("kill", signal.SIGKILL))

    def fake_killpg(pid, signum):  # type: ignore[no-untyped-def]
        calls.append((pid, signum))

    monkeypatch.setattr(run_with_redis.os, "killpg", fake_killpg)

    stop_child_process(HangingProcess(), signal.SIGTERM, timeout_s=0)

    assert calls == [(24680, signal.SIGTERM), (24680, signal.SIGKILL)]
