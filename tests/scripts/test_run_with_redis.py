from __future__ import annotations

from pathlib import Path

from scripts.run_with_redis import build_redis_cache_cfg, main


def _write_demo_exp(tmp_path: Path) -> Path:
    exp_file = tmp_path / "demo_exp.py"
    exp_file.write_text(
        "from dataclasses import dataclass\n"
        "from tinyexp import RedisCfgMixin, TinyExp\n"
        "@dataclass\n"
        "class DemoExp(TinyExp, RedisCfgMixin):\n"
        "    pass\n"
    )
    return exp_file


class _FinishedProcess:
    def __init__(self) -> None:
        self.returncode = 0

    def wait(self, timeout=None) -> int:  # type: ignore[no-untyped-def]
        return self.returncode

    def poll(self) -> int:
        return self.returncode


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
        return _FinishedProcess()

    monkeypatch.setattr("scripts.run_with_redis.start_local_redis", fake_start_local_redis)
    monkeypatch.setattr("scripts.run_with_redis.subprocess.Popen", fake_popen)

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
        return _FinishedProcess()

    monkeypatch.setattr(
        "scripts.run_with_redis.start_rendezvous_redis_cluster",
        fake_start_rendezvous_redis_cluster,
    )
    monkeypatch.setattr("scripts.run_with_redis.subprocess.Popen", fake_popen)

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
