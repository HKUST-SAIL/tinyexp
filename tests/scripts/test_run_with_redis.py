from __future__ import annotations

from pathlib import Path

from scripts.run_with_redis import build_redis_cache_cfg, main


def test_build_redis_cache_cfg_uses_exp_class_overrides(tmp_path: Path) -> None:
    exp_file = tmp_path / "demo_exp.py"
    exp_file.write_text(
        "from dataclasses import dataclass\n"
        "from tinyexp import RedisCfgMixin, TinyExp\n"
        "@dataclass\n"
        "class DemoExp(TinyExp, RedisCfgMixin):\n"
        "    pass\n"
    )

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


def test_rendezvous_mode_requires_non_local_master_host(tmp_path: Path, monkeypatch) -> None:
    exp_file = tmp_path / "demo_exp.py"
    exp_file.write_text(
        "from dataclasses import dataclass\n"
        "from tinyexp import RedisCfgMixin, TinyExp\n"
        "@dataclass\n"
        "class DemoExp(TinyExp, RedisCfgMixin):\n"
        "    pass\n"
    )

    assert (
        main(
            [
                "python",
                str(exp_file),
                "redis_cache_cfg.redis_rendezvous_world_size=2",
                "redis_cache_cfg.redis_cluster_ports=[7010,7011]",
            ]
        )
        == 1
    )


def test_rendezvous_mode_uses_exp_ports(tmp_path: Path, monkeypatch) -> None:
    exp_file = tmp_path / "demo_exp.py"
    exp_file.write_text(
        "from dataclasses import dataclass\n"
        "from tinyexp import RedisCfgMixin, TinyExp\n"
        "@dataclass\n"
        "class DemoExp(TinyExp, RedisCfgMixin):\n"
        "    pass\n"
    )
    captured = {}

    def fake_start_rendezvous_redis_cluster(redis_cache_cfg, world_size, started_nodes, env):
        captured["world_size"] = world_size
        captured["ports"] = list(redis_cache_cfg.redis_cluster_ports)
        env["TINYEXP_REDIS_CLUSTER_HOST"] = "10.0.0.1"
        env["TINYEXP_REDIS_CLUSTER_PORTS"] = "7010,7011"
        return True

    def fake_popen(args, **kwargs):
        captured["argv"] = args
        return type("P", (), {"wait": lambda self: 0})()

    monkeypatch.setattr("scripts.run_with_redis.start_rendezvous_redis_cluster", fake_start_rendezvous_redis_cluster)
    monkeypatch.setattr("scripts.run_with_redis.subprocess.Popen", fake_popen)

    assert (
        main(
            [
                "python",
                str(exp_file),
                "redis_cache_cfg.redis_cluster_host=10.0.0.1",
                "redis_cache_cfg.redis_cluster_ports=[7010,7011]",
                "redis_cache_cfg.redis_rendezvous_world_size=2",
            ]
        )
        == 0
    )
    assert captured["world_size"] == 2
    assert captured["ports"] == [7010, 7011]
    assert "redis_cache_cfg.redis_cluster_host=10.0.0.1" in captured["argv"]
    assert "redis_cache_cfg.redis_cluster_ports=[7010,7011]" in captured["argv"]
