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
        ]
    )

    assert redis_cache_cfg.redis_cluster_ports == [7002]
    assert redis_cache_cfg.redis_cache_enabled is False


def test_cluster_node_without_ports_uses_exp_ports(tmp_path: Path, monkeypatch) -> None:
    exp_file = tmp_path / "demo_exp.py"
    exp_file.write_text(
        "from dataclasses import dataclass\n"
        "from tinyexp import RedisCfgMixin, TinyExp\n"
        "@dataclass\n"
        "class DemoExp(TinyExp, RedisCfgMixin):\n"
        "    pass\n"
    )
    captured = {}

    def fake_start_redis_cluster(nodes, max_memory_gb, started_nodes, env):
        captured["nodes"] = nodes
        return True

    monkeypatch.setattr("scripts.run_with_redis.start_redis_cluster", fake_start_redis_cluster)
    monkeypatch.setattr(
        "scripts.run_with_redis.subprocess.Popen", lambda *args, **kwargs: type("P", (), {"wait": lambda self: 0})()
    )

    assert (
        main(
            [
                "--cluster-node",
                "ssh_target:10.0.0.1",
                "--",
                "python",
                str(exp_file),
                "redis_cache_cfg.redis_cluster_ports=[7010,7011]",
            ]
        )
        == 0
    )
    assert captured["nodes"] == [("ssh_target", "10.0.0.1", [7010, 7011])]
