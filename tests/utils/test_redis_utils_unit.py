from __future__ import annotations

import pytest

from tinyexp import RedisCfgMixin
from tinyexp.utils.redis_utils import RedisClusterManager, RedisClusterStartupError


def test_redis_cluster_manager_validates_inputs() -> None:
    with pytest.raises(ValueError, match="ports must not be empty"):
        RedisClusterManager(ports=[], max_memory_per_port=1.0)

    with pytest.raises(ValueError, match="ports must be unique"):
        RedisClusterManager(ports=[7000, 7000], max_memory_per_port=1.0)

    with pytest.raises(ValueError, match="Invalid port"):
        RedisClusterManager(ports=[0], max_memory_per_port=1.0)

    with pytest.raises(ValueError, match="max_memory_per_port must be > 0 GB"):
        RedisClusterManager(ports=[7000], max_memory_per_port=0.0)


def test_redis_cluster_manager_converts_gb_to_bytes() -> None:
    mgr = RedisClusterManager(ports=[7000], max_memory_per_port=0.5)
    assert mgr.max_memory_per_port_gb == 0.5
    assert mgr.max_memory_per_port_bytes == int(0.5 * (1024**3))


def test_redis_cluster_manager_context_raises_when_redis_server_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tinyexp.utils.redis_utils.shutil.which", lambda _cmd: None)
    with (
        pytest.raises(RedisClusterStartupError, match="redis-server command not found"),
        RedisClusterManager(ports=[7000], max_memory_per_port=0.5),
    ):
        pass


def test_redis_cache_cfg_disabled_leaves_no_manager() -> None:
    cfg = RedisCfgMixin.RedisCacheCfg(redis_cache_enabled=False)
    assert cfg.build_redis_cache() is True
    assert cfg.redis_cluster_manager is None


def test_redis_cache_cfg_teardown_redis_cache_idempotent() -> None:
    cfg = RedisCfgMixin.RedisCacheCfg(redis_cache_enabled=False)
    cfg.teardown_redis_cache()
    cfg.teardown_redis_cache()
