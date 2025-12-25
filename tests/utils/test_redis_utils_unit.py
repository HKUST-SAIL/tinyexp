from __future__ import annotations

import pytest

from tinyexp.utils.redis_utils import RedisClusterManager


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
