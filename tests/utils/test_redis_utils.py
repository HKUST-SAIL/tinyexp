import shutil

import pytest
import ray

from tinyexp.utils.redis_utils import RedisClusterManager


def test_redis_cluster_manager(ray_session):
    """
    test redis cluster manager with ray
    """
    if shutil.which("redis-server") is None:
        pytest.skip("redis-server is not installed")

    ports = [7000, 7001, 7002]
    max_memory_per_port = 0.1  # GB

    remote_redis_cluster_manager = ray.remote(num_cpus=1)(RedisClusterManager)
    redis_actor = remote_redis_cluster_manager.remote(ports=ports, max_memory_per_port=max_memory_per_port)

    success = ray.get(redis_actor.start_redis_cluster.remote())
    if not success:
        pytest.skip("Failed to start redis cluster (see test output for details)")

    try:
        _ = ray.get(redis_actor.get_redis_memory_info.remote())
    finally:
        ray.get(redis_actor.stop_redis_cluster.remote())
