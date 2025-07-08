import ray

from tinyexp.utils.redis_utils import RedisClusterManager


def test_redis_cluster_manager():
    """
    test redis cluster manager with ray
    """
    if not ray.is_initialized():
        ray.init()

    ports = [7000, 7001, 7002]
    max_memory_per_port = 100 * 1024 * 1024  # 100 MB

    remote_redis_cluster_manager = ray.remote(num_cpus=len(ports))(RedisClusterManager)
    redis_actor = remote_redis_cluster_manager.remote(ports=ports, max_memory_per_port=max_memory_per_port)

    success = ray.get(redis_actor.start_redis_cluster.remote())
    print("Redis cluster started:", success)
