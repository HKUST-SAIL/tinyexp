import socket

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


def get_1gpu_placement_group(num_gpus, num_cpus_per_gpu=10):
    """Create and return a placement group for GPU allocation."""
    bundles = [{"CPU": num_cpus_per_gpu, "GPU": 1} for _ in range(num_gpus)]
    pg = placement_group(bundles=bundles, strategy="STRICT_PACK")
    ray.get(pg.ready())
    return pg


def get_1gpu_worker_options(pg, rank, local_rank, num_gpus, master_addr, master_port):
    """Create options for Ray workers."""
    return {
        "runtime_env": {
            "env_vars": {
                "WORLD_SIZE": str(num_gpus),
                "RANK": str(rank),
                "MASTER_ADDR": master_addr,
                "MASTER_PORT": str(master_port),
                "LOCAL_RANK": str(local_rank),
            }
        },
        "scheduling_strategy": PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_bundle_index=rank),
        "num_gpus": 1.0,
    }


def get_network_config():
    """Get network configuration for distributed setup."""
    master_addr = ray._private.services.get_node_ip_address()
    with socket.socket() as sock:
        sock.bind(("", 0))
        master_port = sock.getsockname()[1]
    return master_addr, master_port


def get_num_gpus_worker_options(num_gpus, num_cpus_per_gpu=10):
    """Create options for multiple Ray workers with GPU allocation."""
    pg = get_1gpu_placement_group(num_gpus=num_gpus, num_cpus_per_gpu=num_cpus_per_gpu)
    master_addr, master_port = get_network_config()
    options_list = []
    for i in range(num_gpus):
        options = get_1gpu_worker_options(pg, i, i, num_gpus, master_addr, master_port)
        options_list.append(options)
    return options_list
