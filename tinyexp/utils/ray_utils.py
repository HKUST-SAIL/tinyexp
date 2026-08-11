from __future__ import annotations

import os
import socket
from collections import Counter
from collections.abc import Sequence

import ray
from omegaconf import DictConfig
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from .redis_utils import RayRedisClusterManager

_RAY_HEAD_NODE_RESOURCE = "node:__internal_head__"
# Ray node resources are capacity-1 logical labels. A tiny fractional request pins
# bundle 0 to the head node without meaningfully consuming CPU/GPU resources.
_RAY_NODE_RESOURCE_PIN = 0.001


def _maybe_start_ray_redis_cache(cfg: DictConfig) -> RayRedisClusterManager | None:
    redis_cfg = getattr(cfg, "redis_cfg", None)
    if redis_cfg is None or not bool(getattr(redis_cfg, "redis_cache_enabled", False)):
        return None

    requested_world_size = int(getattr(redis_cfg, "redis_rendezvous_world_size", 1))
    if requested_world_size == 1:
        cluster_enabled = False
    elif requested_world_size == -1:
        cluster_enabled = True
    elif requested_world_size > 1:
        return None
    else:
        raise ValueError("redis_rendezvous_world_size must be -1, 1, or > 1")  # noqa: TRY003

    manager = RayRedisClusterManager(redis_cfg)
    startup_host, startup_ports, world_size = manager.start(cluster_enabled=cluster_enabled)
    redis_cfg.redis_cluster_host = startup_host
    redis_cfg.redis_cluster_ports = startup_ports
    redis_cfg.redis_rendezvous_world_size = world_size
    return manager


def get_placement_group(
    num_worker,
    num_gpus_per_worker: float = 1.0,
    num_cpus_per_worker=10,
    strategy="PACK",
):
    """Create and return a placement group for worker allocation."""
    bundles = [{"CPU": num_cpus_per_worker, "GPU": num_gpus_per_worker} for _ in range(num_worker)]
    cluster_resources = ray.cluster_resources() if ray.is_initialized() else {}
    if num_worker > 1 and _RAY_HEAD_NODE_RESOURCE in cluster_resources:
        # PyTorch env:// starts TCPStore on rank 0 at MASTER_ADDR. get_network_config()
        # uses the Ray head address, so rank 0's bundle must be scheduled on the head.
        bundles[0][_RAY_HEAD_NODE_RESOURCE] = _RAY_NODE_RESOURCE_PIN
    pg = placement_group(bundles=bundles, strategy=strategy)
    ray.get(pg.ready())
    return pg


def get_worker_options(gpu_ratio, num_cpus, pg, rank, local_rank, num_worker, master_addr, master_port):
    """Create options for Ray workers."""
    env_vars = _build_worker_env_vars(
        num_worker=num_worker,
        rank=rank,
        local_rank=local_rank,
        master_addr=master_addr,
        master_port=master_port,
    )
    return {
        "runtime_env": {"env_vars": env_vars},
        "scheduling_strategy": PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_bundle_index=rank),
        "num_cpus": num_cpus,
        "num_gpus": gpu_ratio,
    }


def _build_worker_env_vars(
    num_worker,
    rank,
    local_rank,
    master_addr,
    master_port,
    local_world_size=1,
):
    env_vars = {
        "WORLD_SIZE": str(num_worker),
        "RANK": str(rank),
        "MASTER_ADDR": master_addr,
        "MASTER_PORT": str(master_port),
        "LOCAL_RANK": str(local_rank),
        "LOCAL_WORLD_SIZE": str(local_world_size),
    }
    if os.getenv("GLOO_SOCKET_IFNAME"):
        env_vars["GLOO_SOCKET_IFNAME"] = os.environ["GLOO_SOCKET_IFNAME"]
    return env_vars


def build_ray_worker_env_vars(
    num_worker: int,
    node_ids: Sequence[str],
    master_addr: str,
    master_port: int,
) -> list[dict[str, str]]:
    """Build distributed environment variables from the actual Ray node layout."""
    if len(node_ids) != num_worker:
        raise ValueError("node_ids must contain one node id for every Ray worker")  # noqa: TRY003
    if not node_ids:
        return []

    local_world_sizes = Counter(node_ids)
    local_ranks: Counter[str] = Counter()
    env_vars = []
    for rank, node_id in enumerate(node_ids):
        local_rank = local_ranks[node_id]
        local_ranks[node_id] += 1
        env_vars.append(
            _build_worker_env_vars(
                num_worker=num_worker,
                rank=rank,
                local_rank=local_rank,
                master_addr=master_addr,
                master_port=master_port,
                local_world_size=local_world_sizes[node_id],
            )
        )
    return env_vars


def get_network_config():
    """Get network configuration for distributed setup."""
    master_addr = ray.util.get_node_ip_address()
    with socket.socket() as sock:
        sock.bind(("", 0))
        master_port = sock.getsockname()[1]
    return master_addr, master_port


def get_num_worker_options(
    pg,
    num_worker,
    gpu_ratio=1.0,
    num_cpus_per_worker=None,
    master_addr=None,
    master_port=None,
):
    """Create options for multiple Ray workers with GPU allocation."""

    if num_cpus_per_worker is None:
        num_cpus_per_worker = pg.bundle_specs[0].get("CPU", 0)

    if master_addr is None or master_port is None:
        master_addr, master_port = get_network_config()
    options_list = []
    for i in range(num_worker):
        options = get_worker_options(
            gpu_ratio,
            num_cpus_per_worker,
            pg,
            i,
            i,
            num_worker,
            master_addr,
            master_port,
        )
        options_list.append(options)
    return options_list
