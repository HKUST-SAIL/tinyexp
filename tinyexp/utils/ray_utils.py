from __future__ import annotations

import os
import socket
from collections import Counter
from collections.abc import Sequence
from contextlib import suppress
from typing import Any, Optional

import ray
from omegaconf import DictConfig
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from .redis_utils import RayRedisClusterManager

_RAY_HEAD_NODE_RESOURCE = "node:__internal_head__"
# Ray node resources are capacity-1 logical labels. A tiny fractional request pins
# bundle 0 to the head node without meaningfully consuming CPU/GPU resources.
_RAY_NODE_RESOURCE_PIN = 0.001


def _maybe_start_ray_redis_cache(
    cfg: DictConfig,
) -> Optional[RayRedisClusterManager]:  # noqa: UP007
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
    timeout_s: float = 120.0,
):
    """Create and return a placement group for worker allocation."""
    bundles = [{"CPU": num_cpus_per_worker, "GPU": num_gpus_per_worker} for _ in range(num_worker)]
    cluster_resources = ray.cluster_resources() if ray.is_initialized() else {}
    if num_worker > 1 and _RAY_HEAD_NODE_RESOURCE in cluster_resources:
        # PyTorch env:// starts TCPStore on rank 0 at MASTER_ADDR. get_network_config()
        # uses the Ray head address, so rank 0's bundle must be scheduled on the head.
        bundles[0][_RAY_HEAD_NODE_RESOURCE] = _RAY_NODE_RESOURCE_PIN
    pg = placement_group(bundles=bundles, strategy=strategy)
    try:
        ray.get(pg.ready(), timeout=timeout_s)
    except ray.exceptions.GetTimeoutError as exc:
        with suppress(Exception):
            ray.util.remove_placement_group(pg)
        raise TimeoutError(  # noqa: TRY003
            f"Ray placement group timed out after {timeout_s}s: "
            f"workers={num_worker}, CPU/worker={num_cpus_per_worker}, "
            f"GPU/worker={num_gpus_per_worker}, strategy={strategy}"
        ) from exc
    except Exception:
        with suppress(Exception):
            ray.util.remove_placement_group(pg)
        raise
    return pg


def get_worker_options(
    gpu_ratio: float,
    num_cpus: int,
    pg: Any,
    bundle_index: int,
    env_vars: dict[str, str],
) -> dict[str, Any]:
    """Create Ray actor options for one worker and one placement-group bundle."""
    return {
        "runtime_env": {"env_vars": env_vars},
        "scheduling_strategy": PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_bundle_index=bundle_index,
        ),
        "num_cpus": num_cpus,
        "num_gpus": gpu_ratio,
    }


def _build_worker_env_vars(
    *,
    num_worker: int,
    rank: int,
    local_rank: int,
    master_addr: str,
    master_port: int,
) -> dict[str, str]:
    env_vars = {
        "WORLD_SIZE": str(num_worker),
        "RANK": str(rank),
        "MASTER_ADDR": master_addr,
        "MASTER_PORT": str(master_port),
        "LOCAL_RANK": str(local_rank),
    }
    if os.getenv("GLOO_SOCKET_IFNAME"):
        env_vars["GLOO_SOCKET_IFNAME"] = os.environ["GLOO_SOCKET_IFNAME"]
    return env_vars


def get_placement_group_node_ids(pg: Any, num_worker: int) -> list[str]:
    """Return the node hosting each placement-group bundle in bundle-index order."""
    placement_group_state = ray.util.placement_group_table(pg)
    bundles_to_node_id = placement_group_state.get("bundles_to_node_id")
    if not isinstance(bundles_to_node_id, dict):
        raise RuntimeError("Ray did not return placement-group bundle-to-node assignments")  # noqa: TRY003, TRY004

    node_ids = []
    for bundle_index in range(num_worker):
        node_id = bundles_to_node_id.get(bundle_index)
        if node_id is None:
            node_id = bundles_to_node_id.get(str(bundle_index))
        if node_id is None:
            raise RuntimeError(  # noqa: TRY003
                f"Ray did not return a node assignment for placement-group bundle {bundle_index}"
            )
        node_ids.append(str(node_id))
    return node_ids


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

    worker_counts = Counter(node_ids)
    if len(set(worker_counts.values())) != 1:
        counts = ", ".join(f"{node_id}={count}" for node_id, count in worker_counts.items())
        raise ValueError(  # noqa: TRY003
            "Ray distributed workers must be homogeneous: every worker node must host the same "
            f"number of workers; counts: {counts}"
        )

    workers_per_node = next(iter(worker_counts.values()))
    node_ranks = {node_id: node_rank for node_rank, node_id in enumerate(dict.fromkeys(node_ids))}
    node_rank_offsets: dict[str, int] = {}
    rank_offset = 0
    for node_id in node_ranks:
        node_rank_offsets[node_id] = rank_offset
        rank_offset += workers_per_node

    local_ranks: Counter[str] = Counter()
    env_vars = []
    for node_id in node_ids:
        local_rank = local_ranks[node_id]
        local_ranks[node_id] += 1
        env_vars.append(
            _build_worker_env_vars(
                num_worker=num_worker,
                rank=node_rank_offsets[node_id] + local_rank,
                local_rank=local_rank,
                master_addr=master_addr,
                master_port=master_port,
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
    pg: Any,
    num_worker: int,
    gpu_ratio: float = 1.0,
    num_cpus_per_worker: int | None = None,
    master_addr: str | None = None,
    master_port: int | None = None,
    *,
    node_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Create actor options after the placement-group topology has been resolved."""

    if num_cpus_per_worker is None:
        num_cpus_per_worker = pg.bundle_specs[0].get("CPU", 0)

    if master_addr is None or master_port is None:
        master_addr, master_port = get_network_config()
    runtime_envs = build_ray_worker_env_vars(
        num_worker=num_worker,
        node_ids=node_ids,
        master_addr=master_addr,
        master_port=master_port,
    )
    return [
        get_worker_options(
            gpu_ratio=gpu_ratio,
            num_cpus=num_cpus_per_worker,
            pg=pg,
            bundle_index=bundle_index,
            env_vars=runtime_envs[bundle_index],
        )
        for bundle_index in range(num_worker)
    ]
