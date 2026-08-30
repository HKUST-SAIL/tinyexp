from __future__ import annotations

import os
import socket
from collections import Counter
from collections.abc import Sequence
from contextlib import suppress
from datetime import timedelta
from typing import Any, Optional

import ray
from omegaconf import DictConfig
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from .redis_utils import RayRedisClusterManager


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
    pg = placement_group(bundles=bundles, strategy=strategy)
    try:
        ray.get(pg.ready(), timeout=timeout_s)
    except ray.exceptions.GetTimeoutError as exc:
        try:
            total_resources = ray.cluster_resources()
        except Exception:
            total_resources = {}
        try:
            available_resources = ray.available_resources()
        except Exception:
            available_resources = {}
        with suppress(Exception):
            ray.util.remove_placement_group(pg)
        raise TimeoutError(  # noqa: TRY003
            f"Ray placement group timed out after {timeout_s}s: "
            f"workers={num_worker}, CPU/worker={num_cpus_per_worker}, "
            f"GPU/worker={num_gpus_per_worker}, strategy={strategy}; "
            f"requested CPU={num_worker * num_cpus_per_worker}, "
            f"requested GPU={num_worker * num_gpus_per_worker}; "
            f"total CPU={total_resources.get('CPU', 'unknown')}, "
            f"total GPU={total_resources.get('GPU', 'unknown')}; "
            f"available CPU={available_resources.get('CPU', 'unknown')}, "
            f"available GPU={available_resources.get('GPU', 'unknown')}"
        ) from exc
    except Exception:
        with suppress(Exception):
            ray.util.remove_placement_group(pg)
        raise
    return pg


class _RayRendezvousStoreActor:
    def __init__(self, world_size: int, timeout_s: float) -> None:
        import torch.distributed as dist

        self._master_addr = ray.util.get_node_ip_address()
        self._store = dist.TCPStore(
            host_name=self._master_addr,
            port=0,
            world_size=world_size,
            is_master=True,
            timeout=timedelta(seconds=timeout_s),
            wait_for_workers=False,
        )

    def get_endpoint(self) -> tuple[str, int]:
        return self._master_addr, int(self._store.port)


def start_ray_rendezvous_store(
    pg: Any,
    world_size: int,
    timeout_s: float,
) -> tuple[Any, str, int]:
    """Start a TCPStore server in placement-group bundle 0 and return its endpoint."""
    if world_size <= 1:
        raise ValueError("Ray rendezvous store requires world_size greater than 1")  # noqa: TRY003

    remote_actor_cls = ray.remote(num_cpus=0)(_RayRendezvousStoreActor)
    actor = remote_actor_cls.options(
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_bundle_index=0,
        )
    ).remote(world_size, timeout_s)
    try:
        master_addr, master_port = ray.get(actor.get_endpoint.remote(), timeout=timeout_s)
    except ray.exceptions.GetTimeoutError as exc:
        with suppress(Exception):
            ray.kill(actor, no_restart=True)
        raise TimeoutError(  # noqa: TRY003
            f"Ray rendezvous store timed out after {timeout_s}s while starting on placement-group bundle 0"
        ) from exc
    except Exception:
        with suppress(Exception):
            ray.kill(actor, no_restart=True)
        raise
    return actor, str(master_addr), int(master_port)


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
    if num_worker > 1:
        env_vars["TORCHELASTIC_USE_AGENT_STORE"] = "True"
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


def get_network_config(master_node_id: str | None = None) -> tuple[str, int]:
    """Get the distributed master address and an available port."""
    if master_node_id is None:
        master_addr = ray.util.get_node_ip_address()
    else:
        master_node = next(
            (node for node in ray.nodes() if node.get("Alive", True) and str(node.get("NodeID")) == master_node_id),
            None,
        )
        if master_node is None:
            raise RuntimeError(f"Ray master node {master_node_id} is not alive or is missing")  # noqa: TRY003
        master_addr = master_node.get("NodeManagerAddress")
        if not isinstance(master_addr, str) or not master_addr:
            raise RuntimeError(f"Ray master node {master_node_id} is missing NodeManagerAddress")  # noqa: TRY003

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

    if num_worker > 1 and (master_addr is None or master_port is None):
        raise ValueError(  # noqa: TRY003
            "multi-worker Ray runs require master_addr/master_port from start_ray_rendezvous_store()"
        )
    if master_addr is None or master_port is None:
        master_addr, master_port = get_network_config(node_ids[0])
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
