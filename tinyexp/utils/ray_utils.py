import os
import socket

import hydra
import ray
from omegaconf import DictConfig, OmegaConf
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


def get_placement_group(num_worker, num_gpus_per_worker=1, num_cpus_per_worker=10):
    """Create and return a placement group for GPU allocation."""
    bundles = [{"CPU": num_cpus_per_worker, "GPU": num_gpus_per_worker} for _ in range(num_worker)]
    pg = placement_group(bundles=bundles, strategy="STRICT_PACK")
    ray.get(pg.ready())
    return pg


def get_worker_options(gpu_ratio, pg, rank, local_rank, num_worker, master_addr, master_port):
    """Create options for Ray workers."""
    return {
        "runtime_env": {
            "env_vars": {
                "WORLD_SIZE": str(num_worker),
                "RANK": str(rank),
                "MASTER_ADDR": master_addr,
                "MASTER_PORT": str(master_port),
                "LOCAL_RANK": str(local_rank),
            }
        },
        "scheduling_strategy": PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_bundle_index=rank),
        "num_gpus": gpu_ratio,
    }


def get_network_config():
    """Get network configuration for distributed setup."""
    master_addr = ray._private.services.get_node_ip_address()
    with socket.socket() as sock:
        sock.bind(("", 0))
        master_port = sock.getsockname()[1]
    return master_addr, master_port


def get_num_worker_options(pg, num_worker, gpu_ratio=1.0):
    """Create options for multiple Ray workers with GPU allocation."""

    master_addr, master_port = get_network_config()
    options_list = []
    for i in range(num_worker):
        options = get_worker_options(gpu_ratio, pg, i, i, num_worker, master_addr, master_port)
        options_list.append(options)
    return options_list


@hydra.main(version_base=None, config_name="cfg")
def simple_ray_launch_exp(cfg: DictConfig) -> None:
    """This is a template for launching a Ray-based experiment."""
    exp_class = hydra.utils.get_class(cfg.exp_class)
    if cfg.num_worker <= 0:
        raise ValueError(f"Number of workers must be greater than 0, got {cfg.num_worker}.")

    if cfg.launcher == "ray":
        ray.init()

        remote_exp = ray.remote(exp_class)

        # -------------------- allocate resources for redis cache ----------------- #
        cpu_need_list = []
        if hasattr(cfg, "redis_cache_cfg") and cfg.redis_cache_cfg.redis_cache_enabled:
            # hold actor list to avoid garbage collection, otherwise the actors will be garbage collected
            cpu_need_list.append(cfg.redis_cache_cfg.redis_cluster_manager_cpus)
            redis_actor = remote_exp.options(num_cpus=cfg.redis_cache_cfg.redis_cluster_manager_cpus).remote()

            ray.get(redis_actor.set_cfg.remote(cfg))
            ray.get(redis_actor.proxy_build_redis_cache.remote())

        # -------------------- check cpu count for run ----------------- #
        requested_cpu = cfg.num_worker * (
            cfg.dataloader_cfg.train_data_worker_per_gpu + cfg.dataloader_cfg.val_data_worker_per_gpu + 1
        )
        if requested_cpu + sum(cpu_need_list) > os.cpu_count():
            raise RuntimeError(
                f"Total CPU count {os.cpu_count()} is not enough for the experiment, "
                f"please set `num_worker * (train.data_worker_per_gpu + val.data_worker_per_gpu + 1)`"
                f"<= {os.cpu_count()}"
            )

        # -------------------- allocate resources for run ----------------- #
        pg = get_placement_group(
            num_worker=cfg.num_worker,
            num_gpus_per_worker=cfg.num_gpus_per_worker,
            num_cpus_per_worker=cfg.dataloader_cfg.train_data_worker_per_gpu
            + cfg.dataloader_cfg.val_data_worker_per_gpu
            + 1,
        )
        options_list = get_num_worker_options(
            pg,
            cfg.num_worker,
            gpu_ratio=cfg.num_gpus_per_worker,
        )
        worker_group = [remote_exp.options(**options).remote() for options in options_list]

        run_futures = [worker.set_cfg.remote(cfg) for worker in worker_group]
        ray.get(run_futures)
        run_futures = [worker.run.remote() for worker in worker_group]
        ray.get(run_futures)

    elif cfg.launcher == "torchrun":
        exp_class().set_cfg(cfg).run()
    else:
        raise ValueError(f"Unknown launcher {cfg.launcher}, please set `launcher` to 'ray' or 'torchrun'.")
