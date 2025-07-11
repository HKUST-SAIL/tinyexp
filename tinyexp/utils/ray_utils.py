import os
import socket

import hydra
import ray
from omegaconf import DictConfig, OmegaConf
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


@hydra.main(version_base=None, config_name="cfg")
def simple_ray_launch_exp(cfg: DictConfig) -> None:
    """This is a template for launching a Ray-based experiment."""
    print(OmegaConf.to_yaml(cfg))

    exp_class = hydra.utils.get_class(cfg.exp_class)
    if cfg.launch == "ray":
        ray.init()

        # hold actor list to avoid garbage collection, otherwise the actors will be garbage collected
        actor_list = exp_class.after_ray_init_callback(cfg)

        requested_cpu = cfg.num_gpus * (cfg.train_data_worker_per_gpu + cfg.val_data_worker_per_gpu + 1)
        if requested_cpu > os.cpu_count():
            raise RuntimeError(
                f"Total CPU count {os.cpu_count()} is not enough for the experiment, "
                f"please set `num_gpus * (train_data_worker_per_gpu + val_data_worker_per_gpu + 1)`"
                f"<= {os.cpu_count()}"
            )

        remote_exp = ray.remote(exp_class)
        options_list = get_num_gpus_worker_options(
            cfg.num_gpus, num_cpus_per_gpu=cfg.train_data_worker_per_gpu + cfg.val_data_worker_per_gpu + 1
        )

        worker_group = [remote_exp.options(**options).remote(cfg) for options in options_list]
        run_futures = [worker.run.remote() for worker in worker_group]
        ray.get(run_futures)

    else:
        exp_class(cfg).run()
