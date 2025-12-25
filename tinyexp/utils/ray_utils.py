import os
import socket
from typing import Any

import hydra
import psutil
import ray
from omegaconf import DictConfig
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from ..exceptions import InsufficientCPUError, InvalidWorkerCountError, UnknownExperimentModeError, UnknownLauncherError


def _launch_with_ray(cfg: DictConfig, exp_class: type[Any]) -> None:
    ray.init()

    remote_exp = ray.remote(exp_class)

    # -------------------- allocate resources for redis cache ----------------- #
    cpu_need_list: list[int] = []
    if cfg.mode == "train" and hasattr(cfg, "redis_cache_cfg") and cfg.redis_cache_cfg.redis_cache_enabled:
        # hold actor list to avoid garbage collection, otherwise the actors will be garbage collected
        cpu_need_list.append(cfg.redis_cache_cfg.redis_cluster_manager_cpus)
        redis_actor = remote_exp.options(num_cpus=cfg.redis_cache_cfg.redis_cluster_manager_cpus).remote()

        ray.get(redis_actor.set_cfg.remote(cfg))
        ray.get(redis_actor.proxy_build_redis_cache.remote())

    # -------------------- check cpu count for run ----------------- #
    if cfg.mode not in {"train", "val", "help"}:
        raise UnknownExperimentModeError(cfg.mode)
    needed_num_cpus_per_worker = cfg.dataloader_cfg.val_data_worker_per_gpu + 1
    if cfg.mode == "train":
        needed_num_cpus_per_worker += cfg.dataloader_cfg.train_data_worker_per_gpu

    needed_cpu = cfg.num_worker * needed_num_cpus_per_worker
    total_cpu = os.cpu_count() or 0

    if needed_cpu + sum(cpu_need_list) > total_cpu:
        raise InsufficientCPUError(total_cpu=total_cpu, needed_cpu=needed_cpu + sum(cpu_need_list))

    # -------------------- allocate resources for run ----------------- #

    pg = get_placement_group(
        num_worker=cfg.num_worker,
        num_gpus_per_worker=cfg.num_gpus_per_worker,
        num_cpus_per_worker=needed_num_cpus_per_worker,
    )
    options_list = get_num_worker_options(
        pg,
        cfg.num_worker,
        gpu_ratio=cfg.num_gpus_per_worker,
    )
    worker_group = [remote_exp.options(**options).remote() for options in options_list]

    ray.get([worker.set_cfg.remote(cfg) for worker in worker_group])
    ray.get([worker.run.remote() for worker in worker_group])


def _should_print_launcher() -> bool:
    return os.getenv("RANK", "0") == "0"


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


def get_launcher():
    # Get the current process
    current_process = psutil.Process(os.getpid())
    process_chain = [current_process]

    # Trace up the process tree (up to 10 levels to avoid infinite loops)
    for _ in range(10):
        try:
            parent = current_process.parent()
        except (psutil.AccessDenied, psutil.NoSuchProcess, PermissionError):
            break
        if not parent or parent.pid == 1:  # Stop when reaching the root process (PID=1)
            break
        process_chain.append(parent)
        current_process = parent

    # Check if there is torchrun or python in the process chain
    for proc in process_chain:
        try:
            cmd_line = " ".join(proc.cmdline())
            proc_name = proc.name()
            # print(cmd_line, proc_name)
            if "torchrun" in cmd_line or "torchrun" in proc_name:
                return "torchrun"
            if "accelerate" in cmd_line or "accelerate" in proc_name:
                return "accelerate"
        except (psutil.AccessDenied, psutil.NoSuchProcess, PermissionError):
            continue

    return "python"


@hydra.main(version_base=None, config_name="cfg")
def simple_launch_exp(cfg: DictConfig) -> None:
    """
    This is a template for launching a experiment with hydra config.
    The launcher can be torchrun(multi-process), accelerate(multi-process), or python(ray).
    """
    exp_class = hydra.utils.get_class(cfg.exp_class)

    if cfg.mode == "help":
        from omegaconf import OmegaConf

        # Add ANSI color codes for colored output after '==>'
        RESET = "\033[0m"
        CYAN = "\033[96m"
        YELLOW = "\033[93m"

        print(f"{CYAN}==> Experiment Configurations (Available Configs){RESET}")
        print(OmegaConf.to_yaml(cfg).strip())
        print(f"{YELLOW}==> Overridden Configurations{RESET}")
        exp_instance = exp_class()
        exp_instance.set_cfg(cfg)
        print("\n")
        return

    launcher = get_launcher()

    if _should_print_launcher():
        print(f"==> use launcher:{launcher}")

    if cfg.num_worker <= 0:
        raise InvalidWorkerCountError(cfg.num_worker)

    if launcher == "python":
        _launch_with_ray(cfg, exp_class)

    elif launcher == "torchrun" or launcher == "accelerate":
        # if hasattr(cfg, "redis_cache_cfg") and cfg.redis_cache_cfg.redis_cache_enabled:
        #     cfg.redis_cache_cfg.redis_cache_enabled = False

        exp_class().set_cfg(cfg).run()
    else:
        raise UnknownLauncherError(launcher)
