import os
import socket
from contextlib import suppress
from typing import Any

import hydra
import psutil
import ray
from omegaconf import DictConfig
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from ..exceptions import (
    InsufficientCPUError,
    InvalidWorkerCountError,
    UnknownExperimentModeError,
    UnknownLauncherError,
)
from .redis_utils import RayRedisClusterManager

_RAY_HEAD_NODE_RESOURCE = "node:__internal_head__"
# Ray node resources are capacity-1 logical labels. A tiny fractional request pins
# bundle 0 to the head node without meaningfully consuming CPU/GPU resources.
_RAY_NODE_RESOURCE_PIN = 0.001


def _maybe_start_ray_redis_cache(cfg: DictConfig) -> RayRedisClusterManager | None:
    redis_cache_cfg = getattr(cfg, "redis_cache_cfg", None)
    if redis_cache_cfg is None or not bool(getattr(redis_cache_cfg, "redis_cache_enabled", False)):
        return None

    requested_world_size = int(getattr(redis_cache_cfg, "redis_rendezvous_world_size", 1))
    if requested_world_size == 1:
        cluster_enabled = False
    elif requested_world_size == -1:
        cluster_enabled = True
    elif requested_world_size > 1:
        return None
    else:
        raise ValueError("redis_rendezvous_world_size must be -1, 1, or > 1")  # noqa: TRY003

    manager = RayRedisClusterManager(redis_cache_cfg)
    startup_host, startup_ports, world_size = manager.start(cluster_enabled=cluster_enabled)
    redis_cache_cfg.redis_cluster_host = startup_host
    redis_cache_cfg.redis_cluster_ports = startup_ports
    redis_cache_cfg.redis_rendezvous_world_size = world_size
    return manager


def _launch_with_ray(cfg: DictConfig, exp_class: type[Any]) -> None:
    ray_num_worker = int(cfg.ray_num_worker)
    if ray_num_worker < -1 or ray_num_worker == 0:
        raise InvalidWorkerCountError(ray_num_worker)

    pg = None
    redis_manager = None
    worker_group = []

    ray.init()
    try:
        if ray_num_worker == -1:
            ray_num_worker = int(ray.cluster_resources().get("GPU", 0))
            if ray_num_worker <= 0:
                raise InvalidWorkerCountError(ray_num_worker)
            print(
                f"==> ray_num_worker is -1, using all Ray cluster GPU resources: {ray_num_worker}",
                flush=True,
            )
        cfg.ray_num_worker = ray_num_worker

        remote_exp = ray.remote(exp_class)

        # -------------------- check cpu count for run ----------------- #
        if cfg.mode not in {"train", "val", "help"}:
            raise UnknownExperimentModeError(cfg.mode)
        needed_num_cpus_per_worker = cfg.dataloader_cfg.val_data_worker_per_gpu + 1
        if cfg.mode == "train":
            needed_num_cpus_per_worker += cfg.dataloader_cfg.train_data_worker_per_gpu

        needed_cpu = cfg.ray_num_worker * needed_num_cpus_per_worker
        total_cpu = int(ray.cluster_resources().get("CPU", 0))

        if needed_cpu > total_cpu:
            raise InsufficientCPUError(total_cpu=total_cpu, needed_cpu=needed_cpu)

        redis_manager = _maybe_start_ray_redis_cache(cfg)

        # -------------------- allocate resources for run ----------------- #

        pg = get_placement_group(
            num_worker=cfg.ray_num_worker,
            num_gpus_per_worker=cfg.ray_num_gpus_per_worker,
            num_cpus_per_worker=needed_num_cpus_per_worker,
            strategy=cfg.ray_placement_strategy,
        )
        options_list = get_num_worker_options(
            pg,
            cfg.ray_num_worker,
            gpu_ratio=cfg.ray_num_gpus_per_worker,
            num_cpus_per_worker=needed_num_cpus_per_worker,
        )
        worker_group = [remote_exp.options(**options).remote() for options in options_list]

        ray.get([worker.set_cfg.remote(cfg) for worker in worker_group])
        ray.get([worker.run.remote() for worker in worker_group])
    finally:
        if pg is not None:
            with suppress(Exception):
                ray.util.remove_placement_group(pg)

        if redis_manager is not None:
            with suppress(Exception):
                redis_manager.stop()

        if ray.is_initialized():
            with suppress(Exception):
                ray.shutdown()


def get_placement_group(num_worker, num_gpus_per_worker=1, num_cpus_per_worker=10, strategy="PACK"):
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


def _build_worker_env_vars(num_worker, rank, local_rank, master_addr, master_port):
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


def get_network_config():
    """Get network configuration for distributed setup."""
    master_addr = ray._private.services.get_node_ip_address()
    with socket.socket() as sock:
        sock.bind(("", 0))
        master_port = sock.getsockname()[1]
    return master_addr, master_port


def get_num_worker_options(pg, num_worker, gpu_ratio=1.0, num_cpus_per_worker=None):
    """Create options for multiple Ray workers with GPU allocation."""

    if num_cpus_per_worker is None:
        num_cpus_per_worker = pg.bundle_specs[0].get("CPU", 0)

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


def get_launcher() -> str:
    # Launchers may be wrapped by tools like uv, so environment variables are the most reliable signal.
    accelerate_env_keys = (
        "ACCELERATE_USE_CPU",
        "ACCELERATE_PROCESS_INDEX",
        "ACCELERATE_LOCAL_PROCESS_INDEX",
        "ACCELERATE_MIXED_PRECISION",
        "ACCELERATE_DYNAMO_BACKEND",
    )
    torchelastic_run_id = os.getenv("TORCHELASTIC_RUN_ID")
    has_torchelastic_run_id = torchelastic_run_id not in (None, "", "none")
    has_rank_env = (
        os.getenv("LOCAL_RANK") is not None and os.getenv("RANK") is not None and os.getenv("WORLD_SIZE") is not None
    )

    if any(os.getenv(key) is not None for key in accelerate_env_keys):
        return "accelerate"
    elif has_torchelastic_run_id or has_rank_env:
        return "torchrun"

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

    for proc in process_chain:
        try:
            cmdline = proc.cmdline()
            proc_name = proc.name()
        except (psutil.AccessDenied, psutil.NoSuchProcess, PermissionError):
            continue

        executable = os.path.basename(cmdline[0]) if cmdline else proc_name
        if executable == "torchrun" or proc_name == "torchrun" or "torch.distributed.run" in cmdline:
            return "torchrun"
        if executable == "accelerate" or proc_name == "accelerate" or "accelerate.commands.launch" in cmdline:
            return "accelerate"

    return "python"


def _should_print_launcher() -> bool:
    return os.getenv("RANK", "0") == "0"


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
        YELLOW = "\033[93m"
        INDENT = "    "  # 4 spaces

        print(
            f"{YELLOW}==> Experiment Configurations (Available Configs):{RESET}",
            flush=True,
        )
        print(
            INDENT + OmegaConf.to_yaml(cfg).strip().replace("\n", f"\n{INDENT}"),
            flush=True,
        )
        exp_instance = exp_class()
        exp_instance.set_cfg(cfg)

        class _StdoutLogger:
            def info(self, message):  # type: ignore[no-untyped-def]
                print(message, flush=True)

        exp_instance.print_cfg(_StdoutLogger())
        print("\n", flush=True)
        return

    launcher = get_launcher()

    if _should_print_launcher():
        print(f"==> use launcher:{launcher}", flush=True)

    if launcher == "python":
        _launch_with_ray(cfg, exp_class)

    elif launcher == "torchrun" or launcher == "accelerate":
        exp_class().set_cfg(cfg).run()
    else:
        raise UnknownLauncherError(launcher)
