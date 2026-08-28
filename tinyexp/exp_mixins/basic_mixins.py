import os
import random
import tempfile
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import ray
import torch
from loguru._logger import Logger
from omegaconf import DictConfig
from omegaconf.listconfig import ListConfig

from ..exceptions import (
    InsufficientCPUError,
    InvalidWorkerCountError,
    UnsupportedCheckpointFormatError,
)
from ..utils.log_utils import tiny_logger_setup
from ..utils.ray_utils import (
    _maybe_start_ray_redis_cache,
    get_network_config,
    get_num_worker_options,
    get_placement_group,
    get_placement_group_node_ids,
    start_ray_rendezvous_store,
)


@dataclass
class LoggerCfgMixin:
    @dataclass
    class LoggerCfg:
        def build_logger(
            self,
            save_dir: str,
            distributed_rank: int = 0,
            filename: str = "log.txt",
            mode: str = "a",
        ) -> Logger:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            logger = tiny_logger_setup(save_dir, distributed_rank, filename, mode)
            logger.info(f"==> log file: {os.path.join(save_dir, filename)}")
            return logger

    logger_cfg: LoggerCfg = field(default_factory=LoggerCfg)


def _resolve_ray_num_worker(
    ray_num_worker: int,
    num_cpus_per_worker: int,
    num_gpus_per_worker: float,
    cluster_resources: dict[str, float],
) -> int:
    if ray_num_worker != -1:
        return ray_num_worker

    cpu_worker_count = int(cluster_resources.get("CPU", 0) // num_cpus_per_worker)
    if num_gpus_per_worker > 0:
        gpu_worker_count = int(cluster_resources.get("GPU", 0) // num_gpus_per_worker)
        resolved_worker_count = min(cpu_worker_count, gpu_worker_count)
        resource_name = "GPU and CPU"
    else:
        resolved_worker_count = cpu_worker_count
        resource_name = "CPU"

    if resolved_worker_count <= 0:
        raise InvalidWorkerCountError(resolved_worker_count)
    print(
        f"==> ray_num_worker is -1, using available {resource_name} resources: {resolved_worker_count}",
        flush=True,
    )
    return resolved_worker_count


@dataclass
class RayCfgMixin:
    @dataclass
    class RayCfg:
        # ---------------- luancher configuration ---------------- #
        ray_num_worker: int = -1  # Number of Ray workers, -1 means fill available worker resources.
        ray_num_cpus_per_worker: int = 1
        ray_num_gpus_per_worker: float = 1.0
        ray_placement_strategy: str = "PACK"
        ray_placement_timeout_s: float = 120.0

        @classmethod
        def run(cls, exp_class: type[Any], experiment_cfg: DictConfig) -> None:  # noqa: C901
            """
            Run the Ray driver-side orchestration.

            Args:
                exp_class: Experiment class instantiated by each Ray worker.
                experiment_cfg: Complete Hydra configuration for exp_class.
            """
            ray_cfg = experiment_cfg.ray_cfg
            num_cpus_per_worker = int(ray_cfg.ray_num_cpus_per_worker)
            num_gpus_per_worker = float(ray_cfg.ray_num_gpus_per_worker)
            if num_cpus_per_worker <= 0:
                raise ValueError("ray_num_cpus_per_worker must be greater than 0")  # noqa: TRY003
            if num_gpus_per_worker < 0:
                raise ValueError("ray_num_gpus_per_worker must not be negative")  # noqa: TRY003

            placement_timeout_s = float(getattr(ray_cfg, "ray_placement_timeout_s", 120.0))
            if placement_timeout_s <= 0:
                raise ValueError("ray_placement_timeout_s must be greater than 0")  # noqa: TRY003

            if ray_cfg.ray_num_worker < -1 or ray_cfg.ray_num_worker == 0:
                raise InvalidWorkerCountError(ray_cfg.ray_num_worker)
            ray.init()
            pg = None
            redis_manager = None
            rendezvous_actor = None
            worker_group = []

            try:
                cluster_resources = ray.cluster_resources()
                ray_num_worker = _resolve_ray_num_worker(
                    ray_cfg.ray_num_worker,
                    num_cpus_per_worker,
                    num_gpus_per_worker,
                    cluster_resources,
                )
                if ray_cfg.ray_num_worker == -1:
                    ray_cfg.ray_num_worker = ray_num_worker

                remote_exp = ray.remote(exp_class)

                needed_num_cpus_per_worker = num_cpus_per_worker

                needed_cpu = ray_num_worker * needed_num_cpus_per_worker
                needed_gpu = ray_num_worker * num_gpus_per_worker
                total_cpu = int(cluster_resources.get("CPU", 0))
                total_gpu = float(cluster_resources.get("GPU", 0))

                if needed_cpu > total_cpu or needed_gpu > total_gpu:
                    raise InsufficientCPUError(
                        total_cpu=total_cpu,
                        needed_cpu=needed_cpu,
                        total_gpu=total_gpu,
                        needed_gpu=needed_gpu,
                    )

                redis_manager = _maybe_start_ray_redis_cache(experiment_cfg)
                # -------------------- allocate resources for run ----------------- #

                pg = get_placement_group(
                    num_worker=ray_cfg.ray_num_worker,
                    num_gpus_per_worker=num_gpus_per_worker,
                    num_cpus_per_worker=needed_num_cpus_per_worker,
                    strategy=ray_cfg.ray_placement_strategy,
                    timeout_s=placement_timeout_s,
                )
                worker_node_ids = get_placement_group_node_ids(pg, ray_cfg.ray_num_worker)
                if ray_cfg.ray_num_worker > 1:
                    rendezvous_actor, master_addr, master_port = start_ray_rendezvous_store(
                        pg,
                        ray_cfg.ray_num_worker,
                        placement_timeout_s,
                    )
                else:
                    master_addr, master_port = get_network_config(worker_node_ids[0])
                options_list = get_num_worker_options(
                    pg,
                    ray_cfg.ray_num_worker,
                    gpu_ratio=num_gpus_per_worker,
                    num_cpus_per_worker=needed_num_cpus_per_worker,
                    master_addr=master_addr,
                    master_port=master_port,
                    node_ids=worker_node_ids,
                )
                runtime_envs = [options["runtime_env"]["env_vars"] for options in options_list]
                print("==> Ray worker topology:", flush=True)
                node_ranks = {node_id: node_rank for node_rank, node_id in enumerate(dict.fromkeys(worker_node_ids))}
                topology = sorted(
                    zip(worker_node_ids, runtime_envs),
                    key=lambda item: int(item[1]["RANK"]),
                )
                for node_id, env_vars in topology:
                    print(
                        f"    rank={env_vars['RANK']}/{ray_cfg.ray_num_worker} node={node_id} "
                        f"node_rank={node_ranks[node_id]} "
                        f"local_rank={env_vars['LOCAL_RANK']}",
                        flush=True,
                    )
                worker_group = [remote_exp.options(**options).remote() for options in options_list]
                ray.get([worker.set_cfg.remote(experiment_cfg) for worker in worker_group])
                ray.get([worker.run.remote() for worker in worker_group])
            finally:
                if rendezvous_actor is not None:
                    with suppress(Exception):
                        ray.kill(rendezvous_actor, no_restart=True)

                if pg is not None:
                    with suppress(Exception):
                        ray.util.remove_placement_group(pg)

                if redis_manager is not None:
                    with suppress(Exception):
                        redis_manager.stop()

                if ray.is_initialized():
                    with suppress(Exception):
                        # Drain Ray's asynchronous worker logs before tearing down the runtime.
                        ray.shutdown(_exiting_interpreter=True)

    ray_cfg: RayCfg = field(default_factory=RayCfg)


@dataclass
class WandbCfgMixin:
    @dataclass
    class WandbCfg:
        enable_wandb: bool = False

        def build_wandb(self, accelerator=None, **kwargs):
            if self.enable_wandb:
                import wandb

                if accelerator is None or accelerator.rank == 0:
                    wandb.init(**kwargs)
                return wandb

    wandb_cfg: WandbCfg = field(default_factory=WandbCfg)


@dataclass
class RedisCfgMixin:
    """Supplies :attr:`redis_cfg` plus thin Ray entrypoints on the *experiment* object (see below)."""

    @dataclass
    class RedisCfg:
        """
        Hydra-overridable options live in dataclass fields below.

        Single-machine runs use standalone Redis. Multi-machine runs can use Redis Cluster.
        Redis lifecycle is owned by the launcher: ``tinyexp-run-with-redis`` for
        wrapper-based runs, or Ray-managed Redis for Ray launches.
        """

        redis_cache_enabled: bool = True
        redis_cluster_host: str = "127.0.0.1"
        redis_cluster_ports: ListConfig = field(
            default_factory=lambda: ListConfig([7000, 7001, 7002, 7003, 7004, 7005])
        )
        redis_cache_max_memory: int = 160  # Maximum memory is 160GB, according to the ImageNet dataset size

        # 1: standalone Redis. In Ray launches, this starts one local Redis per alive Ray node.
        # -1: Ray-managed Redis Cluster using the alive Ray nodes.
        # >1: externally managed Redis Cluster rendezvous world size.
        redis_rendezvous_world_size: int = 1

    redis_cfg: RedisCfg = field(default_factory=RedisCfg)


@dataclass
class CheckpointCfgMixin:
    @dataclass
    class CheckpointCfg:
        last_ckpt_name: str = "last.ckpt"
        best_ckpt_name: str = "best.ckpt"

        def save_checkpoint(
            self,
            *,
            run_dir: str,
            name: str,
            model=None,
            optimizer=None,
            scheduler=None,
            scaler=None,
            epoch: Optional[int] = None,
            global_step: Optional[int] = None,
            best_metric: Optional[float] = None,
            exp_name: str = "",
            exp_class: str = "",
            extra_state: Optional[dict[str, Any]] = None,
        ) -> str:
            save_path = Path(run_dir) / name
            save_path.parent.mkdir(parents=True, exist_ok=True)

            checkpoint: dict[str, Any] = {
                "epoch": epoch,
                "global_step": global_step,
                "best_metric": best_metric,
                "meta": {
                    "exp_name": exp_name,
                    "exp_class": exp_class,
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                },
            }
            if model is not None:
                checkpoint["model_state_dict"] = model.state_dict()
            if optimizer is not None:
                checkpoint["optimizer_state_dict"] = optimizer.state_dict()
            if scheduler is not None:
                checkpoint["scheduler_state_dict"] = scheduler.state_dict()
            if scaler is not None:
                checkpoint["scaler_state_dict"] = scaler.state_dict()
            if extra_state is not None:
                checkpoint["extra_state"] = extra_state

            temp_path: Optional[Path] = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=save_path.parent,
                    prefix=f".{save_path.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as temp_file:
                    temp_path = Path(temp_file.name)
                    torch.save(checkpoint, temp_file)
                    temp_file.flush()
                    os.fsync(temp_file.fileno())
                os.replace(temp_path, save_path)
            except Exception:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)
                raise
            return str(save_path)

        def _validate_checkpoint_payload(self, path: str, checkpoint: Any) -> dict[str, Any]:
            if not isinstance(checkpoint, dict):
                raise TypeError(  # noqa: TRY003
                    f"Checkpoint at {path} must be a dict, got {type(checkpoint).__name__}."
                )

            if (
                not any(
                    key in checkpoint
                    for key in (
                        "epoch",
                        "global_step",
                        "best_metric",
                        "meta",
                        "extra_state",
                    )
                )
                and "model_state_dict" not in checkpoint
            ):
                raise UnsupportedCheckpointFormatError(path)

            return checkpoint

        def _load_required_state(
            self,
            checkpoint: dict[str, Any],
            *,
            model=None,
            optimizer=None,
            scheduler=None,
            scaler=None,
            strict: bool = True,
        ) -> None:
            if model is not None:
                if "model_state_dict" not in checkpoint:
                    raise KeyError("model_state_dict")
                model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
            if optimizer is not None:
                if "optimizer_state_dict" not in checkpoint:
                    raise KeyError("optimizer_state_dict")
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if scheduler is not None:
                if "scheduler_state_dict" not in checkpoint:
                    raise KeyError("scheduler_state_dict")
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if scaler is not None:
                if "scaler_state_dict" not in checkpoint:
                    raise KeyError("scaler_state_dict")
                scaler.load_state_dict(checkpoint["scaler_state_dict"])

        @staticmethod
        def capture_rng_state() -> dict[str, Any]:
            state: dict[str, Any] = {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
            }
            if torch.cuda.is_available():
                state["torch_cuda"] = torch.cuda.get_rng_state_all()
            return state

        @staticmethod
        def restore_rng_state(state: dict[str, Any]) -> None:
            random.setstate(state["python"])
            np.random.set_state(state["numpy"])
            torch.set_rng_state(state["torch"])
            if torch.cuda.is_available() and "torch_cuda" in state:
                torch.cuda.set_rng_state_all(state["torch_cuda"])

        def load_checkpoint(
            self,
            path: str,
            *,
            model=None,
            optimizer=None,
            scheduler=None,
            scaler=None,
            strict: bool = True,
            map_location=None,
        ) -> dict[str, Any]:
            checkpoint = self._validate_checkpoint_payload(
                path,
                torch.load(path, map_location=map_location, weights_only=False),
            )
            self._load_required_state(
                checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                strict=strict,
            )

            return checkpoint

    checkpoint_cfg: CheckpointCfg = field(default_factory=CheckpointCfg)
