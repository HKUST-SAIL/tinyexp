import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch
from loguru._logger import Logger
from omegaconf.listconfig import ListConfig

from ..exceptions import UnsupportedCheckpointFormatError
from ..utils.log_utils import tiny_logger_setup


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


@dataclass
class RayCfgMixin:
    @dataclass
    class RayCfg:
        # ---------------- luancher configuration ---------------- #
        ray_num_worker: int = -1  # Number of Ray workers, -1 means using all Ray cluster GPU resources.
        ray_num_gpus_per_worker: float = 1.0  # Number of GPUs per Ray worker, should be a float value between 0 and 1.
        ray_placement_strategy: str = "PACK"

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
            if extra_state is not None:
                checkpoint["extra_state"] = extra_state

            torch.save(checkpoint, save_path)
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

        def load_checkpoint(
            self,
            path: str,
            *,
            model=None,
            optimizer=None,
            scheduler=None,
            strict: bool = True,
            map_location=None,
        ) -> dict[str, Any]:
            checkpoint = self._validate_checkpoint_payload(path, torch.load(path, map_location=map_location))
            self._load_required_state(
                checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                strict=strict,
            )

            return checkpoint

    checkpoint_cfg: CheckpointCfg = field(default_factory=CheckpointCfg)
