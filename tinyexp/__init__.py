__author__ = "LI Zeming"
__email__ = "zane.li@connect.ust.hk"
__license__ = "MIT"

import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch
from hydra.conf import HydraConf, RunDir
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from omegaconf.listconfig import ListConfig

from .exceptions import UnknownConfigurationKeyError, UnsupportedCheckpointFormatError
from .utils.log_utils import tiny_logger_setup
from .utils.ray_utils import simple_launch_exp

__all__ = [
    "CheckpointCfg",
    "ConfigStore",
    "RedisCfgMixin",
    "TinyExp",
    "simple_launch_exp",
]


@dataclass
class _HydraConfig(HydraConf):
    """
    To avoid hydra output the config in unexpected directory.
    """

    output_subdir: Optional[str] = None
    run: RunDir = field(default_factory=lambda: RunDir("./output"))


def _default_exp_name() -> str:
    """
    Get the default experiment name from the main module or the command line.
    e.g. if the main module is `resnet_exp.py`, the experiment name will be `resnet_exp`.
    """
    main_module = sys.modules.get("__main__")
    main_file = getattr(main_module, "__file__", None)
    if isinstance(main_file, str) and main_file:
        return os.path.splitext(os.path.basename(main_file))[0]

    argv0 = sys.argv[0] if sys.argv else ""
    if argv0 and argv0 != "-c":
        return os.path.splitext(os.path.basename(argv0))[0]

    return "exp"


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
            raise TypeError(f"Checkpoint at {path} must be a dict, got {type(checkpoint).__name__}.")  # noqa: TRY003

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


@dataclass
class TinyExp:
    """
    Simple experiment configuration class, which use hydra to manage and override configurations.
    The core idea is to provide a unified interface for experiment configurations, which can be instantiated
    and used in various contexts, such as Ray or TorchRun.
    """

    hydra: _HydraConfig = field(default_factory=_HydraConfig)

    # ---------------- luancher configuration ---------------- #
    ray_num_worker: int = -1  # Number of Ray workers, -1 means using all Ray cluster GPU resources.
    ray_num_gpus_per_worker: float = 1.0  # Number of GPUs per Ray worker, should be a float value between 0 and 1.
    ray_placement_strategy: str = "PACK"

    # Fully qualified import path for the experiment class, e.g. "tinyexp.examples.mnist_exp.Exp".
    # It is used in Hydra config store to instantiate the experiment class, and in store_and_run_exp, the exp_class will be automatically set to the fully qualified import path of the experiment class.
    # If you do not use store_and_run_exp, you can set this field to an empty string.
    exp_class: str = ""

    # The experiment name, will be used as the subdirectory name in the output directory.
    # If not provided, the default experiment name will be the name of the main module or the command line.
    # e.g. if the main module is `resnet_exp.py`, the experiment name will be `resnet_exp`.
    exp_name: str = field(default_factory=_default_exp_name)

    # log directory
    output_root: str = "./output"
    mode: str = "train"
    resume_from: str = ""

    # overridden configurations, only for internal use
    overrided_cfg: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __repr__(self):
        # Customize the representation of the Exp object for cleaner Ray logs.
        return f"Exp(rank={os.getenv('RANK', 'N/A')})"

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
    class LoggerCfg:
        def build_logger(
            self,
            save_dir: str,
            distributed_rank: int = 0,
            filename: str = "log.txt",
            mode: str = "a",
        ):
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            logger = tiny_logger_setup(save_dir, distributed_rank, filename, mode)
            logger.info(f"==> log file: {os.path.join(save_dir, filename)}")
            return logger

    logger_cfg: LoggerCfg = field(default_factory=LoggerCfg)
    checkpoint_cfg: CheckpointCfg = field(default_factory=CheckpointCfg)

    def get_run_dir(self) -> str:
        return os.path.join(self.output_root, self.exp_name)

    def set_cfg(self, cfg_hydra, cfg_object=None):
        if cfg_object is None:
            cfg_object = self
        for key, value in cfg_hydra.items():
            if hasattr(cfg_object, key):
                if isinstance(value, (DictConfig, dict)):
                    # If the value is a dictionary, recursively set attributes
                    sub_object = getattr(cfg_object, key)
                    self.set_cfg(value, sub_object)
                else:
                    # Otherwise, set the attribute directly
                    ori_value = getattr(cfg_object, key, None)
                    if ori_value != value:
                        self.overrided_cfg[key] = {
                            "value": value,
                            "original": ori_value,
                        }
                        setattr(cfg_object, key, value)
            else:
                raise UnknownConfigurationKeyError(key)
        return cfg_object

    def print_cfg(self, logger, show_overrided: bool = True):  # type: ignore[no-untyped-def]
        if show_overrided and self.overrided_cfg:
            override_lines = [
                f"    {key}: {item['value']} <-- {item['original']}(original)"
                for key, item in self.overrided_cfg.items()
            ]
            override_msg = "\n".join(override_lines)
            logger.info(f"-------- Overridden Configurations --------\n{override_msg}")

        cfg_dict = OmegaConf.to_container(OmegaConf.structured(self), resolve=True)
        del cfg_dict["hydra"]
        del cfg_dict["overrided_cfg"]
        cfg_msg = OmegaConf.to_yaml(cfg_dict).strip().replace("\n", "\n    ")
        logger.info(f"-------- Configurations --------\n    {cfg_msg}")
        return cfg_dict


@dataclass
class RedisCfgMixin:
    """Supplies :attr:`redis_cache_cfg` plus thin Ray entrypoints on the *experiment* object (see below)."""

    @dataclass
    class RedisCacheCfg:
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

    redis_cache_cfg: RedisCacheCfg = field(default_factory=RedisCacheCfg)


def store_and_run_exp(exp_class: type[TinyExp]) -> None:
    """
    Extract the config from the exp_class and store it in the ConfigStore(hydra config store).
    Then launch the experiment with the config.

    Args:
        exp_class: The class of the experiment to run.

    Returns:
        None: This function does not return anything.
    """

    # this is the hack for hydra to find the experiment class
    exp_class_path = f"{exp_class.__module__}.{exp_class.__qualname__}"
    exp_cfg = exp_class()
    exp_cfg.exp_class = exp_class_path

    # store the experiment configuration in the ConfigStore and launch the experiment
    ConfigStore.instance().store(name="cfg", node=exp_cfg)
    simple_launch_exp()
