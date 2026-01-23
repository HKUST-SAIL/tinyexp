__author__ = "LI Zeming"
__email__ = "zane.li@connect.ust.hk"
__license__ = "MIT"

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from hydra.conf import HydraConf, RunDir
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from omegaconf.listconfig import ListConfig

from .exceptions import UnknownConfigurationKeyError
from .utils.log_utils import tiny_logger_setup
from .utils.ray_utils import simple_launch_exp

__all__ = ["ConfigStore", "RedisCfgMixin", "TinyExp", "simple_launch_exp"]


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
class TinyExp:
    """
    Simple experiment configuration class, which use hydra to manage and override configurations.
    The core idea is to provide a unified interface for experiment configurations, which can be instantiated
    and used in various contexts, such as Ray or TorchRun.
    """

    hydra: _HydraConfig = field(default_factory=_HydraConfig)

    # ---------------- luancher configuration ---------------- #
    num_worker: int = -1  # Number of workers, -1 means to be determined by the user
    num_gpus_per_worker: float = 1.0  # Number of GPUs per worker, should be a float value between 0 and 1.

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

    # overridden configurations, only for internal use
    overrided_cfg: dict = field(default_factory=dict)

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
        def build_logger(self, save_dir: str, distributed_rank: int = 0, filename: str = "log.txt", mode: str = "a"):
            logger = tiny_logger_setup(save_dir, distributed_rank, filename, mode)
            logger.info(f"==> log file: {os.path.join(save_dir, filename)}")
            return logger

    logger_cfg: LoggerCfg = field(default_factory=LoggerCfg)

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
                        if os.getenv("RANK", 0) == 0 or os.getenv("RANK", 0) == "0":
                            print(f"{key}: {value} <-- {ori_value}(original)")
                            # print(f"Override {key} from {ori_value} to {value} in {cfg_object.__class__.__name__}")
                        setattr(cfg_object, key, value)
                        self.overrided_cfg[key] = value
            else:
                raise UnknownConfigurationKeyError(key)
        return cfg_object


@dataclass
class RedisCfgMixin:
    @dataclass
    class RedisCacheCfg:
        redis_cache_enabled: bool = True
        redis_cache_shard_ports: ListConfig = field(
            default_factory=lambda: ListConfig(
                [
                    7000,
                    7001,
                    7002,
                    7003,
                    7004,
                ]
            )
        )  # List of Redis cache shard used ports
        redis_cache_max_memory: int = 160  # Maximum memory is 160GB, according to the ImageNet dataset size
        redis_cluster_manager_cpus: int = 10

        def build_redis_cache(self):
            if self.redis_cache_enabled:
                from tinyexp.utils.redis_utils import RedisClusterManager

                redis_cluster_manager = RedisClusterManager(
                    ports=self.redis_cache_shard_ports,
                    max_memory_per_port=self.redis_cache_max_memory // len(self.redis_cache_shard_ports),
                )
                return redis_cluster_manager.start_redis_cluster()
            return True

    redis_cache_cfg: RedisCacheCfg = field(default_factory=RedisCacheCfg)

    def proxy_build_redis_cache(self):
        """
        Hard-coded method to build Redis cache since ray actor need
        """
        return self.redis_cache_cfg.build_redis_cache()


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
