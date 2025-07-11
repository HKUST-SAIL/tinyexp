"""
author: LI Zeming
email: zengarden2009@gmail.com
"""

import os
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import ray
from hydra.conf import HydraConf, RunDir
from hydra.core.config_store import ConfigStore

from .tiny_engine.accelerator.base_accelerator import BaseAccelerator
from .utils.log_utils import tiny_logger_setup
from .utils.ray_utils import simple_ray_launch_exp

__all__ = ["TinyExp", "TinyCfg", "simple_ray_launch_exp", "ConfigStore"]


@dataclass
class _HydraConfig(HydraConf):
    """
    To avoid hydra output the config in unexpected directory.
    """

    output_subdir: Optional[str] = None
    run: RunDir = field(default_factory=lambda: RunDir("./output"))


@dataclass
class TinyCfg:
    hydra: _HydraConfig = field(default_factory=_HydraConfig)


class TinyExp:
    """
    TinyExp serves as a lightweight framework for running experiments, with support
    for both PyTorch and Ray environments. It handles common experiment setup tasks
    such as configuring accelerators for distributed training, establishing consistent
    experiment naming, managing output directories, and setting up logging.

    Parameters
    ----------
    cfg : object
        Configuration object containing experiment settings. Should include attributes
        that control experiment behavior. May include an optional 'output_root' attribute
        to specify the base output directory.

    Attributes
    ----------
        cfg : object
            The configuration object provided during initialization.
        accelerator : object or None
            The accelerator object for distributed training (if applicable).
        exp_name : str
            The name of the experiment, derived from the class name.
        output_dir : str
            The directory where experiment outputs will be stored.
        logger : Logger
            The logger object configured for this experiment.

    Methods
    -------
        _configure_exp_name()
            Generates the experiment name based on the class name.
        _configure_output_dir()
            Constructs the output directory path based on output_root and exp_name.
        _configure_accelerator()
            Sets up the appropriate accelerator for the experiment.
        _configure_logger()
            Initializes and configures the logging system.

    Notes
    -----
    This class is designed to be subclassed for specific experiment implementations.
    Override the configuration methods to customize experiment behavior.
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.accelerator = self._configure_accelerator()
        self.exp_name = self._configure_exp_name()
        self.output_dir = self._configure_output_dir()
        self.logger = self._configure_logger()
        self.module = self._configure_module()
        self.optimizer = self._configure_optimizer()
        self.lr_scheduler = self._configure_lr_scheduler()
        self.global_step = 0
        self.global_epoch = 0

    def _configure_exp_name(self) -> str:
        return self.__class__.__name__

    def _configure_output_dir(self) -> str:
        output_root = getattr(self.cfg, "output_root", "./output")
        return os.path.join(output_root, self.exp_name)

    @abstractmethod
    def _configure_accelerator(self) -> BaseAccelerator:
        pass

    @abstractmethod
    def _configure_module(self):
        pass

    @abstractmethod
    def _configure_optimizer(self):
        pass

    @abstractmethod
    def _configure_lr_scheduler(self):
        pass

    def _configure_logger(self):
        distributed_rank = self.accelerator.rank if self.accelerator else 0
        logger = tiny_logger_setup(self.output_dir, distributed_rank=distributed_rank, filename="log.log")
        logger.info("{}{}".format("log file: ", os.path.join(self.output_dir, "log.log")))
        # logger.info("{}{}".format("Command line: ", " ".join(sys.argv)))
        return logger

    @staticmethod
    def after_ray_init_callback(cfg):
        # ------------------- build redis server -------------------- #
        actor_list = []
        if hasattr(cfg, "redis_cache_enabled") and cfg.redis_cache_enabled:
            from tinyexp.utils.redis_utils import RedisClusterManager

            requested_cpu = cfg.num_gpus * (cfg.train_data_worker_per_gpu + cfg.val_data_worker_per_gpu + 1)
            if requested_cpu + cfg.redis_cluster_manager_cpus > os.cpu_count():
                raise RuntimeError(
                    f"Total CPU count {os.cpu_count()} is not enough for the experiment, "
                    f"please set `num_gpus * (train_data_worker_per_gpu + val_data_worker_per_gpu + 1)"
                    f"+ redis_cluster_manager_cpus` <= {os.cpu_count()}"
                )
            RemoteRedisClusterManager = ray.remote(num_cpus=cfg.redis_cluster_manager_cpus)(RedisClusterManager)
            redis_actor = RemoteRedisClusterManager.remote(
                ports=cfg.redis_cache_shard_ports,
                max_memory_per_port=cfg.redis_cache_max_memory // len(cfg.redis_cache_shard_ports) + 1,
            )
            success = ray.get(redis_actor.start_redis_cluster.remote())
            if not success:
                raise RuntimeError("Failed to start Redis cluster")
                # Periodically monitor Redis memory usage
            else:
                print(f"Redis cluster started successfully with ports: {cfg.redis_cache_shard_ports}")
            actor_list.append(redis_actor)

            # import threading
            # import time

            # def monitor_redis_memory():
            #     while True:
            #         try:
            #             memory_info = ray.get(redis_actor.get_redis_memory_info.remote())
            #             print(f" ==> Redis Memory Status: {memory_info}")
            #             time.sleep(60)
            #         except:
            #             break

            # monitor_thread = threading.Thread(target=monitor_redis_memory, daemon=True)
            # monitor_thread.start()
        return actor_list
