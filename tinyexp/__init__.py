__author__ = "LI Zeming"
__email__ = "zane.li@connect.ust.hk"
__license__ = "MIT"

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

from hydra.conf import HydraConf, RunDir
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from .exceptions import UnknownConfigurationKeyError
from .exp_mixins import CheckpointCfgMixin, LoggerMixin, RayCfgMixin, RedisCfgMixin, WandbCfgMixin
from .utils.ray_utils import simple_launch_exp

__all__ = [
    "CheckpointCfgMixin",
    "ConfigStore",
    "LoggerMixin",
    "RayCfgMixin",
    "RedisCfgMixin",
    "TinyExp",
    "WandbCfgMixin",
    "simple_launch_exp",
    "store_and_run_exp",
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
class TinyExp:
    """
    Simple experiment configuration class, which use hydra to manage and override configurations.
    The core idea is to provide a unified interface for experiment configurations, which can be instantiated
    and used in various contexts, such as Ray or TorchRun.
    """

    hydra: _HydraConfig = field(default_factory=_HydraConfig)

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
    resume_from: str = ""  # ckpt path to resume from, if empty, will not resume

    # overridden configurations, only for internal use
    overrided_cfg: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __repr__(self):
        # Customize the representation of the Exp object for cleaner Ray logs.
        return f"Exp(rank={os.getenv('RANK', 'N/A')})"

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
