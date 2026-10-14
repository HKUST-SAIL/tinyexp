"""TinyExp: the configured experiment class is the entrypoint.

Minimal experiment (see ``tinyexp/examples/mnist_exp.py`` for a full DDP/CPU recipe)::

    from dataclasses import dataclass, field
    from tinyexp import LoggerCfgMixin, TinyExp, store_and_run_exp

    @dataclass(repr=False)
    class Exp(TinyExp, LoggerCfgMixin):
        @dataclass
        class TrainCfg:
            lr: float = 0.1

        train_cfg: TrainCfg = field(default_factory=TrainCfg)

        def run(self) -> None:  # every mode except ``help`` lands here; dispatch on self.mode
            logger = self.logger_cfg.build_logger(save_dir=self.get_run_dir())
            self.print_cfg(logger)

    if __name__ == "__main__":
        store_and_run_exp(Exp)

    # python my_exp.py mode=help train_cfg.lr=0.01   -> print resolved config, run nothing

``TinyExp`` fields (all overridable as ``key=value`` on the CLI):

| field | default | meaning |
| --- | --- | --- |
| `exp_class` | `""` | import path of the class, filled in by `store_and_run_exp` (not listed as an override); `__main__.Exp` when started as `python file.py` |
| `exp_name` | main file stem | run subdirectory name (`my_exp.py` -> `my_exp`) |
| `launcher` | `"mp"` | `mp`: run `Exp.run()` in this process; `ray`: spawn Ray workers |
| `output_root` | `"./output"` | root of all outputs; run dir `get_run_dir()` = output_root/exp_name; Hydra's own log is `<output_root>/tinyexp.log` |
| `mode` | `"train"` | `help` is handled by TinyExp (print config, exit; it still creates `output_root`); any other value reaches `run()`, whose meaning the experiment defines |
| `resume_from` | `""` | checkpoint path to resume or evaluate from; empty means none |
| `hydra` | internal | Hydra settings; do not override |
| `overrided_cfg` | internal | filled by `set_cfg`; read it, do not set it |

Pairing ``launcher`` with the process owner:

- ``python exp.py launcher=mp``: one process.
- ``python exp.py launcher=ray``: the driver creates ``ray_cfg.ray_num_worker`` workers (``RayCfgMixin``).
- ``torchrun ... exp.py launcher=mp`` / ``accelerate launch ... launcher=mp``: the external tool owns processes.
- ``tinyexp-run-with-ray-cluster ... -- python exp.py launcher=ray``: multi-node static Ray cluster.

CLI overrides are checked by Hydra against the dataclasses: unknown keys are rejected
(``Key 'x' not in 'DataloaderCfg'``) and values are type-checked, but not otherwise validated (``mode=help``
accepts ``accelerator_cfg.accelerator=bogus``). For a ``dict`` field, change an existing key with ``tags.owner=x``
and add a new one with ``+tags.team=y`` (without the ``+``: ``Key 'team' is not in struct``). How the result is
merged onto the experiment and recorded for ``mode=help``: ``TinyExp.set_cfg``. Ray workers and ``python -m`` vs
``python file.py``: ``RayCfgMixin``.

Exports: ``TinyExp``, ``store_and_run_exp``, ``simple_launch_exp``, ``ConfigStore`` and the mixins
``LoggerCfgMixin``, ``RayCfgMixin``, ``CheckpointCfgMixin``, ``WandbCfgMixin``, ``RedisCfgMixin``
(see ``tinyexp.exp_mixins``).
"""

__author__ = "LI Zeming"
__email__ = "zane.li@connect.ust.hk"
__license__ = "MIT"

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Optional

import hydra
from hydra.conf import HydraConf, RunDir
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from .exceptions import UnknownConfigurationKeyError, UnknownLauncherError
from .exp_mixins import (
    CheckpointCfgMixin,
    LoggerCfgMixin,
    RayCfgMixin,
    RedisCfgMixin,
    WandbCfgMixin,
)

__all__ = [
    "CheckpointCfgMixin",
    "ConfigStore",
    "LoggerCfgMixin",
    "RayCfgMixin",
    "RedisCfgMixin",
    "TinyExp",
    "WandbCfgMixin",
    "simple_launch_exp",
    "store_and_run_exp",
]


def _patch_argparse_help_for_python314() -> None:
    """Allow Hydra's lazy help object to work with Python 3.14 argparse."""
    if sys.version_info < (3, 14):
        return

    import argparse

    original_get_help_string = argparse.HelpFormatter._get_help_string
    if getattr(original_get_help_string, "_tinyexp_python314_compat", False):
        return

    def get_help_string(self, action):
        help_string = original_get_help_string(self, action)
        return help_string if isinstance(help_string, str) else str(help_string)

    get_help_string._tinyexp_python314_compat = True
    argparse.HelpFormatter._get_help_string = get_help_string


_patch_argparse_help_for_python314()


@dataclass
class _HydraConfig(HydraConf):
    """
    Keep Hydra's run directory (and its ``tinyexp.log``) under ``output_root`` instead of a fixed ``./output``,
    and do not let Hydra write its own config copies.
    """

    output_subdir: Optional[str] = None
    run: RunDir = field(default_factory=lambda: RunDir("${output_root}"))


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

    # How to launch the experiment workers:
    # - "ray": the driver spawns Ray workers (run with plain `python`).
    # - "mp": run in the current process; process management is owned by an
    #   external multi-process launcher such as torchrun or accelerate.
    launcher: str = "mp"

    # log directory
    output_root: str = "./output"

    # The experiment mode:
    # - "run": run a general experiment workload.
    # - "train": train the model.
    # - "eval": evaluate the model.
    # - "help": print the experiment configuration and exit.
    mode: str = "train"

    # ckpt path to resume from, if empty, will not resume
    resume_from: str = ""

    # overridden configurations, only for internal use
    overrided_cfg: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __repr__(self):
        # Customize the representation of the Exp object for cleaner Ray logs.
        return f"Exp(rank={os.getenv('RANK', 'N/A')})"

    def get_run_dir(self) -> str:
        return os.path.join(self.output_root, self.exp_name)

    def set_cfg(self, cfg_hydra, cfg_object=None):
        """Copy the Hydra config (dataclass defaults plus CLI overrides) onto this instance; returns ``self``.

        Called by the launcher in every process (with ``launcher=ray``, by each worker on its own copy):

        - nested ``XXXCfg`` dataclasses are updated field by field, recursively;
        - fields whose current value is a mapping (``dict``) are replaced as a whole;
        - any other value is assigned directly; a key the object does not have raises
          ``UnknownConfigurationKeyError``.

        Every changed value is recorded in ``overrided_cfg[<leaf key name>] = {"value", "original"}`` (keyed by the
        leaf name only, not the dotted path) and listed under "Overridden" by ``print_cfg`` / ``mode=help``.
        """
        if cfg_object is None:
            cfg_object = self
        for key, value in cfg_hydra.items():
            if cfg_object is self and key == "overrided_cfg":
                # Internal bookkeeping: a full config carries the default ``{}``, which must not
                # overwrite the records collected for fields processed before it.
                continue
            if hasattr(cfg_object, key):
                sub_object = getattr(cfg_object, key)
                if isinstance(value, (DictConfig, Mapping)):
                    if isinstance(sub_object, Mapping):
                        # Hydra has already merged and validated mapping values; store the resolved plain dict.
                        new_value = (
                            dict(OmegaConf.to_container(value, resolve=True))
                            if isinstance(value, DictConfig)
                            else dict(value)
                        )
                        if sub_object != new_value:
                            self.overrided_cfg[key] = {
                                "value": new_value,
                                "original": sub_object,
                            }
                            setattr(cfg_object, key, new_value)
                    else:
                        # Nested config objects still receive recursive attribute updates.
                        self.set_cfg(value, sub_object)
                else:
                    # Otherwise, set the attribute directly
                    ori_value = getattr(cfg_object, key, None)
                    if ori_value != value:
                        if not (cfg_object is self and key == "exp_class"):
                            # exp_class is always filled in by store_and_run_exp; reporting it is noise.
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


@hydra.main(version_base=None, config_path="pkg://tinyexp", config_name="cfg")
def simple_launch_exp(cfg: DictConfig) -> None:
    """
    This is a template for launching a experiment with hydra config.
    The launcher is selected by ``cfg.launcher``: "ray" spawns Ray workers from the
    driver process, while "mp" runs the experiment in the current process (for use
    with external multi-process launchers such as torchrun or accelerate).
    """
    exp_class = hydra.utils.get_class(cfg.exp_class)

    if cfg.mode == "help":
        exp_instance = exp_class()
        exp_instance.set_cfg(cfg)

        class _StdoutLogger:
            def info(self, message):  # type: ignore[no-untyped-def]
                print(message, flush=True)

        exp_instance.print_cfg(_StdoutLogger())
        print("\n", flush=True)
        return

    if cfg.launcher == "ray":
        ray_cfg_class = type(exp_class().ray_cfg)
        ray_cfg_class.run(exp_class, cfg)
    elif cfg.launcher == "mp":
        exp_class().set_cfg(cfg).run()
    else:
        raise UnknownLauncherError(cfg.launcher)


def _resolve_canonical_exp_class(exp_class: type[TinyExp]) -> Optional[type[TinyExp]]:
    """Return the importable twin of a ``__main__``-defined experiment class, if any.

    ``python -m pkg.mod`` executes the module as ``__main__``; the executed class
    object would make ray/cloudpickle serialize the whole class graph *by value*
    (via the actor export and the structured-config metadata shipped with
    ``set_cfg``), which is fragile: any builtin the class graph touches (e.g. a
    platform-agent-patched ``builtins.print``) can break the by-reference
    identity check ("it's not the same object as builtins.print"). When the same
    class is importable under the running module's spec name (``pkg.mod.Exp``),
    prefer that twin so everything ships by reference.
    """
    if exp_class.__module__ != "__main__":
        return None
    main_spec = getattr(sys.modules.get("__main__"), "__spec__", None)
    spec_name = getattr(main_spec, "name", None)
    if not spec_name or spec_name == "__main__":
        return None
    import importlib

    try:
        canonical = getattr(importlib.import_module(spec_name), exp_class.__qualname__)
    except (ImportError, AttributeError):
        return None
    if isinstance(canonical, type) and issubclass(canonical, TinyExp):
        return canonical
    return None


def store_and_run_exp(exp_class: type[TinyExp]) -> None:
    """
    Extract the config from the exp_class and store it in the ConfigStore(hydra config store).
    Then launch the experiment with the config.

    Args:
        exp_class: The class of the experiment to run.

    Returns:
        None: This function does not return anything.
    """

    # this is the hack for hydra to find the experiment class; prefer the
    # canonical importable twin of a __main__ class so ray workers ship it by
    # reference (see _resolve_canonical_exp_class)
    canonical_class = _resolve_canonical_exp_class(exp_class) or exp_class
    exp_class_path = f"{canonical_class.__module__}.{canonical_class.__qualname__}"
    exp_cfg = canonical_class()
    exp_cfg.exp_class = exp_class_path

    # store the experiment configuration in the ConfigStore and launch the experiment
    ConfigStore.instance().store(name="cfg", node=exp_cfg)
    simple_launch_exp()
