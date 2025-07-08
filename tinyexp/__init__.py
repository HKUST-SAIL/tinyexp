import os
from abc import abstractmethod
from sys import stderr

from loguru import logger

from .tiny_engine.accelerator.base_accelerator import BaseAccelerator

__all__ = ["TinyExp", "tiny_logger_setup"]


def tiny_logger_setup(save_dir: str, distributed_rank: int = 0, filename: str = "log.txt", mode: str = "a"):  # type: ignore[no-untyped-def]
    """setup logger for training and testing.
    Args:
        save_dir(str): loaction to save log file
        distributed_rank(int): device rank when multi-gpu environment
        mode(str): log file write mode, `append` or `override`. default is `a`.
    Return:
        logger instance.
    """
    save_file = os.path.join(save_dir, filename)
    if mode == "o" and os.path.exists(save_file):
        os.remove(save_file)

    # Remove all existing processors
    logger.remove()

    # Detailed format for file logging
    file_format = "{time:HH:mm:ss} {message}"

    # Simplified format for console output
    console_format = "<green>{time:HH:mm:ss}</green> {message}"

    # Add file logging processor
    _ = logger.add(
        save_file,
        format=file_format,
        filter="",
        level="INFO" if distributed_rank == 0 else "WARNING",
        enqueue=True,
        colorize=False,
    )

    # Add console logging processor
    _ = logger.add(
        stderr,
        format=console_format,
        filter="",
        level="INFO" if distributed_rank == 0 else "WARNING",
        colorize=True,
    )

    return logger


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
