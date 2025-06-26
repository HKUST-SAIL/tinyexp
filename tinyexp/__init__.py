import os
import sys
from functools import cached_property
from sys import stderr

from loguru import logger


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
    A tiny experiment class that provides basic functionalities for experiments.
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    @cached_property
    def exp_name(self) -> str:
        exp_class_name = self.__class__.__name__
        # if hasattr(self, "_override_cfg"):
        #     for k, v in self._override_cfg.items():
        #         if isinstance(v, str):
        #             exp_class_name += "_{}{}".format(k, v.replace("/", "-"))
        #         else:
        #             exp_class_name += f"_{k}{v}"
        return exp_class_name

    @cached_property
    def output_dir(self) -> str:
        output_root = getattr(self.cfg, "output_root", "./output")
        return os.path.join(output_root, self.exp_name)

    @cached_property
    def accelerator(self):
        return None

    @cached_property
    def logger(self):
        distributed_rank = self.accelerator.rank if self.accelerator else 0
        logger = tiny_logger_setup(self.output_dir, distributed_rank=distributed_rank, filename="log.log")
        logger.info("{}{}".format("Command line: ", " ".join(sys.argv)))
        return logger
