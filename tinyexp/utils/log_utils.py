"""``tiny_logger_setup(save_dir, distributed_rank=0, filename="log.txt", mode="a")`` -> loguru logger.

Rank 0 logs INFO to ``<save_dir>/<filename>`` and stderr; other ranks log WARNING and above only.
``mode="o"`` overwrites the file. Prefer ``LoggerCfgMixin.LoggerCfg.build_logger``, which also creates ``save_dir``.
"""

import os
import sys
from sys import stderr

from loguru import logger


def tiny_logger_setup(save_dir: str, distributed_rank: int = 0, filename: str = "log.txt", mode: str = "a"):  # type: ignore[no-untyped-def]
    """setup logger for training and testing.
    Args:
        save_dir(str): loaction to save log file
        distributed_rank(int): device rank when multi-gpu environment
        mode(str): log file write mode: `a` appends (default), `o` overwrites an existing file.
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
        # Ray workers already isolate their loggers; avoid Loguru's
        # multiprocessing queue, whose Python 3.14 resource tracker can race
        # during Ray actor shutdown.
        enqueue=sys.version_info < (3, 14),
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
