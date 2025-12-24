from collections.abc import Sequence
from typing import Optional


class UnknownConfigurationKeyError(AttributeError):
    def __init__(self, key: str) -> None:
        self.key = key
        super().__init__(f"Configuration key {key!r} does not exist in the provided object.")


class UnknownAcceleratorTypeError(ValueError):
    def __init__(self, accelerator: str) -> None:
        self.accelerator = accelerator
        super().__init__(f"Unknown accelerator type: {accelerator}")


class InvalidWorkerCountError(ValueError):
    def __init__(self, num_worker: int) -> None:
        self.num_worker = num_worker
        super().__init__(f"Number of workers must be greater than 0, got {num_worker}.")


class UnknownExperimentModeError(ValueError):
    def __init__(self, mode: str, allowed: Sequence[str] = ("train", "val", "help")) -> None:
        self.mode = mode
        self.allowed = tuple(allowed)
        super().__init__(f"Unknown mode {mode!r}, please set `mode` to one of: {', '.join(self.allowed)}.")


class InsufficientCPUError(RuntimeError):
    def __init__(self, *, total_cpu: Optional[int], needed_cpu: int) -> None:
        self.total_cpu = total_cpu
        self.needed_cpu = needed_cpu
        super().__init__(f"Total CPU count {total_cpu} is not enough for the experiment; needed at least {needed_cpu}.")


class UnknownLauncherError(ValueError):
    def __init__(self, launcher: str, allowed: Sequence[str] = ("python", "torchrun", "accelerate")) -> None:
        self.launcher = launcher
        self.allowed = tuple(allowed)
        super().__init__(f"Unknown launcher {launcher!r}, please use one of: {', '.join(self.allowed)}.")


class CudaNotAvailableError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("CUDA is required but not available.")
