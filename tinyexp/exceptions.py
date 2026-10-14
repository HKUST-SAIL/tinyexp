"""Errors raised by TinyExp; each keeps the offending value as an attribute.

- ``UnknownConfigurationKeyError`` (``.key``): ``set_cfg`` got a key the experiment does not define.
- ``UnknownLauncherError`` (``.launcher``): ``launcher`` is not ``mp`` or ``ray``.
- ``UnknownExperimentModeError(mode, allowed=(...))`` (``.mode``): raise from ``run()`` for an unsupported
  ``mode``; the message lists ``allowed``, e.g. ``raise UnknownExperimentModeError(self.mode, allowed=("train", "eval", "help"))``.
- ``UnknownAcceleratorTypeError(accelerator)`` (``.accelerator``): raise from ``build_accelerator()`` for an unknown name.
- ``InvalidWorkerCountError`` (``.num_worker``): ``ray_num_worker`` is 0, below -1, or ``-1`` found no resources.
- ``InsufficientCPUError``: the Ray cluster lacks the requested CPU/GPU resources.
- ``CudaNotAvailableError``: a CUDA-only accelerator was built without CUDA.
- ``UnsupportedCheckpointFormatError`` (``.path``): ``load_checkpoint`` got a dict that is not a TinyExp checkpoint.

Import them from ``tinyexp.exceptions``.
"""

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
    def __init__(self, mode: str, allowed: Sequence[str] = ("run", "train", "eval", "help")) -> None:
        self.mode = mode
        self.allowed = tuple(allowed)
        super().__init__(f"Unknown mode {mode!r}, please set `mode` to one of: {', '.join(self.allowed)}.")


class InsufficientCPUError(RuntimeError):
    def __init__(
        self,
        *,
        total_cpu: Optional[int],
        needed_cpu: int,
        total_gpu: Optional[float] = None,
        needed_gpu: Optional[float] = None,
    ) -> None:
        self.total_cpu = total_cpu
        self.needed_cpu = needed_cpu
        self.total_gpu = total_gpu
        self.needed_gpu = needed_gpu
        message = f"Ray resources are insufficient; needed CPU={needed_cpu}, available CPU={total_cpu}"
        if needed_gpu is not None or total_gpu is not None:
            message += f", needed GPU={needed_gpu}, available GPU={total_gpu}"
        super().__init__(message)


class UnknownLauncherError(ValueError):
    def __init__(self, launcher: str, allowed: Sequence[str] = ("ray", "mp")) -> None:
        self.launcher = launcher
        self.allowed = tuple(allowed)
        super().__init__(f"Unknown launcher {launcher!r}, please use one of: {', '.join(self.allowed)}.")


class CudaNotAvailableError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("CUDA is required but not available.")


class UnsupportedCheckpointFormatError(ValueError):
    def __init__(self, path: str) -> None:
        self.path = path
        super().__init__(
            f"Checkpoint at {path} is not a supported tinyexp checkpoint format and does not contain model_state_dict."
        )
