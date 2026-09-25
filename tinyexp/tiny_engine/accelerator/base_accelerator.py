from __future__ import annotations

import abc
import contextlib
import functools
import os
from abc import abstractmethod
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import torch

__all__ = ["AcceleratorProtocol", "BaseAccelerator"]


def _convert_to_fp32(data):  # type: ignore[no-untyped-def]
    """Recursively cast the fp16/bf16 tensors in ``data`` back to fp32.

    Mirrors accelerate's ``convert_to_fp32``, so a prepared model hands the
    training loop the same dtypes whichever accelerator produced it: autocast
    lowers precision inside compute kernels, but what comes back out is fp32
    and losses/metrics therefore accumulate identically.
    """
    if isinstance(data, (tuple, list)):
        converted = (_convert_to_fp32(item) for item in data)
        # A namedtuple takes its fields positionally, not as a single iterable.
        return type(data)(*converted) if hasattr(data, "_fields") else type(data)(converted)
    if isinstance(data, Mapping):
        return type(data)({key: _convert_to_fp32(value) for key, value in data.items()})
    if getattr(data, "dtype", None) in (torch.float16, torch.bfloat16):
        return data.float()
    return data


@runtime_checkable
class AcceleratorProtocol(Protocol):
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    sync_gradients: bool

    @property
    def is_main_process(self) -> bool: ...

    @property
    def is_local_main_process(self) -> bool: ...

    def unwrap_model(self, model: Any) -> Any: ...

    def prepare(self, model: Any, optimizer: Any = None) -> Any: ...

    def prepare_model(self, model: Any) -> Any: ...

    def prepare_optimizer(self, optimizer: Any) -> Any: ...

    def backward(self, loss: torch.Tensor) -> None: ...

    def autocast(self) -> contextlib.AbstractContextManager[None]: ...

    def optimizer_step(self, optimizer: Any) -> Any: ...

    def wait_for_everyone(self) -> None: ...

    def reduce(self, tensor: torch.Tensor, reduction: str = "sum", scale: float = 1.0) -> torch.Tensor: ...

    def print(self, *args: Any, **kwargs: Any) -> None: ...

    def destroy(self) -> None: ...


class BaseAccelerator(abc.ABC):
    """
    basic accelerator, provide basic functions for distributed training.
    """

    def __init__(self) -> None:
        self.rank = int(os.getenv("RANK", 0))
        self.world_size = int(os.getenv("WORLD_SIZE", 1))
        self.local_rank = int(os.getenv("LOCAL_RANK", 0))
        self.sync_gradients = True
        self._destroyed = False
        self._process_group_initialized = False
        # Set by accelerators that support mixed precision; None means fp32.
        self._amp_dtype: torch.dtype | None = None
        self.master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
        self.master_port = int(os.getenv("MASTER_PORT", 12345))

        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:  # in ray env, device count is always 1
                self.device = torch.device("cuda", self.local_rank)
            else:
                self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    @abstractmethod
    def _init_process_group(self) -> None:
        pass

    @abstractmethod
    def unwrap_model(self, model):  # type: ignore[no-untyped-def]
        pass

    @abstractmethod
    def prepare(self, model, optimizer=None):  # type: ignore[no-untyped-def]
        pass

    @abstractmethod
    def prepare_model(self, model):  # type: ignore[no-untyped-def]
        pass

    @abstractmethod
    def prepare_optimizer(self, optimizer):  # type: ignore[no-untyped-def]
        pass

    @abstractmethod
    def backward(self, loss: torch.Tensor) -> None:
        pass

    def autocast(self) -> contextlib.AbstractContextManager[None]:
        """Forward-pass context manager; mixed-precision accelerators override it."""
        return contextlib.nullcontext()

    def _wrap_forward_autocast(self, module):  # type: ignore[no-untyped-def]
        """Run ``module.forward`` inside this accelerator's autocast context.

        Mirrors what accelerate does in its own ``prepare_model``: a training
        loop that never opens ``autocast()`` itself still gets mixed precision,
        so swapping ``HFAccelerator`` for a tiny_engine accelerator cannot
        silently fall back to fp32. Loops that do open ``accelerator.autocast()``
        stay correct -- nesting the same dtype is a no-op.

        Outputs are cast back to fp32 after the context exits, matching
        accelerate's ``convert_outputs_to_fp32`` ordering, so the two adapters
        return identical dtypes.

        Call this before any parallel wrapper (DDP/FSDP), which invokes the
        wrapped module's ``forward`` and therefore picks the patch up.
        """
        if self._amp_dtype is None or hasattr(module, "_original_forward"):
            return module

        module._original_forward = module.forward

        @functools.wraps(module._original_forward)
        def forward(*args, **kwargs):  # type: ignore[no-untyped-def]
            with self.autocast():
                output = module._original_forward(*args, **kwargs)
            return _convert_to_fp32(output)

        module.forward = forward
        return module

    def optimizer_step(self, optimizer: Any) -> Any:
        """Step the optimizer; mixed-precision accelerators override it for loss scaling."""
        return optimizer.step()

    @abstractmethod
    def wait_for_everyone(self) -> None:
        pass

    @abstractmethod
    def reduce(self, tensor: torch.Tensor, reduction: str = "sum", scale: float = 1.0) -> torch.Tensor:
        """Reduce a tensor across processes; mirrors accelerate's Accelerator.reduce."""
        pass

    @abstractmethod
    def print(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        pass

    @abstractmethod
    def destroy(self) -> None:
        pass

    def __del__(self) -> None:
        """Best-effort cleanup when explicit cleanup was skipped."""
        with contextlib.suppress(Exception):
            self.destroy()

    @property
    def is_main_process(self):  # type: ignore[no-untyped-def]
        """True for one process per server."""
        return self.rank == 0

    @property
    def is_local_main_process(self):  # type: ignore[no-untyped-def]
        """True for one process per server."""
        return self.local_rank == 0

    @property
    def is_last_process(self):  # type: ignore[no-untyped-def]
        return self.rank == self.world_size - 1
