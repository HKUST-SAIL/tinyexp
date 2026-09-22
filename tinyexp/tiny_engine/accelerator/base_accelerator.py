from __future__ import annotations

import abc
import contextlib
import os
from abc import abstractmethod
from typing import Any, Protocol, runtime_checkable

import torch

__all__ = ["AcceleratorProtocol", "BaseAccelerator"]


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
