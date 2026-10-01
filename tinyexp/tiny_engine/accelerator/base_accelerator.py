from __future__ import annotations

import abc
import contextlib
import functools
import os
from abc import abstractmethod
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import torch
import torch.distributed as dist

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
        # fp16 accelerators install a GradScaler; None means no loss scaling,
        # and the shared unscale/clip helpers below no-op.
        self._scaler: torch.amp.GradScaler | None = None
        # clip_grad_norm_ must unscale before measuring the norm, and unscaling
        # is per-optimizer, so remember the optimizers that went through prepare.
        # Their parameter ids are cached alongside: optimizer->parameter ownership
        # is static after setup, so the per-step clip check is one set difference.
        self._prepared_optimizers: list[torch.optim.Optimizer] = []
        self._prepared_param_ids: set[int] = set()
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

    def _register_optimizer(self, optimizer: Any) -> Any:
        if optimizer not in self._prepared_optimizers:
            self._prepared_optimizers.append(optimizer)
            self._prepared_param_ids |= self._owned_param_ids(optimizer)
        return optimizer

    @staticmethod
    def _owned_param_ids(optimizer: Any) -> set[int]:
        # Identity, not __eq__: comparing tensors with == returns a tensor.
        return {id(p) for group in optimizer.param_groups for p in group["params"]}

    def unscale_gradients(self, optimizer: Any = None) -> None:
        """Undo the fp16 loss scaling on ``.grad``; a no-op without an enabled scaler."""
        if self._scaler is None or not self._scaler.is_enabled():
            return
        optimizers = self._prepared_optimizers if optimizer is None else [optimizer]
        for opt in optimizers:
            self._scaler.unscale_(opt)

    def _unscale_for_clip(self, parameters: Any) -> Any:
        """Undo the fp16 loss scaling on gradients about to be clipped.

        Unscaling requires the owning optimizer, so gradients of optimizers that
        skipped ``prepare_optimizer`` raise instead of being clipped at their
        ~65536x scale -- a warning would let every following step corrupt the
        update unnoticed.
        """
        if self._scaler is None or not self._scaler.is_enabled():
            return parameters
        parameters = list(parameters)
        clipped = {id(p) for p in parameters if p.grad is not None}
        if clipped - self._prepared_param_ids:
            raise RuntimeError(  # noqa: TRY003
                "clip_grad_norm_ received gradients whose optimizer never went through "
                "prepare_optimizer. Under fp16 such gradients still carry the loss scale, "
                "so clipping them directly would be wrong. Prepare the optimizer first."
            )
        self.unscale_gradients()
        return parameters

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gather ``tensor`` from every rank, concatenated on dim 0, on every rank.

        Mirrors accelerate's gather contract, so a caller like
        ``accelerator.gather(metric).mean()`` computes the same global value on
        all ranks instead of silently reading a rank-local tensor. All ranks
        must pass tensors of equal shape.
        """
        if self.world_size < 2:
            return tensor
        # NCCL only gathers device-resident tensors; stage over and back so
        # cpu-side tensors work too, the same accommodation reduce makes.
        device_tensor = tensor.to(self.device)
        gather_list = [torch.empty_like(device_tensor) for _ in range(self.world_size)]
        dist.all_gather(gather_list, device_tensor)
        return torch.cat(gather_list, dim=0).to(tensor.device)

    @abstractmethod
    def wait_for_everyone(self) -> None:
        pass

    @abstractmethod
    def reduce(self, tensor: torch.Tensor, reduction: str = "sum", scale: float = 1.0) -> torch.Tensor:
        """Reduce a tensor across processes; mirrors accelerate's Accelerator.reduce.

        ``scale`` is accepted for accelerate API compatibility and, exactly as
        in accelerate's non-XLA backends, ignored.
        """
        pass

    def print(self, *args: Any, **kwargs: Any) -> None:
        """Print on the main process only, mirroring accelerate's Accelerator.print."""
        if self.is_main_process:
            print(*args, **kwargs)

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
