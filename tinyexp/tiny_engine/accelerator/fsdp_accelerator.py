"""Fully sharded data parallel accelerator.

The accelerator uses PyTorch's FSDP ``FULL_SHARD`` strategy.  FSDP is kept as
an outer module wrapper so that its forward hooks can all-gather parameters
before a module runs and release them afterwards.  ``use_orig_params=True``
keeps the optimizer interface compatible with ordinary PyTorch optimizers.
"""

from __future__ import annotations

import types
from functools import partial
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

from ...exceptions import CudaNotAvailableError
from .base_accelerator import BaseAccelerator
from .ddp_accelerator import _MIXED_PRECISION_DTYPES

# Full gather materialized on rank 0 only, torch's recommendation for
# cpu_offload; every rank must still join these collectives, and the other
# ranks simply receive empty dicts.
_FULL_STATE_DICT = StateDictOptions(full_state_dict=True, cpu_offload=True)


class FSDPAccelerator(BaseAccelerator):
    """Accelerator for parameter, gradient, and optimizer-state sharding.

    Mixed precision uses ``torch.autocast`` over full-precision parameters,
    the same recipe as DDP: FSDP's native ``MixedPrecision`` policy was
    measured ~2.2x slower per epoch on 8xH200 (its per-block parameter
    cast machinery dominates), while autocast only recasts inside compute
    kernels and keeps parameters, communication, and checkpoints fp32.
    """

    # ResNet's generic training loop builds the optimizer before calling
    # ``prepare``. FSDP replaces the module parameters, so that order must be
    # reversed for this accelerator.
    requires_optimizer_after_model = True

    def __init__(self, mixed_precision: str = "none") -> None:
        super().__init__()
        if not torch.cuda.is_available():
            raise CudaNotAvailableError()
        if mixed_precision != "none" and mixed_precision not in _MIXED_PRECISION_DTYPES:
            raise ValueError(f"Unknown mixed precision {mixed_precision!r}; expected none/fp16/bf16")  # noqa: TRY003

        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(self.device)
        if self.world_size > 1 and not dist.is_initialized():
            self._init_process_group()
            self._process_group_initialized = True
        self.sync_gradients = True

        # fp16 gradients need loss scaling to stay in range; bf16 does not.
        # A disabled GradScaler passes scale/step through unchanged, so one
        # code path covers every mode.
        self._amp_dtype = _MIXED_PRECISION_DTYPES.get(mixed_precision)
        self._scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision == "fp16")
        # The wrapped root module, for clip_grad_norm_ (FSDP's clip is
        # model-scoped: it all-reduces the sharded local norms).
        self._fsdp_root: FullyShardedDataParallel | None = None

    def _init_process_group(self) -> None:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            device_id=self.device,
        )

    def destroy(self) -> None:
        """Destroy the process group once, if this accelerator initialized it."""
        if self._destroyed:
            return
        if self._process_group_initialized:
            if dist.is_initialized():
                dist.destroy_process_group()
            self._process_group_initialized = False
        self._destroyed = True

    def unwrap_model(self, model: Any) -> Any:
        """Return the user model, matching Accelerate's unwrap semantics.

        Callers must keep the prepared FSDP object for forward/backward and
        FSDP state-dict APIs.  This method is only for inspecting the original
        module or APIs that explicitly require it.
        """
        return model.module if isinstance(model, FullyShardedDataParallel) else model

    def prepare(self, model: Any, optimizer: Any = None) -> Any:
        ret_model = self.prepare_model(model)
        if optimizer is not None:
            ret_optimizer = self.prepare_optimizer(optimizer, model=ret_model)
            return ret_model, ret_optimizer
        return ret_model

    def prepare_model(self, module: Any) -> Any:
        module = module.to(self.device)
        module = self._wrap_forward_autocast(module)
        if self.world_size < 2:
            return module

        # Wrap sufficiently large child modules so a ResNet does not
        # all-gather its entire parameter set for every forward.  The policy
        # remains model-agnostic and can also split large MLP/attention blocks.
        auto_wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=1_000_000)
        wrapped = FullyShardedDataParallel(
            module,
            auto_wrap_policy=auto_wrap_policy,
            device_id=self.device,
            sync_module_states=True,
            use_orig_params=True,
        )

        # The experiment checkpoint format expects ordinary full tensors, which
        # FSDP's own default (sharded) Module.load_state_dict does not accept.
        # Route it through the distributed-checkpoint setter; that setter in
        # turn dispatches through Module.load_state_dict, so the patch lifts
        # itself for the duration of the call instead of recursing.
        def load_full_state_dict(_module: Any, state_dict: dict[str, Any], strict: bool = True) -> Any:
            patch = _module.__dict__.pop("load_state_dict", None)
            try:
                options = StateDictOptions(full_state_dict=True, cpu_offload=True, strict=strict)
                return set_model_state_dict(_module, state_dict, options=options)
            finally:
                if patch is not None:
                    _module.__dict__["load_state_dict"] = patch

        wrapped.load_state_dict = types.MethodType(load_full_state_dict, wrapped)
        self._fsdp_root = wrapped
        return wrapped

    def prepare_optimizer(self, optimizer: Any, model: Any = None) -> Any:
        """Bind an optimizer to FSDP parameters and adapt the checkpoint resume path."""
        self._register_optimizer(optimizer)
        if model is None or not isinstance(model, FullyShardedDataParallel):
            return optimizer

        # The generic checkpoint loader resumes through Optimizer.load_state_dict
        # with a full FQN-keyed state dict, which the raw optimizer cannot shard.
        # Route it through the distributed-checkpoint setter; that setter in
        # turn dispatches through Optimizer.load_state_dict, so the patch lifts
        # itself for the duration of the call instead of recursing.
        def load_full_state_dict(_optimizer: torch.optim.Optimizer, state: dict[str, Any]) -> Any:
            patch = _optimizer.__dict__.pop("load_state_dict", None)
            try:
                return set_optimizer_state_dict(model, _optimizer, optim_state_dict=state, options=_FULL_STATE_DICT)
            finally:
                if patch is not None:
                    _optimizer.__dict__["load_state_dict"] = patch

        optimizer.load_state_dict = types.MethodType(load_full_state_dict, optimizer)
        return optimizer

    def backward(self, loss: torch.Tensor) -> None:
        self._scaler.scale(loss).backward()

    def autocast(self):
        if self._amp_dtype is None:
            return super().autocast()
        return torch.autocast(device_type="cuda", dtype=self._amp_dtype)

    def optimizer_step(self, optimizer: Any) -> Any:
        step_return = self._scaler.step(optimizer)
        self._scaler.update()
        return step_return

    def clip_grad_norm_(self, parameters: Any, max_norm: float, norm_type: float = 2) -> Any:
        """Clip with the global norm across sharded gradients, after undoing fp16 scaling.

        ``torch.nn.utils.clip_grad_norm_`` would measure each rank's local shard
        norm, so ranks would clip by different coefficients and silently drift
        apart; FSDP's clip all-reduces the local norms first. That clip is
        model-scoped (FSDP exposes no subset clipping): ``parameters`` is only
        used for the unscale preflight, and every wrapped gradient is clipped.
        """
        parameters = self._unscale_for_clip(parameters)
        if self.world_size < 2 or self._fsdp_root is None:
            return torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=norm_type)
        return self._fsdp_root.clip_grad_norm_(max_norm, norm_type=norm_type)

    def wait_for_everyone(self) -> None:
        if self.world_size > 1:
            dist.barrier(device_ids=[self.device.index])

    def reduce(self, tensor: torch.Tensor, reduction: str = "sum", scale: float = 1.0) -> torch.Tensor:
        if self.world_size < 2 or reduction == "none":
            return tensor
        device_tensor = tensor.to(self.device)
        if device_tensor is tensor:
            device_tensor = tensor.clone()
        dist.all_reduce(device_tensor, op=dist.ReduceOp.SUM)
        result = device_tensor.to(tensor.device)
        if reduction == "mean":
            result = result / self.world_size
        return result

    def dump_model_to_state_dict(self, module: nn.Module) -> dict[str, torch.Tensor]:
        """Gather a full CPU model state dict on rank 0; every rank must call this.

        With full_state_dict + cpu_offload the other ranks receive empty dicts,
        so callers follow the collective but only read the result on rank 0.
        """
        if not isinstance(module, FullyShardedDataParallel):
            return {key: value.cpu() for key, value in module.state_dict().items()}
        return get_model_state_dict(module, options=_FULL_STATE_DICT)

    def collect_checkpoint_state(
        self, module: nn.Module, optimizer: torch.optim.Optimizer
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """Collect the checkpoint payloads on rank 0; every rank must call this.

        These are collectives: with full_state_dict + cpu_offload the full
        tensors materialize on rank 0 and the other ranks get empty dicts, and
        the caller writes the checkpoint only from rank 0.
        """
        return (
            self.dump_model_to_state_dict(module),
            get_optimizer_state_dict(module, optimizer, options=_FULL_STATE_DICT),
        )
