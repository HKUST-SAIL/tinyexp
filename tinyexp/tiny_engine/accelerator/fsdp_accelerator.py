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
from torch.distributed.fsdp import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
    FullyShardedDataParallel,
    StateDictType,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

from ...exceptions import CudaNotAvailableError
from .base_accelerator import BaseAccelerator


class FSDPAccelerator(BaseAccelerator):
    """Accelerator for parameter, gradient, and optimizer-state sharding."""

    # ResNet's generic training loop builds the optimizer before calling
    # ``prepare``. FSDP replaces the module parameters, so that order must be
    # reversed for this accelerator.
    requires_optimizer_after_model = True

    def __init__(self) -> None:
        super().__init__()
        if not torch.cuda.is_available():
            raise CudaNotAvailableError()

        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(self.device)
        if self.world_size > 1 and not dist.is_initialized():
            self._init_process_group()
            self._process_group_initialized = True
        self.sync_gradients = True

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
        # The experiment checkpoint format expects ordinary full tensors.  All
        # ranks participate in these collectives; only the caller writes rank 0.
        FullyShardedDataParallel.set_state_dict_type(
            wrapped,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
            FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
        )
        return wrapped

    def prepare_optimizer(self, optimizer: Any, model: Any = None) -> Any:
        """Bind an optimizer to FSDP parameters and adapt checkpoint state APIs."""
        if model is None or not isinstance(model, FullyShardedDataParallel):
            return optimizer

        # FSDP.optim_state_dict() must receive the optimizer's untransformed
        # state. Capture the original methods before installing the adapters.
        raw_state_dict = optimizer.state_dict
        raw_load_state_dict = optimizer.load_state_dict

        def state_dict(_optimizer: torch.optim.Optimizer) -> dict[str, Any]:
            raw_state = raw_state_dict()
            return FullyShardedDataParallel.optim_state_dict(model, _optimizer, optim_state_dict=raw_state)

        def load_state_dict(_optimizer: torch.optim.Optimizer, state: dict[str, Any]) -> Any:
            sharded_state = FullyShardedDataParallel.optim_state_dict_to_load(model, _optimizer, state)
            return raw_load_state_dict(sharded_state)

        optimizer.state_dict = types.MethodType(state_dict, optimizer)
        optimizer.load_state_dict = types.MethodType(load_state_dict, optimizer)
        return optimizer

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    def wait_for_everyone(self) -> None:
        if self.world_size > 1:
            dist.barrier(device_ids=[self.device.index])

    def reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.world_size < 2:
            return tensor
        device_tensor = tensor.to(self.device)
        if device_tensor is tensor:
            device_tensor = tensor.clone()
        dist.all_reduce(device_tensor, op=dist.ReduceOp.SUM)
        return device_tensor.to(tensor.device)

    def reduce_mean(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.reduce_sum(tensor) / self.world_size

    def dump_model_to_state_dict(self, module: nn.Module) -> dict[str, torch.Tensor]:
        """Gather a full CPU model state dict; every rank must call this."""
        if not isinstance(module, FullyShardedDataParallel):
            return {key: value.cpu() for key, value in module.state_dict().items()}
        with FullyShardedDataParallel.state_dict_type(
            module,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        ):
            return {key: value.cpu() for key, value in module.state_dict().items()}

    def collect_checkpoint_state(
        self, module: nn.Module, optimizer: torch.optim.Optimizer
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """Collect full model and optimizer state on every rank.

        FSDP state-dict APIs are collectives.  The caller must invoke this on
        every rank and write the returned dictionaries only from rank 0.
        """
        return self.dump_model_to_state_dict(module), optimizer.state_dict()

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    @property
    def is_local_main_process(self) -> bool:
        return self.local_rank == 0

    @property
    def is_last_process(self) -> bool:
        return self.rank == self.world_size - 1

    def print(self, *args: Any, **kwargs: Any) -> None:
        if self.is_local_main_process:
            print(*args, **kwargs)
