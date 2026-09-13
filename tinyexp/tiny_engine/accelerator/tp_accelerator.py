"""Tensor-parallel accelerator built on ``torch.distributed.tensor.parallel`` (docs/vit_tp.md).

V1 scope is pure tensor parallelism: ``tp_size == world_size``. Every rank receives the
same input batch; the sharded Linear layers compute partial results that are combined by
an all-reduce inside each transformer block, so the block outputs (and thus the model
outputs) are replicated on all ranks. This is the classic Megatron-style block layout:

- ``attn.q``/``attn.k``/``attn.v`` (ColwiseParallel): shard the output dim on head
  boundaries. Attention is per-head independent, so each rank runs its own heads'
  ``softmax(q k^T) v`` with no communication inside attention.
- ``attn.proj`` (RowwiseParallel): input is Shard(-1) (its local heads' channel chunk),
  the rowwise matmul produces partial sums and one forward all-reduce restores a
  replicated output.
- ``mlp.fc1`` (ColwiseParallel) + ``mlp.fc2`` (RowwiseParallel): the same pattern for
  the MLP, one all-reduce per block pass.

The caller supplies the ``parallelize_plan`` (the model layer layout is not the
accelerator's business); see ``tinyexp/examples/vit_tp_exp.py::build_tp_parallelize_plan``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import parallelize_module

from .base_accelerator import BaseAccelerator


class TPAccelerator(BaseAccelerator):
    """Accelerator that shards transformer Linear layers across the whole world."""

    def __init__(self) -> None:
        super().__init__()
        if torch.cuda.is_available():
            if self.device.index is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
            torch.cuda.set_device(self.device)
            self.backend = "nccl"
            device_type = "cuda"
        else:
            # CPU + gloo keeps the TP code path testable in CI.
            self.device = torch.device("cpu")
            self.backend = "gloo"
            device_type = "cpu"

        if self.world_size > 1:
            if not dist.is_initialized():
                self._init_process_group()
                self._process_group_initialized = True
            # A 1D whole-world mesh reuses the default process group (already
            # initialized with the backend selected above). torch 2.7's
            # init_device_mesh has no backend kwarg.
            self.mesh: DeviceMesh | None = init_device_mesh(device_type, (self.world_size,))
        else:
            self.mesh = None
        self.sync_gradients = True

    def _init_process_group(self) -> None:
        dist.init_process_group(backend=self.backend, init_method="env://")

    def destroy(self) -> None:
        """Destroy the process group once, if this accelerator owns it."""
        if self._destroyed:
            return
        if self._process_group_initialized:
            if dist.is_initialized():
                dist.destroy_process_group()
            self._process_group_initialized = False
        self._destroyed = True

    def unwrap_model(self, model: Any) -> Any:
        # parallelize_module patches modules in place; there is no wrapper to undo.
        return model.module if hasattr(model, "module") else model

    def prepare(self, model: Any, optimizer: Any = None, parallelize_plan: dict | None = None) -> Any:
        ret_model = self.prepare_model(model, parallelize_plan=parallelize_plan)
        if optimizer is not None:
            ret_optimizer = self.prepare_optimizer(optimizer)
            return ret_model, ret_optimizer
        return ret_model

    def prepare_model(self, module: Any, parallelize_plan: dict | None = None) -> Any:
        module = module.to(self.device)
        if self.world_size < 2:
            # Degenerate path: a single rank is mathematically the unparallelized model.
            return module
        if parallelize_plan is None:
            raise ValueError("TPAccelerator requires a parallelize_plan at world_size > 1")  # noqa: TRY003
        module = parallelize_module(module, self.mesh, parallelize_plan)
        # ColwiseParallel/RowwiseParallel (use_local_output=True, the torch default)
        # annotate inputs/outputs at module boundaries; the tensors between a colwise
        # output and its paired rowwise input are plain local tensors sharded on the
        # last dim, which the model's forward must account for with local-shape math
        # (see the reshape(-1) adjustments in vit_tp_exp.Attention, docs/vit_tp.md D2).
        return module

    def prepare_optimizer(self, optimizer: Any) -> Any:
        # Optimizers step DTensor parameters directly; DTensor handles the cross-shard
        # gradient bookkeeping during backward.
        return optimizer

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    def wait_for_everyone(self) -> None:
        if self.world_size < 2:
            return
        dist.barrier()

    def reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.world_size < 2:
            return tensor
        # NCCL only reduces device-resident tensors; move over and back. ``to``
        # is a no-op when the input is already on this device, while all_reduce
        # is in-place, so copy only in that aliasing case to preserve the input.
        device_tensor = tensor.to(self.device)
        if device_tensor is tensor:
            device_tensor = tensor.clone()
        dist.all_reduce(device_tensor, op=dist.ReduceOp.SUM)
        return device_tensor.to(tensor.device)

    def reduce_mean(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.reduce_sum(tensor) / self.world_size

    def dump_model_to_state_dict(self, module: Any) -> dict:
        """
        dump model to a plain cpu state_dict, gathering DTensor shards to full tensors
        """
        model_state = module.state_dict()
        model_state_cpu = type(model_state)()
        for key, val in model_state.items():
            if isinstance(val, DTensor):
                val = val.full_tensor()
            model_state_cpu[key] = val.cpu()
        return model_state_cpu

    @property
    def is_main_process(self) -> bool:
        """True for one process per server."""
        return self.rank == 0

    @property
    def is_local_main_process(self) -> bool:
        """True for one process per server."""
        return self.local_rank == 0

    @property
    def is_last_process(self) -> bool:
        return self.rank == self.world_size - 1

    def print(self, *args: Any, **kwargs: Any) -> None:
        if self.is_local_main_process:
            print(*args, **kwargs)
