from contextlib import suppress

import torch
import torch.distributed as dist

from .base_accelerator import BaseAccelerator


class CPUAccelerator(BaseAccelerator):
    """
    CPU accelerator for distributed training.
    """

    def __init__(self) -> None:
        super().__init__()
        if self.world_size > 1:
            self._init_process_group()
            self._process_group_initialized = True
        else:
            self._process_group_initialized = False

    def _init_process_group(self) -> None:
        dist.init_process_group(
            backend="gloo",
            init_method="env://",
            rank=self.rank,
            world_size=self.world_size,
        )

    def destroy(self):
        """Explicitly destroy the distributed process group"""
        if self._process_group_initialized:
            if dist.is_initialized():
                dist.destroy_process_group()
            self._process_group_initialized = False

    def __del__(self):
        """Destructor, which is automatically called when the object is garbage collected"""
        with suppress(Exception):
            self.destroy()

    def unwrap_model(self, model):
        return model

    def prepare(self, model, optimizer=None):
        model.to(self.device)
        if optimizer is not None:
            optimizer = self.prepare_optimizer(optimizer)
            return model, optimizer
        else:
            return model

    def prepare_model(self, model):
        model.to(self.device)
        return model

    def prepare_optimizer(self, optimizer):
        return optimizer

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    def wait_for_everyone(self) -> None:
        if self.world_size > 1:
            dist.barrier()

    def reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.world_size < 2:
            return tensor
        tensor = tensor.clone().to(self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    def reduce_mean(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.reduce_sum(tensor) / self.world_size

    def print(self, *args, **kwargs) -> None:
        if self.rank == 0:
            print(*args, **kwargs)
