from contextlib import suppress
from typing import Literal

import torch
import torch.distributed as dist
from torch import nn

from ...exceptions import CudaNotAvailableError
from .base_accelerator import BaseAccelerator


class DDPAccelerator(BaseAccelerator):
    def __init__(self):
        super().__init__()
        if not torch.cuda.is_available():
            raise CudaNotAvailableError()
        self._init_process_group()
        self._process_group_initialized = True  # Mark that the process group has been initialized
        self.sync_gradients = True  # currently not support accumulate gradient

    def _init_process_group(self):
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            # rank=self.rank,
            # world_size=self.world_size,
        )

    def destroy(self):
        """Explicitly destroy the distributed process group"""
        if hasattr(self, "_process_group_initialized") and self._process_group_initialized:
            if dist.is_initialized():
                dist.destroy_process_group()
            self._process_group_initialized = False

    def __del__(self):
        """Destructor, which is automatically called when the object is garbage collected"""
        with suppress(Exception):
            self.destroy()

    def unwrap_model(self, model):
        return model.module if hasattr(model, "module") else model

    def prepare(self, model, optimizer=None):
        ret_model = self.prepare_model(model)
        if optimizer is not None:
            ret_optimizer = self.prepare_optimizer(optimizer)
            return ret_model, ret_optimizer
        else:
            return ret_model

    def prepare_model(self, module):
        module.to(self.device)
        wrapped_module = nn.parallel.DistributedDataParallel(
            module,
            # device_ids=[self.local_rank],
            # find_unused_parameters=self.find_unused_parameters,
            # broadcast_buffers=self.broadcast_buffers,
        )
        return wrapped_module

    def prepare_optimizer(self, optimizer):
        # refer to: https://github.com/pytorch/pytorch/issues/8741
        def optimizer_to(optim, device):
            for param in optim.state.values():
                # Not sure there are any global tensors in the state dict
                if isinstance(param, torch.Tensor):
                    param.data = param.data.to(device)
                    if param._grad is not None:
                        param._grad.data = param._grad.data.to(device)
                elif isinstance(param, dict):
                    for subparam in param.values():
                        if isinstance(subparam, torch.Tensor):
                            subparam.data = subparam.data.to(device)
                            if subparam._grad is not None:
                                subparam._grad.data = subparam._grad.data.to(device)

        optimizer_to(optimizer, self.device)
        return optimizer

    def backward(self, loss: torch.Tensor):
        loss.backward()

    def dump_model_to_state_dict(self, module: nn.Module) -> dict:
        """
        dump model to cpu state_dict
        """
        model_state = module.state_dict()
        model_state_cpu = type(model_state)()
        for key, val in model_state.items():
            model_state_cpu[key] = val.cpu()
        return model_state_cpu

    @property
    def is_main_process(self):
        """True for one process per server."""
        return self.rank == 0

    @property
    def is_local_main_process(self) -> bool:
        """True for one process per server."""
        return self.local_rank == 0

    @property
    def is_last_process(self) -> bool:
        return self.rank == self.world_size - 1

    def wait_for_everyone(self) -> None:
        if self.world_size < 2:
            return
        dist.barrier()

    def reduce(self, tensor, reduction: Literal["sum", "mean"] = "sum", scale=1.0):
        if reduction == "sum":
            return self.reduce_sum(tensor)
        elif reduction == "mean":
            return self.reduce_mean(tensor)

    def clip_grad_norm_(self, parameters, max_norm, norm_type=2):
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=norm_type)

    def reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        world_size = self.world_size
        if world_size < 2:
            return tensor
        tensor = tensor.clone()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    def reduce_mean(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.reduce_sum(tensor) / self.world_size

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Gather tensors from all processes to the main process (rank 0).
        Only rank 0 will have the gathered result, other ranks will return None.
        """
        world_size = self.world_size
        if world_size < 2:
            return tensor

        if self.rank == 0:
            # Main process: gather tensors from all processes
            gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
            dist.gather(tensor, gather_list, dst=0)
            return torch.cat(gather_list, dim=0)
        else:
            # Other processes: send tensor to main process
            dist.gather(tensor, dst=0)
            return tensor

    def print(self, *args, **kwargs) -> None:
        if self.is_local_main_process:
            print(*args, **kwargs)
