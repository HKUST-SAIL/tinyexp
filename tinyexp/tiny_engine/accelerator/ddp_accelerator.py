import torch
import torch.distributed as dist
from torch import nn

from .base_accelerator import BaseAccelerator


class DDPAcceleratedOptimizer(torch.optim.Optimizer):
    """
    Internal wrapper around a torch optimizer.

    Conditionally will perform `step` and `zero_grad` if gradients should be synchronized when performing gradient
    accumulation.

    Args:
        optimizer (`torch.optim.optimizer.Optimizer`):
            The optimizer to wrap.
        scaler (`torch.cuda.amp.grad_scaler.GradScaler`, *optional*):
            The scaler to use in the step function if training with mixed precision.
    """

    def __init__(self, optimizer, scaler):
        self.optimizer = optimizer
        self.scaler = scaler

    @property
    def state(self):
        return self.optimizer.state

    @state.setter
    def state(self, state):
        self.optimizer.state = state

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @param_groups.setter
    def param_groups(self, param_groups):
        self.optimizer.param_groups = param_groups

    @property
    def defaults(self):
        return self.optimizer.defaults

    @defaults.setter
    def defaults(self, defaults):
        self.optimizer.defaults = defaults

    def add_param_group(self, param_group):
        self.optimizer.add_param_group(param_group)

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def state_dict(self):
        return self.optimizer.state_dict()

    def zero_grad(self, set_to_none=None):
        self.optimizer.zero_grad()

    def train(self) -> None:
        """
        Sets the optimizer to "train" mode. Useful for optimizers like `schedule_free`
        """
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

    def eval(self) -> None:
        """
        Sets the optimizer to "eval" mode. Useful for optimizers like `schedule_free`
        """
        if hasattr(self.optimizer, "eval") and callable(self.optimizer.eval):
            self.optimizer.eval()

    def step(self, closure=None) -> None:
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
            self.scaler.step(self.optimizer, closure)
            self.scaler.update()
        else:
            self.optimizer.step(closure)

    def _switch_parameters(self, parameters_map) -> None:
        for param_group in self.optimizer.param_groups:
            param_group["params"] = [parameters_map.get(p, p) for p in param_group["params"]]


class DDPAccelerator(BaseAccelerator):
    def __init__(self, mixed_precision=None):
        super().__init__()
        assert torch.cuda.is_available(), "DDPAccelerator requires CUDA to be available."
        self._init_process_group()

        self.mixed_precision = mixed_precision
        self.dtype = None
        if self.mixed_precision == torch.float16 or self.mixed_precision == torch.bfloat16:
            self.grad_scaler = torch.amp.GradScaler(self.device.type)
            self.autocast = torch.autocast(device_type=self.device.type, dtype=mixed_precision)
            self.backward = self._amp_backward
            self.dtype = mixed_precision
        else:
            self.backward = self._naive_backward
            self.dtype = torch.float32

    def _init_process_group(self):
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=self.rank,
            world_size=self.world_size,
        )

    def unwrap_model(self, model):
        return model.module if hasattr(model, "module") else model

    def prepare(self, model, optimizer=None):
        ret_model = self.prepare_model(model)
        if optimizer is not None:
            ret_optimizer = self.prepare_optimizer(optimizer)
            return ret_model, ret_optimizer
        else:
            return ret_model

    def prepare_model(self, model):
        if self._model is None:
            model.to(self.device)
            self._model = nn.parallel.DistributedDataParallel(
                model,
                # device_ids=[self.local_rank],
                # find_unused_parameters=self.find_unused_parameters,
                # broadcast_buffers=self.broadcast_buffers,
            )

        if self.mixed_precision == torch.bfloat16 or self.mixed_precision == torch.float16:

            def new_forward(old_forward, *args, **kwargs):
                with self.autocast:
                    return old_forward(*args, **kwargs)

            self._model.forward = new_forward.__get__(self._model.forward)
        return self._model

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

        if self._optimizer is None:
            self._optimizer = optimizer
            optimizer_to(self._optimizer, self.device)

        if self.mixed_precision == torch.bfloat16 or self.mixed_precision == torch.float16:
            self._optimizer = DDPAcceleratedOptimizer(self._optimizer, self.grad_scaler)
        return self._optimizer

    def _naive_backward(self, loss: torch.Tensor):
        loss.backward()

    def _amp_backward(self, loss: torch.Tensor):
        self.grad_scaler.scale(loss).backward()

    def backward(self, loss: torch.Tensor):
        pass

    def load_model_from_state_dict(self, state_dict, *args, **kwargs):
        return self._model.module.load_state_dict(state_dict, *args, **kwargs)

    def dump_model_to_state_dict(self):
        """
        dump model to cpu state_dict
        """
        model_state = self._model.module.state_dict()
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

    def reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        world_size = self.world_size
        if world_size < 2:
            return tensor
        tensor = tensor.clone()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    def print(self, *args, **kwargs) -> None:
        if self.is_local_main_process:
            print(*args, **kwargs)
