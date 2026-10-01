import torch
import torch.distributed as dist
from torch import nn

from ...exceptions import CudaNotAvailableError
from .base_accelerator import BaseAccelerator

_MIXED_PRECISION_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


class DDPAccelerator(BaseAccelerator):
    def __init__(self, mixed_precision: str = "none"):
        super().__init__()
        if not torch.cuda.is_available():
            raise CudaNotAvailableError()
        if mixed_precision != "none" and mixed_precision not in _MIXED_PRECISION_DTYPES:
            raise ValueError(f"Unknown mixed precision {mixed_precision!r}; expected none/fp16/bf16")  # noqa: TRY003

        # Select the concrete local CUDA device before initializing NCCL.
        # Ray workers may expose only one GPU, in which case the visible
        # device is local index 0 even when LOCAL_RANK is non-zero.
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(self.device)

        # Match Accelerate's normal single-process behavior: no process group
        # and no DDP wrapper when there is only one worker.
        if self.world_size > 1 and not dist.is_initialized():
            self._init_process_group()
            self._process_group_initialized = True
        self.sync_gradients = True  # currently not support accumulate gradient

        # fp16 gradients need loss scaling to stay in range; bf16 does not.
        # A disabled GradScaler passes scale/step through unchanged, so one
        # code path covers every mode.
        self._amp_dtype = _MIXED_PRECISION_DTYPES.get(mixed_precision)
        self._scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision == "fp16")

    def _init_process_group(self):
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            device_id=self.device,
        )

    def destroy(self):
        """Destroy the distributed process group once, if this accelerator owns it."""
        if self._destroyed:
            return
        if self._process_group_initialized:
            if dist.is_initialized():
                dist.destroy_process_group()
            self._process_group_initialized = False
        self._destroyed = True

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
        module = self._wrap_forward_autocast(module)
        if self.world_size < 2:
            return module

        device_index = self.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        return nn.parallel.DistributedDataParallel(
            module,
            device_ids=[device_index],
            output_device=device_index,
        )

    def prepare_optimizer(self, optimizer):
        # Optimizer state only materializes after steps; resume-time device
        # placement is handled by torch's Optimizer.load_state_dict, which
        # casts loaded state to each parameter's device.
        return self._register_optimizer(optimizer)

    def backward(self, loss: torch.Tensor):
        self._scaler.scale(loss).backward()

    def autocast(self):
        if self._amp_dtype is None:
            return super().autocast()
        return torch.autocast(device_type="cuda", dtype=self._amp_dtype)

    def optimizer_step(self, optimizer):
        step_return = self._scaler.step(optimizer)
        self._scaler.update()
        return step_return

    def dump_model_to_state_dict(self, module: nn.Module) -> dict:
        """Dump a full cpu state_dict; the DDP wrapper's ``module.`` prefix is stripped."""
        if isinstance(module, nn.parallel.DistributedDataParallel):
            module = module.module
        return {key: value.cpu() for key, value in module.state_dict().items()}

    def wait_for_everyone(self) -> None:
        if self.world_size < 2:
            return
        dist.barrier(device_ids=[self.device.index])

    def reduce(self, tensor, reduction: str = "sum", scale: float = 1.0):
        world_size = self.world_size
        if world_size < 2 or reduction == "none":
            return tensor
        # NCCL only reduces device-resident tensors; move over and back so
        # cpu-side metric tensors also reduce correctly.
        device_tensor = tensor.to(self.device)
        if device_tensor is tensor:
            device_tensor = tensor.clone()
        dist.all_reduce(device_tensor, op=dist.ReduceOp.SUM)
        result = device_tensor.to(tensor.device)
        if reduction == "mean":
            result = result / world_size
        return result

    def clip_grad_norm_(self, parameters, max_norm, norm_type=2):
        # Gradients still carry the fp16 loss scale here, so clipping them
        # directly would compare a ~65536x inflated norm against max_norm and
        # never clip meaningfully. Unscale first, as accelerate does; the later
        # scaler.step() sees the optimizer is already unscaled and skips it.
        parameters = self._unscale_for_clip(parameters)
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=norm_type)
