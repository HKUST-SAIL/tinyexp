import torch

from .base_accelerator import BaseAccelerator


class CPUAccelerator(BaseAccelerator):
    """
    CPU accelerator for distributed training.
    """

    def __init__(self) -> None:
        super().__init__()

    def _init_process_group(self) -> None:
        pass

    def unwrap_model(self, model):
        return model

    def prepare(self, model, optimizer=None):
        model.to(self.device)
        if optimizer is not None:
            optimizer = self.prepare_optimizer(optimizer)
            return model, optimizer
        else:
            return model

    def prepare_optimizer(self, optimizer):
        return optimizer

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    def wait_for_everyone(self) -> None:
        pass

    def reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def print(self, *args, **kwargs) -> None:
        print(*args, **kwargs)
