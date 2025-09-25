import abc
import os
from abc import abstractmethod

import torch

__all__ = ["BaseAccelerator"]


class BaseAccelerator(abc.ABC):
    """
    basic accelerator, provide basic functions for distributed training.
    """

    def __init__(self) -> None:
        self.rank = int(os.getenv("RANK", 0))
        self.world_size = int(os.getenv("WORLD_SIZE", 1))
        self.local_rank = int(os.getenv("LOCAL_RANK", 0))
        self.local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", 1))
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
    def backward(self, loss: torch.Tensor) -> None:
        pass

    @abstractmethod
    def wait_for_everyone(self) -> None:
        pass

    @abstractmethod
    def reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def print(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        pass

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
