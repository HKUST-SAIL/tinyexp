"""Repeated-augmentation sampler for distributed training.

Ported verbatim from facebookresearch/deit ``samplers.py`` at commit
``7e160fe43f0252d17191b71cbb5826254114ea5b`` (pinned in ``docs/vit_tp.md``). Only
formatting and type annotations differ from the original; the sampling logic is
unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import torch
import torch.distributed as dist


class RASampler(torch.utils.data.Sampler[int]):
    """Sampler that restricts data loading to a subset of the dataset for distributed,
    with repeated augmentation.
    It ensures that different each augmented version of a sample will be visible to a
    different process (GPU)
    Heavily based on torch.utils.data.DistributedSampler
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        num_repeats: int = 3,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if num_repeats < 1:
            raise ValueError("num_repeats should be greater than 0")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_repeats = num_repeats
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * self.num_repeats / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.num_selected_samples = int(math.floor(len(self.dataset) // 256 * 256 / self.num_replicas))
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)
        else:
            indices = torch.arange(start=0, end=len(self.dataset))

        # add extra samples to make it evenly divisible
        indices = torch.repeat_interleave(indices, repeats=self.num_repeats, dim=0).tolist()
        padding_size: int = self.total_size - len(indices)
        if padding_size > 0:
            indices += indices[:padding_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices[: self.num_selected_samples])

    def __len__(self) -> int:
        return self.num_selected_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
