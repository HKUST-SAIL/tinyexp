import itertools
from typing import Optional

import torch
from torch.utils.data.sampler import Sampler

__all__ = ["InfiniteSampler"]


class InfiniteSampler(Sampler):
    """
    In training, we only care about the "infinite stream" of training data.
    So this sampler produces an infinite stream of indices and
    all workers cooperate to correctly shuffle the indices and sample different indices.
    The samplers in each worker effectively produces `indices[worker_id::num_workers]`
    where `indices` is an infinite stream of indices consisting of
    `shuffle(range(size)) + shuffle(range(size)) + ...` (if shuffle is True)
    or `range(size) + range(size) + ...` (if shuffle is False)
    """

    def __init__(self, size: int, shuffle: bool = True, seed: Optional[int] = 0, drop_last=False, accelerator=None):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
            drop_last (bool): whether to drop the last incomplete batch
        """
        self._size = size
        self._shuffle = shuffle
        self._seed = int(seed)
        self.drop_last = drop_last
        if accelerator is not None:
            self._rank = accelerator.rank
            self._world_size = accelerator.world_size
        else:
            self._rank = 0
            self._world_size = 1

    def set_epoch(self, epoch):
        pass

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            if self._shuffle:
                yield from torch.randperm(self._size, generator=g).tolist()
            else:
                # yield from torch.arange(self._size)
                yield from list(range(self._size))

    def __len__(self):
        if self.drop_last:
            return self._size // self._world_size
        else:
            return (self._size + self._world_size - 1) // self._world_size
