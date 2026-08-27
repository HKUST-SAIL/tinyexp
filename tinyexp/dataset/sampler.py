import itertools
from collections.abc import Iterator
from typing import Any, Optional

import torch
from torch.utils.data.sampler import Sampler

__all__ = ["InfiniteSampler"]


class InfiniteSampler(Sampler[int]):
    """Sample an infinite, rank-sharded stream of dataset indices.

    The global stream is ``shuffle(range(size))`` repeated forever when
    ``shuffle`` is true, or ``range(size)`` repeated forever otherwise. Each
    rank consumes ``global_stream[rank::world_size]``. This is a continuous
    stream sampler; call ``set_epoch`` before creating the iterator to resume
    from a logical epoch boundary. An existing iterator always continues the
    same stream and does not need to be recreated between epochs.

    ``drop_last`` affects only the logical epoch length returned by ``len``.
    It does not stop the infinite stream or remove samples permanently. With
    a non-divisible ``size``, ``drop_last=False`` gives each rank the ceiling
    number of samples per logical epoch, while ``drop_last=True`` gives the
    floor number.

    Resume contract:

    - Resume is supported only at logical epoch boundaries. Call
      ``set_epoch`` before creating the sampler/DataLoader iterator; calling it
      after iterator creation does not reposition that iterator.
    - ``set_epoch(epoch)`` assumes every completed logical epoch consumed
      exactly ``len(self)`` indices per rank.
    - Resume must keep the dataset size and index order, ``seed``, ``shuffle``,
      ``drop_last``, world size, and rank assignment unchanged.
    - The batching contract must also stay unchanged. In particular, changing
      batch size, DataLoader ``drop_last``, batch sampler, or steps per epoch
      when continuing from a checkpoint is unsupported and may change the
      resumed stream position.
    - Iteration-level or mid-epoch resume is intentionally not supported. If
      batching consumes a different number of indices than ``len(self)`` per
      rank, only the nominal epoch is restored, not the exact next batch.
    """

    def __init__(
        self,
        size: int,
        shuffle: bool = True,
        seed: Optional[int] = 0,
        drop_last: bool = False,
        accelerator: Any = None,
    ):
        """Create an infinite stream sampler.

        Args:
            size: The positive number of samples in the underlying dataset.
            shuffle: Whether to shuffle each repeated dataset permutation.
            seed: The initial shuffle seed. ``None`` creates a random seed
                once when the sampler is constructed. Distributed callers
                should pass the same explicit seed to every rank.
            drop_last: Whether a logical epoch uses the floor rather than the
                ceiling number of samples per rank. This only affects ``len``
                and the position selected by ``set_epoch``.
            accelerator: Optional object providing ``rank`` and ``world_size``.
        """
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise ValueError("size must be a positive integer")  # noqa: TRY003

        if accelerator is not None:
            rank = accelerator.rank
            world_size = accelerator.world_size
        else:
            rank = 0
            world_size = 1

        if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size < 1:
            raise ValueError("world_size must be a positive integer")  # noqa: TRY003
        if not isinstance(rank, int) or isinstance(rank, bool) or not 0 <= rank < world_size:
            raise ValueError("rank must be an integer in [0, world_size)")  # noqa: TRY003

        self._size = size
        self._shuffle = shuffle
        self._seed = torch.Generator().seed() if seed is None else int(seed)
        self.drop_last = drop_last
        self._rank = rank
        self._world_size = world_size
        self._sample_offset = 0

    def set_epoch(self, epoch: int) -> None:
        """Position the continuous stream at the start of logical ``epoch``.

        This only affects iterators created after the call. The sampler
        deliberately does not expose iteration-level resume state. The caller
        must keep the resume contract documented on ``InfiniteSampler`` and
        consume exactly ``len(self)`` indices per rank for each completed
        logical epoch.
        """
        epoch = int(epoch)
        if epoch < 0:
            raise ValueError("epoch must not be negative")  # noqa: TRY003
        self._sample_offset = epoch * len(self) * self._world_size

    def __iter__(self) -> Iterator[int]:
        start = self._rank + self._sample_offset
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            if self._shuffle:
                yield from torch.randperm(self._size, generator=g).tolist()
            else:
                yield from range(self._size)

    def __len__(self) -> int:
        if self.drop_last:
            return self._size // self._world_size
        return (self._size + self._world_size - 1) // self._world_size
