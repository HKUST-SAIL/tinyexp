import itertools
from collections.abc import Iterator
from typing import Any, Optional

import torch
from torch.utils.data.sampler import Sampler

__all__ = ["InfiniteSampler"]


class InfiniteSampler(Sampler[int]):
    """Yield an infinite, deterministic, rank-sharded stream of indices.

    Stream model:

    - With ``shuffle=True``, the global stream is an endless sequence of
      shuffled ``range(size)`` permutations generated from ``seed``. With
      ``shuffle=False``, it is ``range(size)`` repeated forever.
    - Rank ``r`` consumes ``global_stream[r::world_size]``. Distributed ranks
      therefore need the same resolved seed to consume disjoint positions from
      one global stream.
    - The iterator never raises ``StopIteration``. Training code must impose a
      finite step count. Create one iterator and keep consuming it across
      logical epochs; there is no need to recreate it or call ``set_epoch`` at
      every epoch.

    Logical epoch and length:

    - ``len(sampler)`` is a nominal per-rank epoch length used for bookkeeping;
      it is not the finite length of the iterator.
    - This sampler's ``drop_last`` only selects ``floor(size / world_size)``
      instead of ``ceil(size / world_size)`` for that nominal length. It does
      not truncate the stream or permanently discard indices. The floor can be
      zero when ``size < world_size``.
    - When ``size`` is not divisible by ``world_size``, a nominal epoch is not
      guaranteed to contain every dataset index exactly once. With
      ``drop_last=False`` it may cross a permutation boundary; with
      ``drop_last=True`` the remainder continues into a later logical epoch.
    - ``DataLoader(drop_last=...)`` is separate from this sampler option.
      Because the source iterator is infinite, standard DataLoader auto-
      batching keeps yielding full batches forever. ``len(dataloader)`` is
      merely computed from ``len(sampler)`` with batch rounding. Consuming
      ``len(dataloader)`` batches can therefore consume more or fewer than
      ``len(sampler)`` indices per rank when the batch size does not divide the
      nominal sampler length.

    Resume contract:

    - Resume is supported at logical epoch boundaries only. Call
      ``set_epoch(epoch)`` before ``iter(sampler)`` or ``iter(dataloader)``.
      Calling it afterwards has no effect on that existing iterator.
    - Exact sampler-index continuation requires each completed logical epoch to
      have consumed exactly ``len(sampler)`` indices per rank.
    - Dataset size and index semantics, resolved seed, ``shuffle``, sampler
      ``drop_last``, world size, rank assignment, batch size, DataLoader
      ``drop_last``, batch sampler, and steps per epoch must remain unchanged.
    - If the batching loop consumes a different number of indices, the method
      restores only the nominal epoch. Boundary indices may repeat or be
      skipped relative to an uninterrupted run, while subsequent sampling
      still follows the configured stream.
    - This sampler does not capture model or transform RNG, Python/NumPy/Torch
      RNG, DataLoader worker RNG, worker-local dataset state, or prefetched
      batches. Full training resume is therefore epoch-level and statistical,
      not bit-exact. Mid-epoch or iteration-level resume is intentionally out
      of scope.
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
            seed: The initial shuffle seed. ``None`` resolves a random seed
                once when this sampler is constructed. Distributed callers
                must pass the same explicit seed to every rank, or synchronize
                the resolved seed themselves.
            drop_last: Whether a logical epoch uses the floor rather than the
                ceiling number of samples per rank. This sampler-level option
                only affects ``len`` and the position selected by
                ``set_epoch``; it is independent of DataLoader ``drop_last``.
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

        The position is ``epoch * len(self) * world_size`` in the global
        stream. This only affects iterators created after the call. The sampler
        deliberately does not expose iteration-level state; exact index
        continuation depends on the consumption and configuration contract
        documented on :class:`InfiniteSampler`. Positioning is implemented by
        replaying and skipping the deterministic stream, so a large epoch
        offset may add startup work when the new iterator is first consumed.
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
