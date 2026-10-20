from __future__ import annotations

from types import SimpleNamespace

import pytest
from torch.utils.data import DataLoader, Dataset

from tinyexp.dataset.fake_dataloader import HoldOnesampleDataLoader
from tinyexp.dataset.sampler import InfiniteSampler


def _take(sampler: InfiniteSampler, count: int) -> list[int]:
    stream = iter(sampler)
    return [next(stream) for _ in range(count)]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"size": 0}, "size"),
        ({"size": -1}, "size"),
        ({"size": 3, "accelerator": SimpleNamespace(rank=0, world_size=0)}, "world_size"),
        ({"size": 3, "accelerator": SimpleNamespace(rank=0, world_size=-1)}, "world_size"),
        ({"size": 3, "accelerator": SimpleNamespace(rank=-1, world_size=2)}, "rank"),
        ({"size": 3, "accelerator": SimpleNamespace(rank=2, world_size=2)}, "rank"),
    ],
)
def test_infinite_sampler_rejects_invalid_arguments(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        InfiniteSampler(**kwargs)


@pytest.mark.parametrize(
    ("size", "rank", "world_size", "expected"),
    [(3, 0, 1, [0, 1, 2, 0, 1, 2, 0]), (4, 1, 2, [1, 3, 1, 3, 1, 3])],
)
def test_infinite_sampler_no_shuffle_repeats_the_rank_slice(
    size: int, rank: int, world_size: int, expected: list[int]
) -> None:
    accelerator = SimpleNamespace(rank=rank, world_size=world_size)
    sampler = InfiniteSampler(size=size, shuffle=False, seed=0, accelerator=accelerator)

    assert _take(sampler, len(expected)) == expected


def test_infinite_sampler_non_divisible_stream_with_drop_last() -> None:
    rank_zero = SimpleNamespace(rank=0, world_size=2)
    rank_one = SimpleNamespace(rank=1, world_size=2)

    keep_remainder = InfiniteSampler(size=5, shuffle=False, accelerator=rank_zero, drop_last=False)
    drop_remainder = InfiniteSampler(size=5, shuffle=False, accelerator=rank_zero, drop_last=True)
    other_rank = InfiniteSampler(size=5, shuffle=False, accelerator=rank_one, drop_last=False)

    assert len(keep_remainder) == 3
    assert len(drop_remainder) == 2
    assert _take(keep_remainder, 4) == [0, 2, 4, 1]
    assert _take(other_rank, 4) == [1, 3, 0, 2]

    drop_remainder.set_epoch(1)
    assert _take(drop_remainder, 2) == [4, 1]


def test_infinite_sampler_seed_none_is_resolved_once() -> None:
    sampler = InfiniteSampler(size=5, shuffle=True, seed=None)
    first_values = _take(sampler, 10)

    sampler.set_epoch(0)
    assert _take(sampler, 10) == first_values


def test_infinite_sampler_set_epoch_continues_fixed_stream() -> None:
    expected_values = _take(InfiniteSampler(size=10, shuffle=True, seed=17), 20)

    sampler = InfiniteSampler(size=10, shuffle=True, seed=17)
    sampler.set_epoch(1)
    assert _take(sampler, 10) == expected_values[10:20]

    sampler.set_epoch(0)
    assert _take(sampler, 10) == expected_values[:10]


def test_infinite_sampler_set_epoch_preserves_distributed_stream_position() -> None:
    expected_global = _take(InfiniteSampler(size=12, shuffle=True, seed=23), 40)

    sampler = InfiniteSampler(size=12, shuffle=True, seed=23, accelerator=SimpleNamespace(rank=1, world_size=2))
    sampler.set_epoch(2)

    assert _take(sampler, 5) == expected_global[2 * 6 * 2 + 1 :: 2][:5]


def test_infinite_sampler_reuses_one_dataloader_iterator_across_epochs() -> None:
    class IndexDataset(Dataset[int]):
        def __len__(self) -> int:
            return 4

        def __getitem__(self, index: int) -> int:
            return index

    class CountingInfiniteSampler(InfiniteSampler):
        iterator_count = 0

        def __iter__(self):
            type(self).iterator_count += 1
            return super().__iter__()

    sampler = CountingInfiniteSampler(size=4, shuffle=False, seed=0)
    dataloader = DataLoader(IndexDataset(), batch_size=2, sampler=sampler, num_workers=0)
    iterator = iter(dataloader)

    epochs = [[next(iterator).tolist() for _ in range(len(dataloader))] for _ in range(3)]

    assert CountingInfiniteSampler.iterator_count == 1
    assert epochs == [[[0, 1], [2, 3]]] * 3


def test_hold_one_sample_dataloader_deepcopies() -> None:
    it = iter(HoldOnesampleDataLoader([{"x": [1]}, {"x": [2]}]))
    first = next(it)
    second = next(it)

    assert first == second == {"x": [1]}
    assert first is not second
    assert first["x"] is not second["x"]
