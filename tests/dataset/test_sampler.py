from __future__ import annotations

from types import SimpleNamespace

from tinyexp.dataset.fake_dataloader import HoldOnesampleDataLoader
from tinyexp.dataset.sampler import InfiniteSampler


def test_hold_one_sample_dataloader_deepcopies() -> None:
    data = [{"x": [1]}, {"x": [2]}]
    dl = HoldOnesampleDataLoader(data)

    it = iter(dl)
    first = next(it)
    second = next(it)

    assert first == {"x": [1]}
    assert second == {"x": [1]}
    assert first is not second
    assert first["x"] is not second["x"]


def test_infinite_sampler_no_shuffle_single_worker() -> None:
    sampler = InfiniteSampler(size=3, shuffle=False, seed=0)
    it = iter(sampler)
    first_seven = [next(it) for _ in range(7)]
    assert first_seven == [0, 1, 2, 0, 1, 2, 0]


def test_infinite_sampler_no_shuffle_multi_worker_slices() -> None:
    accelerator = SimpleNamespace(rank=1, world_size=2)
    sampler = InfiniteSampler(size=4, shuffle=False, seed=0, accelerator=accelerator)
    it = iter(sampler)
    first_six = [next(it) for _ in range(6)]
    assert first_six == [1, 3, 1, 3, 1, 3]


def test_infinite_sampler_len_respects_drop_last() -> None:
    accelerator = SimpleNamespace(rank=0, world_size=2)
    assert len(InfiniteSampler(size=5, shuffle=False, seed=0, accelerator=accelerator, drop_last=False)) == 3
    assert len(InfiniteSampler(size=5, shuffle=False, seed=0, accelerator=accelerator, drop_last=True)) == 2
