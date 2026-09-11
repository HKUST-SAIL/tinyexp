"""Tests for the ported RASampler (docs/vit_tp.md).

The oracle recomputes the official index math from facebookresearch/deit ``samplers.py``
inline, so any accidental change to the ported sampling logic shows up as a mismatch.
"""

from __future__ import annotations

import math

import pytest
import torch

from tinyexp.dataset.ra_sampler import RASampler


class _LenOnlyDataset:
    def __init__(self, length: int) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length


def _official_indices(dataset_len: int, num_replicas: int, rank: int, epoch: int, num_repeats: int) -> list[int]:
    """Verbatim re-implementation of the official RASampler iteration math."""
    g = torch.Generator()
    g.manual_seed(epoch)
    indices = torch.randperm(dataset_len, generator=g)
    indices = torch.repeat_interleave(indices, repeats=num_repeats, dim=0).tolist()
    num_samples = int(math.ceil(dataset_len * num_repeats / num_replicas))
    total_size = num_samples * num_replicas
    padding_size = total_size - len(indices)
    if padding_size > 0:
        indices += indices[:padding_size]
    indices = indices[rank:total_size:num_replicas]
    num_selected = int(math.floor(dataset_len // 256 * 256 / num_replicas))
    return indices[:num_selected]


@pytest.mark.parametrize("num_repeats", [1, 3])
@pytest.mark.parametrize("rank", [0, 1])
def test_matches_official_index_math(num_repeats: int, rank: int) -> None:
    dataset = _LenOnlyDataset(1000)
    sampler = RASampler(dataset, num_replicas=2, rank=rank, shuffle=True, num_repeats=num_repeats)
    sampler.set_epoch(7)

    assert list(sampler) == _official_indices(1000, 2, rank, epoch=7, num_repeats=num_repeats)
    assert len(sampler) == 384  # floor(1000 // 256 * 256 / 2)


def test_epoch_changes_permutation_deterministically() -> None:
    dataset = _LenOnlyDataset(1000)
    sampler = RASampler(dataset, num_replicas=2, rank=0)
    first = list(sampler)

    sampler.set_epoch(0)
    assert list(sampler) == first  # default epoch is 0
    sampler.set_epoch(1)
    assert list(sampler) != first
    sampler.set_epoch(0)
    assert list(sampler) == first


def test_no_index_exceeds_num_repeats_copies() -> None:
    dataset = _LenOnlyDataset(1000)
    rank0 = RASampler(dataset, num_replicas=2, rank=0, num_repeats=3)
    rank1 = RASampler(dataset, num_replicas=2, rank=1, num_repeats=3)
    rank0.set_epoch(2)
    rank1.set_epoch(2)

    counts: dict[int, int] = {}
    for index in [*rank0, *rank1]:
        counts[index] = counts.get(index, 0) + 1
    assert counts  # sanity: the merged stream is non-empty
    assert max(counts.values()) <= 3


def test_num_repeats_must_be_positive() -> None:
    with pytest.raises(ValueError):
        RASampler(_LenOnlyDataset(10), num_replicas=2, rank=0, num_repeats=0)
