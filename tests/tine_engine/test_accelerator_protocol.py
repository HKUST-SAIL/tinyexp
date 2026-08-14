from __future__ import annotations

import pytest
import torch

from tinyexp.tiny_engine.accelerator import AcceleratorProtocol, CPUAccelerator


def test_cpu_accelerator_implements_accelerator_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "1")

    accelerator = CPUAccelerator()

    assert isinstance(accelerator, AcceleratorProtocol)


def test_hf_accelerator_implements_accelerator_protocol() -> None:
    pytest.importorskip("accelerate")
    from tinyexp.tiny_engine.accelerator import HFAccelerator

    accelerator = HFAccelerator(cpu=True)

    assert isinstance(accelerator, AcceleratorProtocol)
    assert torch.equal(accelerator.reduce_sum(torch.tensor([2.0])), torch.tensor([2.0]))
    assert torch.equal(accelerator.reduce_mean(torch.tensor([2.0])), torch.tensor([2.0]))
    accelerator.destroy()
