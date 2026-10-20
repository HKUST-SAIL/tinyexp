from __future__ import annotations

import pytest
import torch

pytest.importorskip("accelerate")

from tinyexp.tiny_engine.accelerator import AcceleratorProtocol, HFAccelerator


def test_hf_accelerator_implements_protocol_and_aliases_accelerate_indices() -> None:
    accelerator = HFAccelerator(cpu=True)

    assert isinstance(accelerator, AcceleratorProtocol)
    assert accelerator.rank == accelerator.process_index
    assert accelerator.world_size == accelerator.num_processes
    assert accelerator.local_rank == accelerator.local_process_index
    assert torch.equal(accelerator.reduce(torch.tensor([2.0]), reduction="sum"), torch.tensor([2.0]))
    assert torch.equal(accelerator.reduce(torch.tensor([2.0]), reduction="mean"), torch.tensor([2.0]))
    accelerator.destroy()


def test_hf_accelerator_set_attributes_refuses_to_shadow_existing_names() -> None:
    accelerator = HFAccelerator(cpu=True)
    accelerator.rank = 0  # simulate a future accelerate version defining the name

    with pytest.raises(AttributeError, match="already defines"):
        accelerator._set_attributes()
    accelerator.destroy()
