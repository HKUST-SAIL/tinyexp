from __future__ import annotations

import pytest

from tinyexp.tiny_engine.accelerator import HFAccelerator


def test_hf_accelerator_aliases_accelerate_indices() -> None:
    accelerator = HFAccelerator(cpu=True)

    assert accelerator.rank == accelerator.process_index
    assert accelerator.world_size == accelerator.num_processes
    assert accelerator.local_rank == accelerator.local_process_index
    accelerator.destroy()


def test_hf_accelerator_set_attributes_refuses_to_shadow_existing_names() -> None:
    accelerator = HFAccelerator(cpu=True)
    accelerator.rank = 0  # simulate a future accelerate version defining the name

    with pytest.raises(AttributeError, match="already defines"):
        accelerator._set_attributes()
    accelerator.destroy()
