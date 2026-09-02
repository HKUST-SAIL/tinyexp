from __future__ import annotations

import pytest
import torch

from tinyexp.exceptions import CudaNotAvailableError
from tinyexp.tiny_engine.accelerator import DDPAccelerator


def test_ddp_accelerator_requires_cuda() -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA is available in this environment")

    with pytest.raises(CudaNotAvailableError):
        DDPAccelerator()


def test_ddp_accelerator_single_process_skips_process_group_and_ddp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")

    accelerator = DDPAccelerator()
    model = torch.nn.Linear(2, 1)
    monkeypatch.setattr(model, "to", lambda device: model)

    assert accelerator.device == torch.device("cuda", 0)
    assert accelerator._process_group_initialized is False
    assert accelerator.prepare_model(model) is model
    accelerator.destroy()
