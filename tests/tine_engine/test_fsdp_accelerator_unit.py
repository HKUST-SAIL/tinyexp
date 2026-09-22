from __future__ import annotations

import pytest
import torch

from tinyexp.exceptions import CudaNotAvailableError
from tinyexp.tiny_engine.accelerator import FSDPAccelerator


def test_fsdp_accelerator_requires_cuda() -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA is available in this environment")

    with pytest.raises(CudaNotAvailableError):
        FSDPAccelerator()


def test_fsdp_accelerator_single_process_skips_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")

    accelerator = FSDPAccelerator()
    model = torch.nn.Linear(2, 1)
    monkeypatch.setattr(model, "to", lambda device: model)

    assert accelerator.device == torch.device("cuda", 0)
    assert accelerator._process_group_initialized is False
    assert accelerator.prepare_model(model) is model
    accelerator.destroy()


def test_fsdp_unwrap_model_returns_plain_module_without_wrapper() -> None:
    accelerator = FSDPAccelerator.__new__(FSDPAccelerator)
    model = torch.nn.Linear(2, 1)

    assert accelerator.unwrap_model(model) is model


def _fake_cuda_single_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")


def test_fsdp_accelerator_rejects_unknown_mixed_precision(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_cuda_single_process(monkeypatch)

    with pytest.raises(ValueError, match="mixed precision"):
        FSDPAccelerator(mixed_precision="fp8")


def test_fsdp_accelerator_bf16_disables_scaler_and_builds_autocast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_cuda_single_process(monkeypatch)

    accelerator = FSDPAccelerator(mixed_precision="bf16")

    assert accelerator._amp_dtype == torch.bfloat16
    assert not accelerator._scaler.is_enabled()
    # autocast() builds the same torch.autocast as DDP's; a cuda autocast
    # cannot even be constructed on a GPU-less host, so that path is
    # exercised by GPU jobs.
    accelerator.destroy()


def test_fsdp_accelerator_fp16_enables_grad_scaler(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_cuda_single_process(monkeypatch)

    accelerator = FSDPAccelerator(mixed_precision="fp16")

    assert accelerator._amp_dtype == torch.float16
    # An enabled scaler would allocate its scale on CUDA, so only its state is
    # asserted here; the scaled path is exercised by GPU jobs.
    assert accelerator._scaler.is_enabled()
    accelerator.destroy()
