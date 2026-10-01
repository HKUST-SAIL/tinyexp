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


def test_fsdp_accelerator_clip_grad_norm_world_one_clips_prepared_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without sharding (world 1) clip falls back to torch's, after the shared preflight."""
    _fake_cuda_single_process(monkeypatch)
    accelerator = FSDPAccelerator(mixed_precision="bf16")
    parameter = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    parameter.grad = torch.tensor([3.0, 4.0])
    accelerator.prepare_optimizer(torch.optim.SGD([parameter], lr=0.1))

    total_norm = accelerator.clip_grad_norm_([parameter], 1.0)

    # The returned norm is measured before clipping; the gradients are clipped in place.
    assert total_norm.item() == pytest.approx(5.0)
    assert parameter.grad.norm().item() == pytest.approx(1.0)
    accelerator.destroy()


def test_fsdp_accelerator_clip_grad_norm_rejects_unprepared_optimizer_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """fp16 grads of an optimizer that skipped prepare_optimizer still carry the scale."""
    _fake_cuda_single_process(monkeypatch)
    accelerator = FSDPAccelerator(mixed_precision="fp16")
    orphan = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    orphan.grad = torch.tensor([3.0, 4.0])

    with pytest.raises(RuntimeError, match="never went through prepare_optimizer"):
        accelerator.clip_grad_norm_([orphan], 1.0)
    accelerator.destroy()
