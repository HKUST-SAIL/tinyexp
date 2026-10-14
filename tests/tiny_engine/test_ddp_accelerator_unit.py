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


def test_ddp_accelerator_gather_world_size_one_returns_input(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_cuda_single_process(monkeypatch)
    accelerator = DDPAccelerator()
    tensor = torch.tensor([1.0, 2.0])

    assert accelerator.gather(tensor) is tensor
    accelerator.destroy()


def test_ddp_accelerator_dump_model_to_state_dict_keeps_plain_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_cuda_single_process(monkeypatch)
    accelerator = DDPAccelerator()
    model = torch.nn.Sequential(torch.nn.Linear(2, 1))

    state = accelerator.dump_model_to_state_dict(model)

    assert list(state) == ["0.weight", "0.bias"]
    assert all(value.device.type == "cpu" for value in state.values())
    accelerator.destroy()


def _fake_cuda_single_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")


def _train_one_step(accelerator: DDPAccelerator, *, enter_autocast: bool = True) -> torch.Tensor:
    """Run one full step through autocast/backward/optimizer_step; return the weight delta.

    ``enter_autocast=False`` skips the context for real cuda autocasts, which
    cannot be entered on a GPU-less host.
    """
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    weight_before = model.weight.detach().clone()
    images = torch.randn(8, 4)
    labels = torch.zeros(8, dtype=torch.long)

    if enter_autocast:
        with accelerator.autocast():
            loss = torch.nn.functional.cross_entropy(model(images), labels)
    else:
        loss = torch.nn.functional.cross_entropy(model(images), labels)
    accelerator.backward(loss)
    accelerator.optimizer_step(optimizer)

    return (model.weight.detach() - weight_before).abs().max()


def test_ddp_accelerator_rejects_unknown_mixed_precision(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_cuda_single_process(monkeypatch)

    with pytest.raises(ValueError, match="mixed precision"):
        DDPAccelerator(mixed_precision="fp8")


def test_ddp_accelerator_default_keeps_full_precision(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_cuda_single_process(monkeypatch)

    accelerator = DDPAccelerator()

    assert accelerator._amp_dtype is None
    assert not accelerator._scaler.is_enabled()
    # The disabled scaler passes tensors through, so a full step runs in fp32 on CPU tensors.
    assert _train_one_step(accelerator) > 0
    accelerator.destroy()


def test_ddp_accelerator_bf16_selects_dtype_and_disables_scaler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_cuda_single_process(monkeypatch)

    accelerator = DDPAccelerator(mixed_precision="bf16")

    assert accelerator._amp_dtype == torch.bfloat16
    assert not accelerator._scaler.is_enabled()
    # bf16 needs no loss scaling, so the CPU-side step path is identical;
    # the real cuda autocast itself can only be exercised on a GPU host.
    assert _train_one_step(accelerator, enter_autocast=False) > 0
    accelerator.destroy()


def test_ddp_accelerator_fp16_enables_grad_scaler(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_cuda_single_process(monkeypatch)

    accelerator = DDPAccelerator(mixed_precision="fp16")

    assert accelerator._amp_dtype == torch.float16
    # An enabled scaler would allocate its scale on CUDA, so only its state is
    # asserted here; the scaled path is exercised by GPU jobs.
    assert accelerator._scaler.is_enabled()
    accelerator.destroy()


def test_ddp_accelerator_bf16_prepare_model_wraps_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    """Loops that never open autocast() themselves must still get mixed precision."""
    _fake_cuda_single_process(monkeypatch)
    accelerator = DDPAccelerator(mixed_precision="bf16")
    model = torch.nn.Linear(4, 2)
    monkeypatch.setattr(model, "to", lambda device: model)

    prepared = accelerator.prepare_model(model)

    assert prepared is model
    assert hasattr(model, "_original_forward")
    accelerator.destroy()


def test_ddp_accelerator_full_precision_prepare_model_keeps_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_cuda_single_process(monkeypatch)
    accelerator = DDPAccelerator()
    model = torch.nn.Linear(4, 2)
    monkeypatch.setattr(model, "to", lambda device: model)

    assert not hasattr(accelerator.prepare_model(model), "_original_forward")
    accelerator.destroy()


class _RecordingScaler:
    """Stub for an enabled GradScaler, whose scale tensor lives on CUDA."""

    def __init__(self) -> None:
        self.unscaled: list[object] = []

    @staticmethod
    def is_enabled() -> bool:
        return True

    def unscale_(self, opt: object) -> None:
        self.unscaled.append(opt)


def test_ddp_accelerator_clip_grad_norm_unscales_prepared_optimizers_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clipping scaled fp16 gradients would compare an inflated norm against max_norm."""
    _fake_cuda_single_process(monkeypatch)
    accelerator = DDPAccelerator(mixed_precision="fp16")
    parameter = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    parameter.grad = torch.tensor([3.0, 4.0])
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    accelerator.prepare_optimizer(optimizer)
    scaler = _RecordingScaler()
    monkeypatch.setattr(accelerator, "_scaler", scaler)

    accelerator.clip_grad_norm_([parameter], 1.0)

    assert scaler.unscaled == [optimizer]
    # Unscaling ran before the norm was measured, so clipping actually applied.
    assert parameter.grad.norm().item() == pytest.approx(1.0)
    accelerator.destroy()


def test_ddp_accelerator_clip_grad_norm_rejects_unprepared_optimizer_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """fp16 grads of an optimizer that skipped prepare_optimizer still carry the scale.

    That includes the mixed case: one prepared optimizer does not make another
    optimizer's gradients safe to clip.
    """
    _fake_cuda_single_process(monkeypatch)
    accelerator = DDPAccelerator(mixed_precision="fp16")
    accelerator.prepare_optimizer(torch.optim.SGD([torch.nn.Parameter(torch.randn(2))], lr=0.1))
    orphan = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    orphan.grad = torch.tensor([3.0, 4.0])

    with pytest.raises(RuntimeError, match="never went through prepare_optimizer"):
        accelerator.clip_grad_norm_([orphan], 1.0)
    accelerator.destroy()


def test_ddp_accelerator_clip_grad_norm_does_not_unscale_in_bf16(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_cuda_single_process(monkeypatch)
    accelerator = DDPAccelerator(mixed_precision="bf16")
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.randn(2))], lr=0.1)
    accelerator.prepare_optimizer(optimizer)

    unscaled: list[object] = []
    monkeypatch.setattr(accelerator._scaler, "unscale_", unscaled.append)
    parameter = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    parameter.grad = torch.tensor([3.0, 4.0])

    accelerator.clip_grad_norm_([parameter], 1.0)

    # bf16 disables the scaler, so there is nothing to unscale.
    assert unscaled == []
    assert parameter.grad.norm().item() == pytest.approx(1.0)
    accelerator.destroy()
