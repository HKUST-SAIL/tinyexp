"""DDPAccelerator and FSDPAccelerator share their single-process, mixed-precision and clipping behavior.

CUDA is faked (``torch.cuda.is_available`` etc.), so these run on CPU; the multi-process paths run on GPU jobs.
"""

from __future__ import annotations

import pytest
import torch

from tinyexp.exceptions import CudaNotAvailableError
from tinyexp.tiny_engine.accelerator import DDPAccelerator, FSDPAccelerator

ACCELERATORS = pytest.mark.parametrize("accelerator_cls", [DDPAccelerator, FSDPAccelerator])


@pytest.fixture
def fake_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")


def _cpu_model(monkeypatch: pytest.MonkeyPatch) -> torch.nn.Linear:
    """A model whose ``.to(cuda)`` is a no-op, so prepare_model works on a GPU-less host."""
    model = torch.nn.Linear(4, 2)
    monkeypatch.setattr(model, "to", lambda device: model)
    return model


def _clippable_parameter() -> torch.nn.Parameter:
    parameter = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    parameter.grad = torch.tensor([3.0, 4.0])
    return parameter


@ACCELERATORS
def test_requires_cuda(accelerator_cls: type) -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA is available in this environment")

    with pytest.raises(CudaNotAvailableError):
        accelerator_cls()


@ACCELERATORS
def test_single_process_skips_process_group_and_wrapper(
    accelerator_cls: type, fake_cuda: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    accelerator = accelerator_cls()
    model = _cpu_model(monkeypatch)

    assert accelerator.device == torch.device("cuda", 0)
    assert accelerator._process_group_initialized is False
    assert accelerator.prepare_model(model) is model
    # Full precision leaves forward alone; the wrapper is only installed for bf16/fp16.
    assert not hasattr(model, "_original_forward")
    assert accelerator.unwrap_model(model) is model
    accelerator.destroy()


@ACCELERATORS
@pytest.mark.parametrize(
    ("mixed_precision", "amp_dtype", "scaler_enabled"),
    [("none", None, False), ("bf16", torch.bfloat16, False), ("fp16", torch.float16, True)],
)
def test_mixed_precision_selects_dtype_and_scaler(
    accelerator_cls: type, fake_cuda: None, mixed_precision: str, amp_dtype: torch.dtype | None, scaler_enabled: bool
) -> None:
    accelerator = accelerator_cls(mixed_precision=mixed_precision)

    assert accelerator._amp_dtype == amp_dtype
    # An enabled scaler would allocate its scale on CUDA, so only its state is
    # asserted here; the scaled path is exercised by GPU jobs.
    assert accelerator._scaler.is_enabled() is scaler_enabled
    accelerator.destroy()


@ACCELERATORS
def test_rejects_unknown_mixed_precision(accelerator_cls: type, fake_cuda: None) -> None:
    with pytest.raises(ValueError, match="mixed precision"):
        accelerator_cls(mixed_precision="fp8")


@ACCELERATORS
@pytest.mark.parametrize("mixed_precision", ["none", "bf16"])
def test_unscaled_train_step_runs_on_cpu_tensors(accelerator_cls: type, fake_cuda: None, mixed_precision: str) -> None:
    """A disabled scaler passes tensors through, so backward/optimizer_step run on CPU tensors.

    The real cuda autocast of bf16 cannot be entered on a GPU-less host, so it is only entered for "none".
    """
    accelerator = accelerator_cls(mixed_precision=mixed_precision)
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    weight_before = model.weight.detach().clone()

    def loss_fn() -> torch.Tensor:
        return torch.nn.functional.cross_entropy(model(torch.randn(8, 4)), torch.zeros(8, dtype=torch.long))

    if mixed_precision == "none":
        with accelerator.autocast():
            loss = loss_fn()
    else:
        loss = loss_fn()
    accelerator.backward(loss)
    accelerator.optimizer_step(optimizer)

    assert (model.weight.detach() - weight_before).abs().max() > 0
    accelerator.destroy()


@ACCELERATORS
def test_bf16_prepare_model_wraps_forward(
    accelerator_cls: type, fake_cuda: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Loops that never open autocast() themselves must still get mixed precision."""
    accelerator = accelerator_cls(mixed_precision="bf16")
    model = _cpu_model(monkeypatch)

    assert accelerator.prepare_model(model) is model
    assert hasattr(model, "_original_forward")
    accelerator.destroy()


@ACCELERATORS
def test_bf16_clip_grad_norm_clips_without_unscaling(
    accelerator_cls: type, fake_cuda: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without sharding (world 1) both clip with torch's clip, after the shared unscale preflight."""
    accelerator = accelerator_cls(mixed_precision="bf16")
    parameter = _clippable_parameter()
    accelerator.prepare_optimizer(torch.optim.SGD([parameter], lr=0.1))
    unscaled: list[object] = []
    monkeypatch.setattr(accelerator._scaler, "unscale_", unscaled.append)

    total_norm = accelerator.clip_grad_norm_([parameter], 1.0)

    # bf16 disables the scaler, so there is nothing to unscale.
    assert unscaled == []
    # The returned norm is measured before clipping; the gradients are clipped in place.
    assert total_norm.item() == pytest.approx(5.0)
    assert parameter.grad.norm().item() == pytest.approx(1.0)
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


@ACCELERATORS
def test_fp16_clip_grad_norm_unscales_prepared_optimizers_first(
    accelerator_cls: type, fake_cuda: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clipping scaled fp16 gradients would compare an inflated norm against max_norm."""
    accelerator = accelerator_cls(mixed_precision="fp16")
    parameter = _clippable_parameter()
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    accelerator.prepare_optimizer(optimizer)
    scaler = _RecordingScaler()
    monkeypatch.setattr(accelerator, "_scaler", scaler)

    accelerator.clip_grad_norm_([parameter], 1.0)

    assert scaler.unscaled == [optimizer]
    # Unscaling ran before the norm was measured, so clipping actually applied.
    assert parameter.grad.norm().item() == pytest.approx(1.0)
    accelerator.destroy()


@ACCELERATORS
def test_fp16_clip_grad_norm_rejects_unprepared_optimizer_parameters(accelerator_cls: type, fake_cuda: None) -> None:
    """fp16 grads of an optimizer that skipped prepare_optimizer still carry the scale.

    That includes the mixed case: one prepared optimizer does not make another
    optimizer's gradients safe to clip.
    """
    accelerator = accelerator_cls(mixed_precision="fp16")
    accelerator.prepare_optimizer(torch.optim.SGD([torch.nn.Parameter(torch.randn(2))], lr=0.1))

    with pytest.raises(RuntimeError, match="never went through prepare_optimizer"):
        accelerator.clip_grad_norm_([_clippable_parameter()], 1.0)
    accelerator.destroy()


def test_ddp_gather_and_state_dict_in_single_process(fake_cuda: None) -> None:
    accelerator = DDPAccelerator()
    tensor = torch.tensor([1.0, 2.0])

    assert accelerator.gather(tensor) is tensor
    state = accelerator.dump_model_to_state_dict(torch.nn.Sequential(torch.nn.Linear(2, 1)))
    assert list(state) == ["0.weight", "0.bias"]
    assert all(value.device.type == "cpu" for value in state.values())
    accelerator.destroy()
