from __future__ import annotations

from typing import NamedTuple

import pytest
import torch

from tinyexp.tiny_engine.accelerator import AcceleratorProtocol, CPUAccelerator


@pytest.fixture(autouse=True)
def _single_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")


def test_cpu_accelerator_world_size_one_is_a_local_protocol_implementation(monkeypatch: pytest.MonkeyPatch) -> None:
    # CPU is forced even when CUDA is visible.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    acc = CPUAccelerator()

    assert isinstance(acc, AcceleratorProtocol)
    assert acc.device == torch.device("cpu")
    tensor = torch.tensor([1.0], dtype=torch.float32)
    assert torch.equal(acc.reduce(tensor, reduction="sum"), tensor)
    assert torch.equal(acc.reduce(tensor, reduction="mean"), tensor)
    acc.wait_for_everyone()  # no-op for a single worker


def test_cpu_accelerator_full_precision_is_a_pass_through() -> None:
    acc = CPUAccelerator()

    model = acc.prepare_model(torch.nn.Linear(4, 2))
    assert not hasattr(model, "_original_forward")
    with acc.autocast():  # no-op
        assert model(torch.randn(3, 4)).dtype == torch.float32

    stepped = []

    class _Optimizer:
        def step(self):
            stepped.append(True)

    acc.optimizer_step(_Optimizer())
    assert stepped == [True]


def test_cpu_accelerator_rejects_fp16() -> None:
    with pytest.raises(ValueError, match="none/bf16"):
        CPUAccelerator(mixed_precision="fp16")


def test_cpu_accelerator_bf16_autocast_runs_a_full_train_step() -> None:
    acc = CPUAccelerator(mixed_precision="bf16")

    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    weight_before = model.weight.detach().clone()

    with acc.autocast():
        output = model(torch.randn(3, 4))
        assert output.dtype == torch.bfloat16
        loss = torch.nn.functional.cross_entropy(output, torch.zeros(3, dtype=torch.long))
    acc.backward(loss)
    acc.optimizer_step(optimizer)

    assert (model.weight.detach() - weight_before).abs().max() > 0


class _Output(NamedTuple):
    logits: torch.Tensor


def test_cpu_accelerator_bf16_prepared_model_autocasts_and_returns_fp32_outputs() -> None:
    """A loop written against accelerate never opens autocast() itself; prepare_model covers it."""
    acc = CPUAccelerator(mixed_precision="bf16")

    class _Model(torch.nn.Linear):
        def forward(self, images):  # type: ignore[override]
            hidden = super().forward(images)
            # What autocast actually computed, before the output cast applies.
            return {
                "dtype_inside": hidden.dtype,
                "nested": [hidden, {"pair": (hidden, hidden)}],
                "named": _Output(logits=hidden),
                "text": "not-a-tensor",
                "number": 7,
            }

    model = acc.prepare_model(_Model(4, 2))
    output = model(torch.randn(3, 4))

    assert output["dtype_inside"] == torch.bfloat16
    # Outputs come back fp32 at every nesting level, matching accelerate's convert_outputs_to_fp32.
    first, mapping = output["nested"]
    assert first.dtype == torch.float32
    assert [tensor.dtype for tensor in mapping["pair"]] == [torch.float32, torch.float32]
    assert isinstance(output["named"], _Output)
    assert output["named"].logits.dtype == torch.float32
    # Non-tensor leaves pass through untouched.
    assert output["text"] == "not-a-tensor"
    assert output["number"] == 7
    # Nesting the explicit context stays correct rather than double-casting.
    with acc.autocast():
        assert model(torch.randn(3, 4))["named"].logits.dtype == torch.float32
    # Preparing twice wraps forward only once.
    once = model.forward
    assert acc.prepare_model(model).forward is once


def test_cpu_accelerator_destroy_is_idempotent_and_the_destructor_falls_back_to_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accelerator = CPUAccelerator()

    accelerator.destroy()
    accelerator.destroy()

    assert accelerator._destroyed is True
    assert accelerator._process_group_initialized is False

    destroyed = []
    monkeypatch.setattr(accelerator, "destroy", lambda: destroyed.append(True))
    accelerator.__del__()
    assert destroyed == [True]


def test_cpu_accelerator_destroy_does_not_mark_failed_cleanup_as_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accelerator = CPUAccelerator()
    calls = []

    def fail_once() -> None:
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError

    monkeypatch.setattr(torch.distributed, "destroy_process_group", fail_once)
    accelerator._process_group_initialized = True
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    with pytest.raises(RuntimeError):
        accelerator.destroy()
    accelerator.destroy()

    assert calls == [True, True]


def test_cpu_accelerator_attaches_to_an_existing_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An already-initialized group must be reused, not initialized a second time."""
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    inits: list[object] = []
    monkeypatch.setattr(torch.distributed, "init_process_group", lambda *args, **kwargs: inits.append(args))

    accelerator = CPUAccelerator()

    assert inits == []
    # The accelerator does not own the group, so destroy leaves it alone.
    assert accelerator._process_group_initialized is False
    accelerator.destroy()
    assert torch.distributed.is_initialized()


@pytest.mark.parametrize(
    ("world_size", "rank", "expected"),
    [
        ("1", "0", "hello\n"),
        # A per-node main (LOCAL_RANK=0) on a non-first node must not print; print mirrors accelerate.
        ("2", "1", ""),
    ],
)
def test_cpu_accelerator_print_gates_on_main_process_not_local_main(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], world_size: str, rank: str, expected: str
) -> None:
    monkeypatch.setenv("WORLD_SIZE", world_size)
    monkeypatch.setenv("RANK", rank)
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    accelerator = CPUAccelerator()

    accelerator.print("hello")

    assert capsys.readouterr().out == expected
