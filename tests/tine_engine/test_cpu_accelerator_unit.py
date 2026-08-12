from __future__ import annotations

import pytest
import torch

from tinyexp.tiny_engine.accelerator import CPUAccelerator


def test_cpu_accelerator_reduce_ops_world_size_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    acc = CPUAccelerator()

    tensor = torch.tensor([1.0], dtype=torch.float32)
    assert torch.equal(acc.reduce_sum(tensor), tensor)
    assert torch.equal(acc.reduce_mean(tensor), tensor)

    # No-op for single worker.
    acc.wait_for_everyone()


def test_cpu_accelerator_forces_cpu_even_when_cuda_is_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setenv("WORLD_SIZE", "1")

    accelerator = CPUAccelerator()

    assert accelerator.device == torch.device("cpu")


def test_cpu_accelerator_destroy_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "1")
    accelerator = CPUAccelerator()

    accelerator.destroy()
    accelerator.destroy()

    assert accelerator._destroyed is True
    assert accelerator._process_group_initialized is False


def test_cpu_accelerator_destructor_is_a_cleanup_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "1")
    destroyed = []
    accelerator = CPUAccelerator()
    monkeypatch.setattr(accelerator, "destroy", lambda: destroyed.append(True))

    accelerator.__del__()

    assert destroyed == [True]


def test_cpu_accelerator_destroy_does_not_mark_failed_cleanup_as_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "1")
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


def test_cpu_accelerator_print_only_rank0(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("WORLD_SIZE", "1")
