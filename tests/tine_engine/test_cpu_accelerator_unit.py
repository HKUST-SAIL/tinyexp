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


def test_cpu_accelerator_print_only_rank0(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("WORLD_SIZE", "1")

    monkeypatch.setenv("RANK", "1")
    acc_rank1 = CPUAccelerator()
    acc_rank1.print("should not print")
    assert capsys.readouterr().out == ""

    monkeypatch.setenv("RANK", "0")
    acc_rank0 = CPUAccelerator()
    acc_rank0.print("hello")
    assert "hello" in capsys.readouterr().out
