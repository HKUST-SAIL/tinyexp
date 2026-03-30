from __future__ import annotations

from types import SimpleNamespace

import pytest

from tinyexp.examples.mnist_exp import Exp


def test_mnist_val_mode_requires_resume_from(tmp_path) -> None:
    exp = Exp(output_root=str(tmp_path), exp_name="mnist_test", mode="val", resume_from="")

    dummy_accelerator = SimpleNamespace(rank=0, device="cpu", is_main_process=True)
    dummy_logger = SimpleNamespace(info=lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="resume_from"):
        exp._validate_from_checkpoint(accelerator=dummy_accelerator, logger=dummy_logger)


def test_mnist_validate_from_checkpoint_calls_evaluate(monkeypatch, tmp_path) -> None:
    exp = Exp(output_root=str(tmp_path), exp_name="mnist_test", mode="val", resume_from="demo.ckpt")

    called: dict[str, object] = {}

    def fake_evaluate(*, accelerator, logger, module_or_module_path, val_dataloader=None):
        called["accelerator"] = accelerator
        called["logger"] = logger
        called["module_or_module_path"] = module_or_module_path
        called["val_dataloader"] = val_dataloader
        return 0.5

    monkeypatch.setattr(exp, "_evaluate", fake_evaluate)

    dummy_accelerator = SimpleNamespace(rank=0, device="cpu", is_main_process=True)
    dummy_logger = SimpleNamespace(info=lambda *args, **kwargs: None)
    exp._validate_from_checkpoint(accelerator=dummy_accelerator, logger=dummy_logger)

    assert called["module_or_module_path"] == "demo.ckpt"
