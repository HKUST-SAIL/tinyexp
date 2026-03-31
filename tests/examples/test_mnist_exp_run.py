from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tinyexp.examples.mnist_exp import Exp


def test_mnist_run_val_mode_requires_resume_from(tmp_path: Path, monkeypatch) -> None:
    exp = Exp(output_root=str(tmp_path), exp_name="mnist_val", mode="val", resume_from="")

    dummy_accelerator = SimpleNamespace(rank=0, device="cpu", is_main_process=True)
    dummy_logger = SimpleNamespace(info=lambda *args, **kwargs: None)

    monkeypatch.setattr(exp.accelerator_cfg, "build_accelerator", lambda: dummy_accelerator)
    monkeypatch.setattr(exp.logger_cfg, "build_logger", lambda **kwargs: dummy_logger)

    with pytest.raises(ValueError, match="resume_from"):
        exp.run()


def test_mnist_run_val_mode_uses_checkpoint(tmp_path: Path, monkeypatch) -> None:
    exp_for_ckpt = Exp(output_root=str(tmp_path), exp_name="mnist_val")
    checkpoint_path = exp_for_ckpt.checkpoint_cfg.save_checkpoint(
        run_dir=str(tmp_path / "mnist_val"),
        name="demo.ckpt",
        model=exp_for_ckpt.module_cfg.build_module(),
        exp_name=exp_for_ckpt.exp_name,
        exp_class=exp_for_ckpt.exp_class,
    )

    exp = Exp(output_root=str(tmp_path), exp_name="mnist_val", mode="val", resume_from=checkpoint_path)

    dummy_accelerator = SimpleNamespace(rank=0, device="cpu", is_main_process=True)
    dummy_logger = SimpleNamespace(info=lambda *args, **kwargs: None)

    monkeypatch.setattr(exp.accelerator_cfg, "build_accelerator", lambda: dummy_accelerator)
    monkeypatch.setattr(exp.logger_cfg, "build_logger", lambda **kwargs: dummy_logger)

    called: dict[str, object] = {}

    def fake_evaluate(*, accelerator, logger, module_or_module_path, val_dataloader=None):
        called["accelerator"] = accelerator
        called["logger"] = logger
        called["module_or_module_path"] = module_or_module_path
        called["val_dataloader"] = val_dataloader
        return 0.5

    monkeypatch.setattr(exp, "_evaluate", fake_evaluate)

    exp.run()

    assert called["accelerator"] is dummy_accelerator
    assert called["logger"] is dummy_logger
    assert called["module_or_module_path"] == checkpoint_path
