from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

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


def test_mnist_evaluate_loads_model_state_from_checkpoint(tmp_path) -> None:
    exp = Exp(output_root=str(tmp_path), exp_name="mnist_test")
    checkpoint_path = exp.checkpoint_cfg.save_checkpoint(
        run_dir=str(tmp_path),
        name="demo.ckpt",
        model=exp.module_cfg.build_module(),
        exp_name=exp.exp_name,
        exp_class=exp.exp_class,
    )

    class DummyAccelerator:
        device = "cpu"
        rank = 0
        world_size = 1
        is_main_process = True

        def prepare(self, module):
            return module

        def reduce_sum(self, tensor):
            return tensor

        def wait_for_everyone(self) -> None:
            return None

    dummy_logger = SimpleNamespace(info=lambda *args, **kwargs: None)

    class DummyDataLoader(list):
        def __init__(self):
            super().__init__([(torch.zeros(1, 1, 28, 28), torch.zeros(1, dtype=torch.long))])
            self.dataset = [0]

    val_dataloader = DummyDataLoader()
    metric = exp._evaluate(
        accelerator=DummyAccelerator(),
        logger=dummy_logger,
        module_or_module_path=checkpoint_path,
        val_dataloader=val_dataloader,
    )

    assert isinstance(metric, float)
