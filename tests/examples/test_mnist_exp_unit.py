from __future__ import annotations

from types import SimpleNamespace

import torch

from tinyexp.examples.mnist_exp import Exp


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
