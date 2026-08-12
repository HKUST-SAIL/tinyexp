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

        def unwrap_model(self, module):
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


def test_mnist_evaluate_reduces_correct_and_seen_counts_globally(tmp_path) -> None:
    exp = Exp(output_root=str(tmp_path), exp_name="mnist_metric_test")

    class FixedModule(torch.nn.Module):
        def forward(self, features):  # type: ignore[no-untyped-def]
            return torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=features.device)

    class UnevenDistributedAccelerator:
        device = "cpu"
        is_main_process = True
        reduce_calls = 0

        def unwrap_model(self, module):
            return module

        def reduce_sum(self, tensor):  # type: ignore[no-untyped-def]
            remote_count = (2, 3)[self.reduce_calls]
            self.reduce_calls += 1
            return tensor + remote_count

        def wait_for_everyone(self) -> None:
            pass

    class DummyDataLoader(list):
        def __init__(self):
            super().__init__([(torch.zeros(2, 1), torch.tensor([0, 0]))])
            self.dataset = [0, 1]

    accelerator = UnevenDistributedAccelerator()
    metric = exp._evaluate(
        accelerator=accelerator,
        logger=SimpleNamespace(info=lambda *args, **kwargs: None),
        module_or_module_path=FixedModule(),
        val_dataloader=DummyDataLoader(),
    )

    assert metric == 3 / 5
    assert accelerator.reduce_calls == 2


def test_mnist_evaluate_empty_dataset_returns_zero(tmp_path) -> None:
    exp = Exp(output_root=str(tmp_path), exp_name="mnist_empty_metric_test")

    class EmptyAccelerator:
        device = "cpu"

        def unwrap_model(self, module):
            return module

        def reduce_sum(self, tensor):  # type: ignore[no-untyped-def]
            return tensor

        def wait_for_everyone(self) -> None:
            pass

    class EmptyDataLoader(list):
        def __init__(self):
            super().__init__()
            self.dataset: list[int] = []

    metric = exp._evaluate(
        accelerator=EmptyAccelerator(),
        logger=SimpleNamespace(info=lambda *args, **kwargs: None),
        module_or_module_path=torch.nn.Identity(),
        val_dataloader=EmptyDataLoader(),
    )

    assert metric == 0.0
