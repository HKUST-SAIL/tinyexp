from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tinyexp.examples.mnist_exp import Exp
from tinyexp.tiny_engine.accelerator import CPUAccelerator

_LOGGER = SimpleNamespace(info=lambda *args, **kwargs: None)


class _EvalAccelerator:
    device = "cpu"
    rank = 0
    world_size = 1
    is_main_process = True

    def prepare(self, module):  # type: ignore[no-untyped-def]
        return module

    def unwrap_model(self, module):  # type: ignore[no-untyped-def]
        return module

    def reduce(self, tensor, reduction="sum", scale=1.0):  # type: ignore[no-untyped-def]
        return tensor

    def wait_for_everyone(self) -> None:
        pass


class _Loader(list):
    def __init__(self, batches: list, dataset: list) -> None:
        super().__init__(batches)
        self.dataset = dataset


def _evaluate(tmp_path: Path, module_or_module_path, accelerator, val_dataloader, trackers=()) -> float:  # type: ignore[no-untyped-def]
    exp = Exp(output_root=str(tmp_path), exp_name="mnist_test")
    return exp._evaluate(
        accelerator=accelerator,
        logger=_LOGGER,
        module_or_module_path=module_or_module_path,
        val_dataloader=val_dataloader,
        trackers=trackers,
    )


def test_mnist_dataset_download_is_owned_by_rank_zero(tmp_path, monkeypatch) -> None:
    calls = []
    dataset = torch.utils.data.TensorDataset(torch.arange(2), torch.arange(2))

    def fake_mnist(*args, **kwargs):
        calls.append(kwargs["download"])
        return dataset

    monkeypatch.setattr("tinyexp.examples.mnist_exp.datasets.MNIST", fake_mnist)
    cfg = Exp.DataloaderCfg(data_root=str(tmp_path))
    transform = object()

    class DummyAccelerator:
        def __init__(self, rank: int) -> None:
            self.rank = rank
            self.barrier_calls = 0

        def wait_for_everyone(self) -> None:
            self.barrier_calls += 1

    rank_zero = DummyAccelerator(rank=0)
    rank_one = DummyAccelerator(rank=1)
    cfg._build_mnist_dataset(rank_zero, train=True, transform=transform)
    cfg._build_mnist_dataset(rank_one, train=True, transform=transform)

    assert calls == [True, False]
    assert rank_zero.barrier_calls == 1
    assert rank_one.barrier_calls == 1


def test_mnist_val_dataloader_partitions_without_padding(tmp_path: Path, monkeypatch) -> None:
    dataset = torch.utils.data.TensorDataset(torch.arange(10), torch.arange(10))
    monkeypatch.setattr("tinyexp.examples.mnist_exp.datasets.MNIST", lambda *args, **kwargs: dataset)

    cfg = Exp.DataloaderCfg(data_root=str(tmp_path), train_batch_size_per_device=4, val_data_worker_per_gpu=0)
    partitions = []
    for rank in range(3):
        accelerator = SimpleNamespace(rank=rank, world_size=3, wait_for_everyone=lambda: None)
        dataloader = cfg.build_val_dataloader(accelerator)
        assert dataloader.drop_last is False
        partitions.append(list(dataloader.dataset.indices))

    assert partitions == [[0, 3, 6, 9], [1, 4, 7], [2, 5, 8]]


def test_mnist_evaluate_loads_model_state_from_checkpoint(tmp_path) -> None:
    exp = Exp(output_root=str(tmp_path), exp_name="mnist_test")
    checkpoint_path = exp.checkpoint_cfg.save_checkpoint(
        run_dir=str(tmp_path),
        name="demo.ckpt",
        model=exp.module_cfg.build_module(),
        global_step=7,
        exp_name=exp.exp_name,
        exp_class=exp.exp_class,
    )
    val_dataloader = _Loader([(torch.zeros(1, 1, 28, 28), torch.zeros(1, dtype=torch.long))], dataset=[0])
    logged = []
    tracker = SimpleNamespace(log=lambda metrics, step: logged.append((metrics, step)))

    assert isinstance(_evaluate(tmp_path, checkpoint_path, _EvalAccelerator(), val_dataloader, [tracker]), float)
    assert [step for _, step in logged] == [7]


def test_mnist_evaluate_reduces_correct_and_seen_counts_globally(tmp_path) -> None:
    class FixedModule(torch.nn.Module):
        def forward(self, features):  # type: ignore[no-untyped-def]
            return torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=features.device)

    class UnevenDistributedAccelerator(_EvalAccelerator):
        reduce_calls = 0

        def reduce(self, tensor, reduction="sum", scale=1.0):  # type: ignore[no-untyped-def]
            remote_count = (2, 3)[self.reduce_calls]  # remote correct, then remote seen
            self.reduce_calls += 1
            return tensor + remote_count

    accelerator = UnevenDistributedAccelerator()
    val_dataloader = _Loader([(torch.zeros(2, 1), torch.tensor([0, 0]))], dataset=[0, 1])

    logged = []
    tracker = SimpleNamespace(log=lambda metrics, step: logged.append((metrics, step)))
    assert _evaluate(tmp_path, FixedModule(), accelerator, val_dataloader, trackers=[tracker]) == 3 / 5
    assert logged == [({"val_metric": 3 / 5}, None)]
    assert accelerator.reduce_calls == 2


def test_mnist_evaluate_empty_dataset_returns_zero(tmp_path) -> None:
    assert _evaluate(tmp_path, torch.nn.Identity(), _EvalAccelerator(), _Loader([], dataset=[])) == 0.0


@pytest.mark.parametrize(
    "batches_per_epoch, train_steps",
    [(2, [2, 4, 6]), (20, [20, 40, 60]), (21, [20, 21, 40, 42, 60, 63])],
)
def test_mnist_train_logs_metrics_to_all_trackers(tmp_path: Path, monkeypatch, batches_per_epoch, train_steps) -> None:
    exp = Exp(output_root=str(tmp_path), launcher="mp")
    loader = _Loader(
        [(torch.ones(1, 2), torch.tensor([0]))] * batches_per_epoch,
        dataset=[0] * batches_per_epoch,
    )
    loader.batch_size = 1
    monkeypatch.setattr(exp.dataloader_cfg, "build_train_dataloader", lambda accelerator: loader)
    monkeypatch.setattr(exp.dataloader_cfg, "build_val_dataloader", lambda accelerator: loader)
    monkeypatch.setattr(exp.module_cfg, "build_module", lambda: torch.nn.Linear(2, 2))
    logged = [[], []]
    trackers = [
        SimpleNamespace(log=lambda metrics, step, entries=entries: entries.append({**metrics, "step": step}))
        for entries in logged
    ]
    accelerator = CPUAccelerator()
    try:
        exp._train(accelerator=accelerator, logger=_LOGGER, run_dir=str(tmp_path), trackers=trackers)
    finally:
        accelerator.destroy()

    assert logged[0] == logged[1]
    train_entries = [entry for entry in logged[0] if "loss" in entry]
    assert [entry["step"] for entry in train_entries] == train_steps
    assert all({"epoch", "loss", "lr", "step"} == entry.keys() for entry in train_entries)
    assert [entry["epoch"] for entry in train_entries] == [(step - 1) // batches_per_epoch for step in train_steps]
    assert [entry["step"] for entry in logged[0] if "val_metric" in entry] == [
        batches_per_epoch * epoch for epoch in (1, 2, 3)
    ]
