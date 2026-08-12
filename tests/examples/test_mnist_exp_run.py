from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

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


def test_mnist_val_dataloader_partitions_without_padding(tmp_path: Path, monkeypatch) -> None:
    dataset = torch.utils.data.TensorDataset(torch.arange(10), torch.arange(10))
    monkeypatch.setattr("tinyexp.examples.mnist_exp.datasets.MNIST", lambda *args, **kwargs: dataset)

    cfg = Exp.DataloaderCfg(
        data_root=str(tmp_path),
        train_batch_size_per_device=4,
        val_data_worker_per_gpu=0,
    )
    partitions = []
    for rank in range(3):
        dataloader = cfg.build_val_dataloader(SimpleNamespace(rank=rank, world_size=3))
        assert dataloader.drop_last is False
        partitions.append(list(dataloader.dataset.indices))

    assert partitions == [[0, 3, 6, 9], [1, 4, 7], [2, 5, 8]]
    assert sorted(index for partition in partitions for index in partition) == list(range(10))
