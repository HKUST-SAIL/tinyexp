from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from tinyexp import CheckpointCfgMixin, LoggerCfgMixin, RayCfgMixin, RedisCfgMixin, TinyExp, WandbCfgMixin
from tinyexp.exceptions import UnsupportedCheckpointFormatError


@dataclass
class LoggerExp(TinyExp, LoggerCfgMixin):
    pass


def test_root_package_re_exports_mixins() -> None:
    assert CheckpointCfgMixin.__name__ == "CheckpointCfgMixin"
    assert LoggerCfgMixin.__name__ == "LoggerCfgMixin"
    assert RayCfgMixin.__name__ == "RayCfgMixin"
    assert RedisCfgMixin.__name__ == "RedisCfgMixin"
    assert WandbCfgMixin.__name__ == "WandbCfgMixin"


def test_get_run_dir(tmp_path: Path) -> None:
    exp = TinyExp(output_root=str(tmp_path), exp_name="demo_exp")

    expected = tmp_path / "demo_exp"
    assert exp.get_run_dir() == str(expected)


def test_logger_cfg_creates_run_dir(tmp_path: Path) -> None:
    exp = LoggerExp(output_root=str(tmp_path), exp_name="demo_exp")
    run_dir = Path(exp.get_run_dir())

    exp.logger_cfg.build_logger(save_dir=str(run_dir), distributed_rank=0)

    assert run_dir.is_dir()
    assert (run_dir / "log.txt").is_file()


def test_checkpoint_cfg_save_and_load_roundtrip(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

    with torch.no_grad():
        model.weight.fill_(1.5)
        model.bias.fill_(0.5)

    checkpoint_cfg = CheckpointCfgMixin.CheckpointCfg()
    checkpoint_path = checkpoint_cfg.save_checkpoint(
        run_dir=str(tmp_path),
        name=checkpoint_cfg.last_ckpt_name,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=3,
        global_step=12,
        best_metric=0.9,
        exp_name="demo_exp",
        exp_class="tests.demo.Exp",
        extra_state={"custom_value": 7},
    )

    reloaded_model = torch.nn.Linear(2, 1)
    reloaded_optimizer = torch.optim.SGD(reloaded_model.parameters(), lr=0.1)
    reloaded_scheduler = torch.optim.lr_scheduler.StepLR(reloaded_optimizer, step_size=1)

    checkpoint = checkpoint_cfg.load_checkpoint(
        checkpoint_path,
        model=reloaded_model,
        optimizer=reloaded_optimizer,
        scheduler=reloaded_scheduler,
    )

    assert Path(checkpoint_path).is_file()
    assert checkpoint["epoch"] == 3
    assert checkpoint["global_step"] == 12
    assert checkpoint["best_metric"] == 0.9
    assert checkpoint["extra_state"]["custom_value"] == 7
    assert checkpoint["meta"]["exp_name"] == "demo_exp"
    assert checkpoint["meta"]["exp_class"] == "tests.demo.Exp"
    assert "saved_at" in checkpoint["meta"]

    for original_param, reloaded_param in zip(model.parameters(), reloaded_model.parameters()):
        assert torch.equal(original_param, reloaded_param)


def test_checkpoint_cfg_extra_state_does_not_override_reserved_keys(
    tmp_path: Path,
) -> None:
    checkpoint_cfg = CheckpointCfgMixin.CheckpointCfg()

    checkpoint_path = checkpoint_cfg.save_checkpoint(
        run_dir=str(tmp_path),
        name=checkpoint_cfg.last_ckpt_name,
        epoch=3,
        extra_state={"epoch": 99, "meta": {"exp_name": "bad"}},
    )

    checkpoint = checkpoint_cfg.load_checkpoint(checkpoint_path)

    assert checkpoint["epoch"] == 3
    assert checkpoint["extra_state"]["epoch"] == 99
    assert checkpoint["meta"]["exp_name"] == ""


def test_checkpoint_cfg_rejects_unsupported_model_only_format(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "model_only.ckpt"
    torch.save({"state_dict": {"weight": torch.tensor([1.0])}}, checkpoint_path)

    checkpoint_cfg = CheckpointCfgMixin.CheckpointCfg()

    with pytest.raises(
        UnsupportedCheckpointFormatError,
        match="not a supported tinyexp checkpoint format",
    ):
        checkpoint_cfg.load_checkpoint(str(checkpoint_path))


def test_checkpoint_cfg_rejects_non_dict_payload(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "not_a_dict.ckpt"
    torch.save([1, 2, 3], checkpoint_path)

    checkpoint_cfg = CheckpointCfgMixin.CheckpointCfg()

    with pytest.raises(TypeError, match="must be a dict, got list"):
        checkpoint_cfg.load_checkpoint(str(checkpoint_path))


def test_checkpoint_cfg_requires_model_state_when_model_is_provided(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "missing_model_state.ckpt"
    torch.save({"meta": {}, "epoch": 1}, checkpoint_path)

    checkpoint_cfg = CheckpointCfgMixin.CheckpointCfg()
    model = torch.nn.Linear(2, 1)

    with pytest.raises(KeyError, match="model_state_dict"):
        checkpoint_cfg.load_checkpoint(str(checkpoint_path), model=model)


def test_checkpoint_cfg_requires_optimizer_state_when_optimizer_is_provided(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "missing_optimizer_state.ckpt"
    torch.save({"meta": {}, "model_state_dict": {}}, checkpoint_path)

    checkpoint_cfg = CheckpointCfgMixin.CheckpointCfg()
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(KeyError, match="optimizer_state_dict"):
        checkpoint_cfg.load_checkpoint(str(checkpoint_path), optimizer=optimizer)


def test_checkpoint_cfg_requires_scheduler_state_when_scheduler_is_provided(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "missing_scheduler_state.ckpt"
    torch.save(
        {"meta": {}, "model_state_dict": {}, "optimizer_state_dict": {}},
        checkpoint_path,
    )

    checkpoint_cfg = CheckpointCfgMixin.CheckpointCfg()
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

    with pytest.raises(KeyError, match="scheduler_state_dict"):
        checkpoint_cfg.load_checkpoint(str(checkpoint_path), scheduler=scheduler)
