from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
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
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

    with torch.no_grad():
        model.weight.fill_(1.5)
        model.bias.fill_(0.5)

    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()

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
    reloaded_optimizer = torch.optim.SGD(reloaded_model.parameters(), lr=0.1, momentum=0.9)
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
    for original_state, reloaded_state in zip(optimizer.state.values(), reloaded_optimizer.state.values()):
        assert original_state.keys() == reloaded_state.keys()
        for key in original_state:
            assert torch.equal(original_state[key], reloaded_state[key])
    assert reloaded_scheduler.state_dict() == scheduler.state_dict()


def test_checkpoint_cfg_scaler_state_roundtrip(tmp_path: Path) -> None:
    class DummyScaler:
        def __init__(self, scale: float) -> None:
            self.scale = scale

        def state_dict(self) -> dict[str, float]:
            return {"scale": self.scale}

        def load_state_dict(self, state: dict[str, float]) -> None:
            self.scale = state["scale"]

    checkpoint_cfg = CheckpointCfgMixin.CheckpointCfg()
    checkpoint_path = checkpoint_cfg.save_checkpoint(
        run_dir=str(tmp_path),
        name=checkpoint_cfg.last_ckpt_name,
        scaler=DummyScaler(1024.0),
    )
    reloaded_scaler = DummyScaler(1.0)

    checkpoint_cfg.load_checkpoint(checkpoint_path, scaler=reloaded_scaler)

    assert reloaded_scaler.scale == 1024.0


def test_checkpoint_cfg_rng_state_restores_continuation() -> None:
    checkpoint_cfg = CheckpointCfgMixin.CheckpointCfg()
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    state = checkpoint_cfg.capture_rng_state()
    expected = (random.random(), np.random.random(), torch.rand(3))  # noqa: S311

    random.random()  # noqa: S311
    np.random.random()
    torch.rand(3)
    checkpoint_cfg.restore_rng_state(state)
    resumed = (random.random(), np.random.random(), torch.rand(3))  # noqa: S311

    assert resumed[0] == expected[0]
    assert resumed[1] == expected[1]
    assert torch.equal(resumed[2], expected[2])


def test_checkpoint_cfg_loads_checkpoint_with_rng_state_on_torch_26_plus(tmp_path: Path) -> None:
    checkpoint_cfg = CheckpointCfgMixin.CheckpointCfg()
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    rng_state = checkpoint_cfg.capture_rng_state()
    checkpoint_path = checkpoint_cfg.save_checkpoint(
        run_dir=str(tmp_path),
        name=checkpoint_cfg.last_ckpt_name,
        extra_state={"rng_state": rng_state},
    )

    expected = (random.random(), np.random.random(), torch.rand(3))  # noqa: S311
    random.random()  # noqa: S311
    np.random.random()
    torch.rand(3)

    checkpoint = checkpoint_cfg.load_checkpoint(checkpoint_path)
    checkpoint_cfg.restore_rng_state(checkpoint["extra_state"]["rng_state"])
    resumed = (random.random(), np.random.random(), torch.rand(3))  # noqa: S311

    assert resumed[0] == expected[0]
    assert resumed[1] == expected[1]
    assert torch.equal(resumed[2], expected[2])


def test_checkpoint_cfg_atomic_save_preserves_previous_checkpoint_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_cfg = CheckpointCfgMixin.CheckpointCfg()
    checkpoint_path = checkpoint_cfg.save_checkpoint(
        run_dir=str(tmp_path),
        name=checkpoint_cfg.last_ckpt_name,
        epoch=3,
    )

    def fail_save(*args, **kwargs):
        file_obj = args[1]
        file_obj.write(b"partial checkpoint")
        raise RuntimeError("simulated checkpoint write failure")  # noqa: TRY003

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(RuntimeError, match="simulated checkpoint write failure"):
        checkpoint_cfg.save_checkpoint(
            run_dir=str(tmp_path),
            name=checkpoint_cfg.last_ckpt_name,
            epoch=4,
        )

    checkpoint = checkpoint_cfg.load_checkpoint(checkpoint_path)
    assert checkpoint["epoch"] == 3
    assert list(tmp_path.glob(f".{checkpoint_cfg.last_ckpt_name}.*.tmp")) == []


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
