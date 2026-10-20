from __future__ import annotations

import random
import subprocess
import sys
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


def test_import_tinyexp_does_not_load_optional_heavy_packages() -> None:
    """``import tinyexp`` must work without the ``[pytorch]`` extra; torch/wandb stay lazily imported.

    Runs in a subprocess because this pytest process has already imported torch.
    """
    code = "import sys, tinyexp, tinyexp.tiny_engine; print(sorted({'torch', 'wandb'} & set(sys.modules)))"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)  # noqa: S603

    assert result.stdout.strip() == "[]"


def test_logger_cfg_creates_run_dir(tmp_path: Path) -> None:
    exp = LoggerExp(output_root=str(tmp_path), exp_name="demo_exp")
    run_dir = Path(exp.get_run_dir())
    assert run_dir == tmp_path / "demo_exp"

    exp.logger_cfg.build_logger(save_dir=str(run_dir), distributed_rank=0)

    assert run_dir.is_dir()
    assert (run_dir / "log.txt").is_file()


class _DummyScaler:
    def __init__(self, scale: float) -> None:
        self.scale = scale

    def state_dict(self) -> dict[str, float]:
        return {"scale": self.scale}

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.scale = state["scale"]


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
        scaler=_DummyScaler(1024.0),
        epoch=3,
        global_step=12,
        best_metric=0.9,
        exp_name="demo_exp",
        exp_class="tests.demo.Exp",
        # Reserved top-level keys inside extra_state stay inside extra_state.
        extra_state={"custom_value": 7, "epoch": 99, "meta": {"exp_name": "bad"}},
    )

    reloaded_model = torch.nn.Linear(2, 1)
    reloaded_optimizer = torch.optim.SGD(reloaded_model.parameters(), lr=0.1, momentum=0.9)
    reloaded_scheduler = torch.optim.lr_scheduler.StepLR(reloaded_optimizer, step_size=1)
    reloaded_scaler = _DummyScaler(1.0)

    checkpoint = checkpoint_cfg.load_checkpoint(
        checkpoint_path,
        model=reloaded_model,
        optimizer=reloaded_optimizer,
        scheduler=reloaded_scheduler,
        scaler=reloaded_scaler,
    )

    assert Path(checkpoint_path).is_file()
    assert checkpoint["epoch"] == 3
    assert checkpoint["global_step"] == 12
    assert checkpoint["best_metric"] == 0.9
    assert checkpoint["extra_state"]["custom_value"] == 7
    assert checkpoint["extra_state"]["epoch"] == 99
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
    assert reloaded_scaler.scale == 1024.0


def test_checkpoint_cfg_rng_state_survives_save_and_load_on_torch_26_plus(tmp_path: Path) -> None:
    """The numpy RNG state must load under torch>=2.6's ``weights_only`` default and restore the continuation."""
    checkpoint_cfg = CheckpointCfgMixin.CheckpointCfg()
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    checkpoint_path = checkpoint_cfg.save_checkpoint(
        run_dir=str(tmp_path),
        name=checkpoint_cfg.last_ckpt_name,
        extra_state={"rng_state": checkpoint_cfg.capture_rng_state()},
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


@pytest.mark.parametrize(
    ("payload", "load_arg", "error", "match"),
    [
        ({"state_dict": {"weight": torch.tensor([1.0])}}, None, UnsupportedCheckpointFormatError, "not a supported"),
        ([1, 2, 3], None, TypeError, "must be a dict, got list"),
        ({"meta": {}, "epoch": 1}, "model", KeyError, "model_state_dict"),
        ({"meta": {}}, "optimizer", KeyError, "optimizer_state_dict"),
        ({"meta": {}}, "scheduler", KeyError, "scheduler_state_dict"),
    ],
)
def test_checkpoint_cfg_rejects_invalid_payloads(
    tmp_path: Path, payload: object, load_arg: str | None, error: type, match: str
) -> None:
    checkpoint_path = tmp_path / "bad.ckpt"
    torch.save(payload, checkpoint_path)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    targets = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=1),
    }
    kwargs = {load_arg: targets[load_arg]} if load_arg else {}

    with pytest.raises(error, match=match):
        CheckpointCfgMixin.CheckpointCfg().load_checkpoint(str(checkpoint_path), **kwargs)
