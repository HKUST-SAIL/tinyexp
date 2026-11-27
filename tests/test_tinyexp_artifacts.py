from __future__ import annotations

import random
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tinyexp import (
    CheckpointCfgMixin,
    GeneralTrackCfgMixin,
    LoggerCfgMixin,
    RayCfgMixin,
    RedisCfgMixin,
    SwanlabCfgMixin,
    TinyExp,
    TrackioCfgMixin,
    WandbCfgMixin,
)
from tinyexp.exceptions import UnsupportedCheckpointFormatError


@dataclass
class LoggerExp(TinyExp, LoggerCfgMixin):
    pass


def test_root_package_re_exports_mixins() -> None:
    assert CheckpointCfgMixin.__name__ == "CheckpointCfgMixin"
    assert GeneralTrackCfgMixin.__name__ == "GeneralTrackCfgMixin"
    assert LoggerCfgMixin.__name__ == "LoggerCfgMixin"
    assert RayCfgMixin.__name__ == "RayCfgMixin"
    assert RedisCfgMixin.__name__ == "RedisCfgMixin"
    assert SwanlabCfgMixin.__name__ == "SwanlabCfgMixin"
    assert TrackioCfgMixin.__name__ == "TrackioCfgMixin"
    assert WandbCfgMixin.__name__ == "WandbCfgMixin"


def test_trackio_cfg_builds_trackio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Disabled: ``None``. Enabled: the module, with ``init`` on rank 0 only."""
    cfg = TrackioCfgMixin.TrackioCfg()
    monkeypatch.setenv("TRACKIO_DIR", str(tmp_path / "trackio"))
    assert cfg.build_trackio() is None

    calls: list[dict] = []
    fake = type(sys)("trackio")
    fake.TRACKIO_DIR = tmp_path / "trackio"  # type: ignore[attr-defined]
    fake.init = lambda **kwargs: calls.append(kwargs)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "trackio", fake)

    cfg.enable_trackio = True
    assert cfg.build_trackio(output_root=str(tmp_path), project="p") is fake
    assert cfg.build_trackio(accelerator=SimpleNamespace(rank=1), output_root=str(tmp_path), project="p") is fake
    assert calls == [{"project": "p"}]

    with pytest.raises(RuntimeError, match="already imported"):
        cfg.build_trackio(output_root=str(tmp_path / "another-root"), project="p")
    assert calls == [{"project": "p"}]


def test_trackio_persists_metrics_under_output_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if sys.version_info < (3, 10):
        pytest.skip("Trackio requires Python 3.10+")
    monkeypatch.setenv("TRACKIO_DIR", str(tmp_path / "old-cache"))
    code = """
import sys
from tinyexp import TrackioCfgMixin
tracker = TrackioCfgMixin.TrackioCfg(enable_trackio=True).build_trackio(
    output_root=sys.argv[1], project="test", auto_log_cpu=False, auto_log_gpu=False,
)
tracker.log({"val_metric": 0.5})
tracker.finish()
"""
    subprocess.run(  # noqa: S603
        [sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True, check=True
    )

    with sqlite3.connect(tmp_path / "trackio" / "test.db") as connection:
        assert connection.execute("SELECT count(*) FROM metrics").fetchone()[0] == 1
    assert not (tmp_path / "old-cache").exists()


def test_swanlab_cfg_builds_run_on_rank_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = SwanlabCfgMixin.SwanlabCfg()
    monkeypatch.setitem(sys.modules, "swanlab", None)
    assert cfg.build_swanlab() is None
    cfg.enable_swanlab = True
    assert cfg.build_swanlab(accelerator=SimpleNamespace(rank=1)) is None

    calls: list[dict] = []
    run = object()

    def init(**kwargs):
        calls.append(kwargs)
        return run

    monkeypatch.setitem(sys.modules, "swanlab", SimpleNamespace(init=init))
    kwargs = {"project": "p", "name": "exp", "config": {"lr": 0.01, "a_cfg": {"b": {"c": 1}}, "empty_cfg": {}}}
    assert cfg.build_swanlab(accelerator=SimpleNamespace(rank=0), output_root=str(tmp_path), **kwargs) is run
    cfg.mode = "local"
    assert cfg.build_swanlab(log_dir=str(tmp_path), **kwargs) is run
    flat = dict(kwargs, config={"lr": 0.01, "a_cfg.b.c": 1})
    assert calls == [
        dict(flat, mode="offline", log_dir=str(tmp_path / "swanlab" / "p")),
        dict(flat, mode="local", log_dir=str(tmp_path)),
    ]


def test_swanlab_persists_metrics_under_output_root(tmp_path: Path) -> None:
    if find_spec("swanlab") is None:
        pytest.skip("SwanLab requires tinyexp[swanlab]")
    code = """
import sys
from tinyexp import SwanlabCfgMixin
tracker = SwanlabCfgMixin.SwanlabCfg(enable_swanlab=True).build_swanlab(
    output_root=sys.argv[1], project="test", name="demo", config={"lr": 0.01},
)
try:
    tracker.log({"val_metric": 0.5}, step=4)
finally:
    tracker.finish()
"""
    subprocess.run(  # noqa: S603
        [sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True, check=True
    )

    (record,) = (tmp_path / "swanlab" / "test").glob("run-*/*.swanlab")
    assert record.stat().st_size > 0


def test_import_tinyexp_does_not_load_optional_heavy_packages() -> None:
    """Keep torch and tracker imports lazy. Run in a subprocess because pytest has already imported torch."""
    code = (
        "import sys, tinyexp, tinyexp.tiny_engine; "
        "print(sorted({'torch', 'wandb', 'trackio', 'swanlab'} & set(sys.modules)))"
    )
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
