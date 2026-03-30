from __future__ import annotations

from pathlib import Path

import torch

from tinyexp import CheckpointCfg, TinyExp


def test_get_run_dir_and_ensure_run_dir(tmp_path: Path) -> None:
    exp = TinyExp(output_root=str(tmp_path), exp_name="demo_exp")

    expected = tmp_path / "demo_exp"
    assert exp.get_run_dir() == str(expected)

    created = Path(exp.ensure_run_dir())
    assert created == expected
    assert created.is_dir()


def test_dump_config_writes_yaml(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RANK", "0")
    exp = TinyExp(output_root=str(tmp_path), exp_name="demo_exp", mode="val", resume_from="checkpoint.ckpt")

    dumped = Path(exp.dump_config())

    assert dumped == tmp_path / "demo_exp" / "config.yaml"
    content = dumped.read_text(encoding="utf-8")
    assert "exp_name: demo_exp" in content
    assert "mode: val" in content
    assert "resume_from: checkpoint.ckpt" in content


def test_checkpoint_cfg_save_and_load_roundtrip(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

    with torch.no_grad():
        model.weight.fill_(1.5)
        model.bias.fill_(0.5)

    checkpoint_cfg = CheckpointCfg()
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
    assert checkpoint["custom_value"] == 7
    assert checkpoint["meta"]["exp_name"] == "demo_exp"
    assert checkpoint["meta"]["exp_class"] == "tests.demo.Exp"
    assert "saved_at" in checkpoint["meta"]

    for original_param, reloaded_param in zip(model.parameters(), reloaded_model.parameters()):
        assert torch.equal(original_param, reloaded_param)
