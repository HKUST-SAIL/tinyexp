"""Run-level tests for the vit_tp_exp example (docs/vit_tp.md).

The train/eval/bench paths are exercised end to end on CPU with fake data (single
process, launcher=mp); the TP-specific numerical equivalence lives in
test_vit_tp_exp_tp.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("timm", reason="tinyexp vit extra (timm) is required for the vit_tp_exp example")

import torch

from tinyexp.examples.vit_tp_exp import VitTpExp


def _tiny_exp(tmp_path: Path, mode: str = "train", **overrides) -> VitTpExp:
    defaults = {
        "mode": mode,
        "launcher": "mp",
        "output_root": str(tmp_path),
        "exp_name": "vit_tp_test",
        "epochs": 1,
        "accelerator_cfg": VitTpExp.AcceleratorCfg(accelerator="cpu"),
        "module_cfg": VitTpExp.ModuleCfg(img_size=32, num_classes=10),
        "dataloader_cfg": VitTpExp.DataloaderCfg(
            fake_data=True,
            fake_data_len=16,
            input_size=32,
            train_batch_size_per_device=4,
            val_batch_size_per_device=4,
            num_workers=0,
            pin_mem=False,
        ),
        "bench_cfg": VitTpExp.BenchCfg(warmup_steps=1, measure_steps=2),
    }
    defaults.update(overrides)
    return VitTpExp(**defaults)


def test_train_smoke_produces_official_layout_checkpoint(tmp_path: Path) -> None:
    exp = _tiny_exp(tmp_path)
    exp.run()

    run_dir = Path(exp.get_run_dir())
    stats_lines = (run_dir / "log.txt").read_text().strip().splitlines()
    assert len(stats_lines) == 1
    stats = json.loads(stats_lines[0])
    assert stats["epoch"] == 0
    assert "train_loss" in stats and "test_acc1" in stats

    checkpoint = torch.load(run_dir / "last.ckpt", map_location="cpu", weights_only=False)
    model_state = checkpoint["model_state_dict"]
    # official fused-qkv layout: byte-compatible with facebookresearch/deit checkpoints
    assert "blocks.0.attn.qkv.weight" in model_state
    assert not any(key.endswith(".attn.q.weight") for key in model_state)
    assert checkpoint["optimizer_state_dict"]


def test_eval_mode_reproduces_training_accuracy(tmp_path: Path) -> None:
    train_exp = _tiny_exp(tmp_path)
    train_exp.run()
    best_metric = torch.load(Path(train_exp.get_run_dir()) / "best.ckpt", map_location="cpu", weights_only=False)[
        "best_metric"
    ]

    eval_exp = _tiny_exp(tmp_path, mode="eval", resume_from=str(Path(train_exp.get_run_dir()) / "best.ckpt"))
    eval_exp.run()

    # deterministic protocol: same checkpoint, sequential full val -> same number
    assert eval_exp._run_result.startswith(f"eval acc@1={best_metric:.2f}%")


def test_eval_mode_requires_checkpoint_source(tmp_path: Path) -> None:
    exp = _tiny_exp(tmp_path, mode="eval")
    with pytest.raises(ValueError, match="resume_from"):
        exp.run()


def test_bench_mode_reports_result(tmp_path: Path) -> None:
    exp = _tiny_exp(tmp_path, mode="bench")
    exp.run()
    assert exp._run_result.startswith("bench[cpu world=1]")


def test_ray_export_survives_builtins_print_patch(tmp_path: Path) -> None:
    """Platform agents may replace ``builtins.print`` between the ``python -m``
    module execution and the ray actor export; shipping the ``__main__``-defined
    exp class by value then dies with "Can't pickle <built-in function print>:
    it's not the same object as builtins.print". ``store_and_run_exp`` must
    resolve the canonical importable twin so both the actor class and the
    structured-config metadata ship by reference."""
    import subprocess
    import sys

    driver = (
        "import builtins, importlib.util, sys\n"
        "import ray\n"
        "_real_init = ray.init\n"
        "def _init_then_patch_print(*args, **kwargs):\n"
        "    context = _real_init(*args, **kwargs)\n"
        "    # platform-agent style patch AFTER the exp module ran as __main__\n"
        "    # (classes captured the real print), BEFORE actor export.\n"
        "    builtins.print = lambda *a, _r=builtins.print, **k: _r(*a, **k)\n"
        "    return context\n"
        "ray.init = _init_then_patch_print\n"
        "# mirror ``python -m``: the running __main__ carries the module spec\n"
        'sys.modules["__main__"].__spec__ = importlib.util.find_spec("tinyexp.examples.vit_tp_exp")\n'
        "import runpy\n"
        'sys.argv = ["vit_tp_exp.py"] + sys.argv[1:]\n'
        'runpy.run_module("tinyexp.examples.vit_tp_exp", run_name="__main__")\n'
        'print("RESULT:ok")\n'
    )
    run_dir = tmp_path / "run"
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-c",
            driver,
            "mode=bench",
            "launcher=ray",
            "accelerator_cfg.accelerator=cpu",
            "module_cfg.img_size=32",
            "module_cfg.num_classes=10",
            "dataloader_cfg.input_size=32",
            "dataloader_cfg.train_batch_size_per_device=4",
            "bench_cfg.warmup_steps=1",
            "bench_cfg.measure_steps=1",
            "ray_cfg.ray_num_worker=1",
            "ray_cfg.ray_num_gpus_per_worker=0.0",
            "ray_cfg.ray_num_cpus_per_worker=2",
            f"hydra.run.dir={run_dir}",
            f"output_root={tmp_path}",
        ],
        cwd=str(Path(__file__).resolve().parents[2]),
        text=True,
        capture_output=True,
        check=False,
        timeout=180,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
    assert "RESULT:ok" in result.stdout
    assert "PicklingError" not in result.stderr
