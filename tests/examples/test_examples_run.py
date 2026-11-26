"""The ``run()`` lifecycle every bundled example implements the same way: mode dispatch and accelerator cleanup."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from tinyexp.examples import mnist_exp, pi_exp, resnet_exp
from tinyexp.exceptions import UnknownExperimentModeError

EXPS = {"mnist_exp": mnist_exp.Exp, "resnet_exp": resnet_exp.ResNetExp, "pi_exp": pi_exp.Exp}
EVAL_EXPS = pytest.mark.parametrize("example", ["mnist_exp", "resnet_exp"])


class _Accelerator:
    rank = 0
    world_size = 1
    device = "cpu"
    is_main_process = True

    def __init__(self) -> None:
        self.destroy_calls = 0

    def destroy(self) -> None:
        self.destroy_calls += 1


def _patch_runtime(monkeypatch: pytest.MonkeyPatch, exp, accelerator: _Accelerator) -> list[str]:  # type: ignore[no-untyped-def]
    """Route the experiment to ``accelerator`` and a recording logger; returns the logged messages."""
    if hasattr(exp, "accelerator_cfg"):
        monkeypatch.setattr(exp.accelerator_cfg, "build_accelerator", lambda: accelerator)
    else:  # pi_exp always runs on CPUAccelerator
        monkeypatch.setattr("tinyexp.tiny_engine.accelerator.CPUAccelerator", lambda: accelerator)
    messages: list[str] = []
    logger = SimpleNamespace(info=lambda message, *args, **kwargs: messages.append(message))
    monkeypatch.setattr(exp.logger_cfg, "build_logger", lambda **kwargs: logger)
    return messages


@pytest.mark.parametrize(
    ("example", "mode", "modes"),
    [
        ("mnist_exp", "val", "train, eval, help"),
        ("resnet_exp", "val", "train, eval, help"),
        ("pi_exp", "train", "run, help"),
    ],
)
def test_run_unknown_mode_lists_supported_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, example: str, mode: str, modes: str
) -> None:
    exp = EXPS[example](output_root=str(tmp_path), mode=mode)
    _patch_runtime(monkeypatch, exp, _Accelerator())

    with pytest.raises(UnknownExperimentModeError, match=rf"Unknown mode '{mode}'.*{modes}"):
        exp.run()


@pytest.mark.parametrize("example", list(EXPS))
def test_run_destroys_accelerator_when_workload_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, example: str
) -> None:
    exp = EXPS[example](output_root=str(tmp_path), exp_name=f"{example}_failure")
    accelerator = _Accelerator()
    _patch_runtime(monkeypatch, exp, accelerator)
    monkeypatch.setattr(exp, "_run", lambda _accelerator: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="boom"):
        exp.run()

    assert accelerator.destroy_calls == 1


@EVAL_EXPS
def test_run_eval_mode_requires_resume_from_and_logs_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, example: str
) -> None:
    exp = EXPS[example](mode="eval", resume_from="")
    exp.set_cfg(OmegaConf.create({"output_root": str(tmp_path), "exp_name": f"{example}_val"}))
    messages = _patch_runtime(monkeypatch, exp, _Accelerator())

    with pytest.raises(ValueError, match="resume_from"):
        exp.run()

    # The overrides are logged with the experiment logger before the mode is dispatched.
    assert any(message.startswith("-------- Overridden Configurations --------\n") for message in messages)


@EVAL_EXPS
def test_run_eval_mode_evaluates_resume_from(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, example: str) -> None:
    checkpoint_path = str(tmp_path / "demo.ckpt")
    exp = EXPS[example](output_root=str(tmp_path), exp_name=f"{example}_val", mode="eval", resume_from=checkpoint_path)
    accelerator = _Accelerator()
    _patch_runtime(monkeypatch, exp, accelerator)
    called: dict[str, object] = {}

    def fake_evaluate(*, accelerator, logger, module_or_module_path, val_dataloader=None, **_kwargs):  # type: ignore[no-untyped-def]
        called.update(
            accelerator=accelerator, module_or_module_path=module_or_module_path, val_dataloader=val_dataloader
        )
        return 0.5

    monkeypatch.setattr(exp, "_evaluate", fake_evaluate)

    exp.run()

    assert called == {"accelerator": accelerator, "module_or_module_path": checkpoint_path, "val_dataloader": None}
    assert accelerator.destroy_calls == 1


@pytest.mark.parametrize(
    "example, mode, rank, fails",
    [("mnist_exp", "train", 0, False), ("resnet_exp", "eval", 0, True), ("vit_tp_exp", "train", 1, False)],
)
def test_run_finishes_trackers_before_destroying_accelerator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, example: str, mode: str, rank: int, fails: bool
) -> None:
    if example == "vit_tp_exp":
        exp_class = pytest.importorskip("tinyexp.examples.vit_tp_exp").VitTpExp
    else:
        exp_class = EXPS[example]
    exp = exp_class(output_root=str(tmp_path), mode=mode, resume_from="demo.ckpt")
    accelerator = _Accelerator()
    accelerator.device = SimpleNamespace(type="cpu")
    accelerator.rank = rank
    accelerator.is_main_process = rank == 0
    _patch_runtime(monkeypatch, exp, accelerator)
    events: list[str] = []
    trackio = SimpleNamespace(finish=lambda: events.append("finish_trackio"))
    swanlab = SimpleNamespace(finish=lambda: events.append("finish_swanlab"))

    def build_trackio(*, output_root, **kwargs):
        assert output_root == exp.output_root
        return trackio

    def build_swanlab(*, accelerator, output_root, project, name, config):
        assert output_root == exp.output_root
        assert name == exp.exp_name
        assert project == ("TinyExp" if example == "vit_tp_exp" else "Baselines")
        assert isinstance(config, dict)
        return swanlab if accelerator.rank == 0 else None

    monkeypatch.setattr(exp.trackio_cfg, "build_trackio", build_trackio)
    monkeypatch.setattr(exp.swanlab_cfg, "build_swanlab", build_swanlab)
    monkeypatch.setattr(accelerator, "destroy", lambda: events.append("destroy"))

    def workload(**kwargs):
        assert kwargs["trackers"] == ([trackio, swanlab] if rank == 0 else [trackio])
        events.append("workload")
        if fails:
            raise RuntimeError("workload")

    method = "_train" if mode == "train" else ("_run_eval" if example == "vit_tp_exp" else "_evaluate")
    monkeypatch.setattr(exp, method, workload)

    if fails:
        with pytest.raises(RuntimeError, match="workload"):
            exp.run()
    else:
        exp.run()

    expected = ["workload", "finish_swanlab", "finish_trackio", "destroy"] if rank == 0 else ["workload", "destroy"]
    assert events == expected
