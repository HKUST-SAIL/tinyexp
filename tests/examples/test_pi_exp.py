from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tinyexp.examples import pi_exp

RESULT = "pi ~= 3.140000 (error=0.000000, samples=10000000)"


@pytest.mark.parametrize("launcher", ["mp", "ray"])
def test_pi_run_prints_result_or_returns_it_to_the_ray_driver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], launcher: str
) -> None:
    accelerator = SimpleNamespace(rank=0, world_size=1, is_main_process=True, destroy=lambda: None)
    exp = pi_exp.Exp(output_root=str(tmp_path), exp_name="pi_test", launcher=launcher)
    monkeypatch.setattr(pi_exp, "torch", SimpleNamespace(pi=3.14))
    monkeypatch.setattr("tinyexp.tiny_engine.accelerator.CPUAccelerator", lambda: accelerator)
    monkeypatch.setattr(exp.logger_cfg, "build_logger", lambda **kwargs: SimpleNamespace())
    monkeypatch.setattr(exp, "print_cfg", lambda logger: {})
    monkeypatch.setattr(exp, "_estimate_pi", lambda accelerator: 3.14)

    exp.run()

    assert exp.get_ray_run_result() == RESULT
    # Under Ray the driver prints the result once; workers stay quiet.
    assert (RESULT in capsys.readouterr().out) is (launcher != "ray")
