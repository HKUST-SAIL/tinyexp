from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import ray

from tinyexp.examples import pi_exp


def test_pi_run_prints_result_on_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    events: list[str] = []

    class DummyAccelerator:
        rank = 0
        world_size = 1
        is_main_process = True

        def reduce_sum(self, tensor):
            return tensor

        def destroy(self) -> None:
            events.append("destroy")

    accelerator = DummyAccelerator()
    logger = SimpleNamespace()
    exp = pi_exp.Exp(output_root=str(tmp_path), exp_name="pi_test")

    monkeypatch.setattr(pi_exp, "torch", SimpleNamespace(pi=3.14))
    monkeypatch.setattr("tinyexp.tiny_engine.accelerator.CPUAccelerator", lambda: accelerator)
    monkeypatch.setattr(exp.logger_cfg, "build_logger", lambda **kwargs: logger)
    monkeypatch.setattr(exp, "print_cfg", lambda logger: {})
    monkeypatch.setattr(exp, "_estimate_pi", lambda accelerator: 3.14)

    exp.run()

    assert events == ["destroy"]
    assert "pi ~= 3.140000 (error=0.000000, samples=10000000)" in capsys.readouterr().out


def test_pi_run_returns_result_to_ray_driver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class DummyAccelerator:
        rank = 0
        world_size = 1
        is_main_process = True

        def destroy(self) -> None:
            return None

    exp = pi_exp.Exp(output_root=str(tmp_path), exp_name="pi_ray")
    monkeypatch.setattr(pi_exp, "torch", SimpleNamespace(pi=3.14))
    monkeypatch.setattr("tinyexp.tiny_engine.accelerator.CPUAccelerator", DummyAccelerator)
    monkeypatch.setattr(exp.logger_cfg, "build_logger", lambda **kwargs: SimpleNamespace())
    monkeypatch.setattr(exp, "print_cfg", lambda logger: {})
    monkeypatch.setattr(exp, "_estimate_pi", lambda accelerator: 3.14)
    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(
        ray,
        "get_runtime_context",
        lambda: SimpleNamespace(get_actor_id=lambda: "actor-id"),
    )

    exp.run()

    assert exp.get_ray_run_result() == "pi ~= 3.140000 (error=0.000000, samples=10000000)"
    assert "pi ~= " not in capsys.readouterr().out


def test_pi_run_destroys_accelerator_when_workload_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class DummyAccelerator:
        rank = 0
        world_size = 1
        is_main_process = True

        def destroy(self) -> None:
            events.append("destroy")

    accelerator = DummyAccelerator()
    exp = pi_exp.Exp(output_root=str(tmp_path), exp_name="pi_failure")

    monkeypatch.setattr("tinyexp.tiny_engine.accelerator.CPUAccelerator", lambda: accelerator)
    monkeypatch.setattr(exp, "_run", lambda _accelerator: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="boom"):
        exp.run()

    assert events == ["destroy"]
