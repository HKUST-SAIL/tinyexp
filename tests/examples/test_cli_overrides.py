import os
import re
import subprocess
import sys
from pathlib import Path


def test_cli_override_prints_updated_value(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    script_path = project_root / "tinyexp" / "examples" / "mnist_exp.py"
    assert script_path.is_file()

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(project_root) if not existing_pythonpath else f"{project_root}{os.pathsep}{existing_pythonpath}"
    )
    env.setdefault("HYDRA_FULL_ERROR", "1")
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("WANDB_SILENT", "true")

    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(script_path),
            "mode=help",
            "dataloader_cfg.train_batch_size_per_device=16",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )

    combined_output = f"{result.stdout}\n{result.stderr}"
    assert re.search(r"train_batch_size_per_device:\s*16\b", combined_output)
