import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "script_name,args,patterns",
    [
        pytest.param(
            "mnist_exp.py",
            ["mode=help", "dataloader_cfg.train_batch_size_per_device=16", "ray_cfg.ray_num_worker=1"],
            [r"train_batch_size_per_device:\s*16\b", r"ray_num_worker:\s*1\b"],
            id="mnist",
        ),
        pytest.param(
            "mae_exp.py",
            ["mode=help", "module_cfg.num_classes=10"],
            [
                r"num_classes:\s*10\b",
                r"mask_ratio:\s*0\.75\b",
                r"norm_pix_loss:\s*true\b",
                r"epochs:\s*800\b",
                r"redis_cache_enabled:\s*true\b",
            ],
            id="mae",
        ),
    ],
)
def test_cli_override_prints_updated_value(
    tmp_path: Path, script_name: str, args: list[str], patterns: list[str]
) -> None:
    project_root = Path(__file__).resolve().parents[2]
    script_path = project_root / "tinyexp" / "examples" / script_name
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
        [sys.executable, str(script_path), *args],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )

    combined_output = f"{result.stdout}\n{result.stderr}"
    for pattern in patterns:
        assert re.search(pattern, combined_output)
