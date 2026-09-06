import json
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
python = root / ".venv" / "bin" / "python"
if not python.exists():
    raise SystemExit(f"Missing virtual environment interpreter: {python}")  # noqa: TRY003
version = subprocess.check_output(  # noqa: S603
    [str(python), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"], text=True
).strip()
config = {
    "exclude": ["**/__pycache__", "dist", "build", ".venv/**", "venv/**", "**/site-packages/**"],
    "executionEnvironments": [{"pythonVersion": version, "root": "."}],
    "include": ["tinyexp", "tests"],
    "typeCheckingMode": "standard",
    "useLibraryCodeForTypes": True,
    "pythonVersion": version,
    "venv": ".venv",
    "venvPath": ".",
}
(root / "pyrightconfig.json").write_text(json.dumps(config, indent=2) + "\n")
print(f"Generated pyrightconfig.json for Python {version}")
