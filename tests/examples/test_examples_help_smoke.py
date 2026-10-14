"""Keep the bundled examples runnable and every documented command truthful.

- Every example answers ``mode=help`` from a clean directory (first-run reliability).
- Every example starts with a run header (``Modes`` / ``Run`` / ``Env`` / ``Needs``), and every mode it lists is
  handled by the experiment.
- Every command in a run header and in the user docs (README, SKILL, docs/) is checked statically:
  overrides for a bundled example are composed with Hydra (unknown keys and bad types fail) and
  ``tinyexp-run-with-*`` arguments are parsed with the tool's own argument parser.
"""

import ast
import importlib
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = PROJECT_ROOT / "tinyexp" / "examples"
EXAMPLES = sorted(p.stem for p in EXAMPLES_DIR.glob("*_exp.py"))
HEADER_KEYS = ("Modes", "Run", "Env", "Needs")
# User-facing docs; the refactor plan is a historical record and may quote superseded commands on purpose.
USER_DOCS = [
    PROJECT_ROOT / "README.md",
    PROJECT_ROOT / "SKILL.md",
    *sorted((PROJECT_ROOT / "docs").glob("*.md")),
]
CLI_TOOLS = {
    "tinyexp-run-with-ray-cluster": "tinyexp.cli.run_with_ray_cluster",
    "tinyexp-run-with-redis": "tinyexp.cli.run_with_redis",
}
_EXAMPLE_TOKEN = re.compile(r"^tinyexp(?:\.examples\.|/examples/)(\w+_exp)(?:\.py)?$")


def _run_help(module: str, cwd: Path, *overrides: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(PROJECT_ROOT) if not existing_pythonpath else f"{PROJECT_ROOT}{os.pathsep}{existing_pythonpath}"
    )
    env.setdefault("HYDRA_FULL_ERROR", "1")
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("WANDB_SILENT", "true")
    return subprocess.run(  # noqa: S603
        [sys.executable, "-m", f"tinyexp.examples.{module}", "mode=help", *overrides],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _header(example: str) -> dict[str, list[str]]:
    """Return ``{key: lines}`` for the run header block (``Key : value`` lines plus indented continuations)."""
    doc = ast.get_docstring(ast.parse((EXAMPLES_DIR / f"{example}.py").read_text())) or ""
    sections: dict[str, list[str]] = {}
    current = None
    for line in doc.splitlines():
        match = re.match(r"^(\w+)\s*:(.*)$", line)
        if match and match.group(1) in HEADER_KEYS:
            current = match.group(1)
            sections[current] = [match.group(2).strip()] if match.group(2).strip() else []
        elif current and line.startswith(" "):
            sections[current].append(line.strip())
        elif sections and not line.strip():
            break  # the header is one contiguous block
    return sections


def _shell_segments(command: str) -> list[list[str]]:
    """Split one logical shell line into token lists at ``&&``, ``||`` and ``;``; comments are dropped."""
    return [tokens for part in re.split(r"\s(?:&&|\|\||;)\s", command) if (tokens := shlex.split(part, comments=True))]


def _doc_commands() -> list[tuple[str, list[str]]]:
    """Return ``(location, tokens)`` for documented commands that use a bundled example or a TinyExp CLI tool."""
    found = []
    for path in USER_DOCS:
        in_block, logical, start = False, "", 0
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if line.lstrip().startswith("```"):
                in_block, logical = not in_block, ""
                continue
            if not in_block:
                continue
            start = start if logical else number
            if line.rstrip().endswith("\\"):
                logical += line.rstrip()[:-1] + " "
                continue
            command, logical = logical + line, ""
            for tokens in _shell_segments(command):
                if any(_EXAMPLE_TOKEN.match(t) or t in CLI_TOOLS for t in tokens):
                    found.append((f"{path.relative_to(PROJECT_ROOT)}:{start}", tokens))
    return found


def _example_and_overrides(tokens: list[str]) -> tuple[str, list[str]]:
    """Return ``(example, overrides)`` for the bundled example started by ``tokens``, or ``("", [])``."""
    for index, token in enumerate(tokens):
        match = _EXAMPLE_TOKEN.match(token)
        if match:
            overrides = [t for t in tokens[index + 1 :] if "=" in t and not t.startswith("-")]
            return match.group(1), overrides
    return "", []


def _exp_class(example: str) -> type:
    module = importlib.import_module(f"tinyexp.examples.{example}")
    class_name = re.search(r"store_and_run_exp\((\w+)\)", Path(module.__file__).read_text()).group(1)
    return getattr(module, class_name)


def _compose(example: str, overrides: list[str]) -> None:
    """Compose the example's config with ``overrides`` exactly as its CLI would; raises on unknown keys/bad types."""
    config_name = f"_command_check_{example}"
    ConfigStore.instance().store(name=config_name, node=_exp_class(example)())
    with initialize(version_base=None):
        compose(config_name=config_name, overrides=overrides)


def _check_command(tokens: list[str]) -> None:
    for position, token in enumerate(tokens):
        if token in CLI_TOOLS:
            importlib.import_module(CLI_TOOLS[token]).parse_args(tokens[position + 1 :])
    example, overrides = _example_and_overrides(tokens)
    if example:
        _compose(example, overrides)


@pytest.mark.parametrize("module", EXAMPLES)
def test_example_help_succeeds_in_clean_directory(module: str, tmp_path: Path) -> None:
    result = _run_help(module, tmp_path)

    assert result.returncode == 0, result.stderr
    assert "-------- Configurations --------" in f"{result.stdout}\n{result.stderr}"


@pytest.mark.parametrize("example", EXAMPLES)
def test_example_has_run_header(example: str) -> None:
    header = _header(example)

    assert set(HEADER_KEYS) <= set(header), f"{example} header has {sorted(header)}"
    assert all(header[key] for key in HEADER_KEYS)
    assert any(f"-m tinyexp.examples.{example} mode=help" in line for line in header["Run"])


@pytest.mark.parametrize("example", EXAMPLES)
def test_example_header_commands_compose(example: str) -> None:
    commands = [tokens for line in _header(example)["Run"] for tokens in _shell_segments(line)]
    assert commands

    for tokens in commands:
        assert _example_and_overrides(tokens)[0] == example, f"{example} header runs something else: {tokens}"
        _check_command(tokens)


@pytest.mark.parametrize("example", EXAMPLES)
def test_example_header_modes_are_handled(example: str) -> None:
    source = (EXAMPLES_DIR / f"{example}.py").read_text()
    modes = [m.strip() for m in _header(example)["Modes"][0].split("(")[0].split("|")]

    assert "help" in modes
    for mode in modes:
        if mode != "help":
            assert re.search(rf'self\.mode\s*[!=]=\s*"{mode}"', source), f"{example}: mode {mode!r} not handled"


def test_user_docs_contain_checked_commands() -> None:
    """Guard against the doc scan silently finding nothing (e.g. after a docs restructure)."""
    locations = {location.split(":")[0] for location, _ in _doc_commands()}

    assert {"README.md", "SKILL.md", "docs/running-modes.md", "docs/vit_tp.md"} <= locations


@pytest.mark.parametrize("location, tokens", _doc_commands(), ids=lambda value: value if isinstance(value, str) else "")
def test_documented_command_is_valid(location: str, tokens: list[str]) -> None:
    try:
        _check_command(tokens)
    except SystemExit as error:  # argparse reports invalid CLI arguments by exiting
        pytest.fail(f"{location}: invalid arguments for {tokens[0]!r}: exit {error.code}")


def test_help_writes_only_under_output_root(tmp_path: Path) -> None:
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    output_root = tmp_path / "out"

    result = _run_help("mnist_exp", cwd, f"output_root={output_root}")

    assert result.returncode == 0, result.stderr
    assert list(cwd.iterdir()) == []


def test_help_ignores_unusable_default_output_in_cwd(tmp_path: Path) -> None:
    """A stale ``./output`` (e.g. a broken symlink) must not break runs that set ``output_root``."""
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / "output").symlink_to(tmp_path / "missing-target")

    result = _run_help("mnist_exp", cwd, f"output_root={tmp_path / 'out'}")

    assert result.returncode == 0, result.stderr
