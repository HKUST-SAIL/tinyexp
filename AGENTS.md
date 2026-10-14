# Repository Guidelines

## Agent Quick Reference
Use this index before reading source: the docstrings below are written to be sufficient. Names in backticks are
importable; print a module's with `python -c "import tinyexp; print(tinyexp.__doc__)"` and a class's with
`python -c "from tinyexp.exp_mixins import basic_mixins as m; print(m.CheckpointCfgMixin.CheckpointCfg.__doc__)"`.

| Task | Read only |
| --- | --- |
| Run or pick a bundled example | `tinyexp/examples/README.md`, then the docstring (run header) of `tinyexp.examples.<name>` |
| Write a new experiment | `tinyexp` (minimal example, fields) and `tinyexp.tiny_engine.accelerator.base_accelerator` (training loop); full training recipe, only if needed: `tinyexp/examples/mnist_exp.py` |
| Config fields, `launcher` pairing, errors; merge rules | `tinyexp`, `tinyexp.exceptions`; `tinyexp.TinyExp.set_cfg` |
| Ray workers (contract, minimal example), multi-node cluster, torchrun, accelerate | `RayCfgMixin` and `RayCfgMixin.RayCfg` in `tinyexp.exp_mixins.basic_mixins`, then one section of `docs/running-modes.md` |
| Checkpoint contents, save/load/resume/eval | `CheckpointCfgMixin.CheckpointCfg` in `tinyexp.exp_mixins.basic_mixins` |
| Logging, W&B, Redis cache | `tinyexp.exp_mixins.basic_mixins` (table), then `LoggerCfgMixin.LoggerCfg`, `WandbCfgMixin.WandbCfg` or `RedisCfgMixin.RedisCfg`; Redis wrapper: "Optional Services" in `docs/running-modes.md` |
| Pick an accelerator; training-loop API | `tinyexp.tiny_engine.accelerator`, `tinyexp.tiny_engine.accelerator.base_accelerator` |
| Infinite sampler, fake dataloader, EMA | `tinyexp.dataset`, `tinyexp.utils` |
| Change one config value | nothing: `python -m <module> mode=help key=value` prints the resolved config and the overridden values, then exits |

`mode=help` checks key names and value types, not whether a value is supported (e.g. an unknown accelerator name
passes), and it still creates `output_root`. If a run fails with `FileExistsError: ... 'output'`, the local
`./output` is unusable (e.g. a broken symlink): pass `output_root=/tmp/tinyexp` instead of editing code.

Where to make a change, in order of preference: a CLI override; a new experiment file or a subclass of an existing
`Exp`; the defaults of an example in `tinyexp/examples/`; the library (`tinyexp/` outside `examples/`), only when
the behavior itself must change.

Do not explore:
- `output/`, `data/`: run artifacts and datasets (may be large or broken symlinks). Derive checkpoint paths as
  `<output_root>/<exp_name>/{last,best}.ckpt` instead of searching; never glob `**/*.ckpt` or `**/*.pt`.
- `local/`: the maintainer's private experiments, git-ignored and not library code. Do not read, import, modify,
  or use it as a reference for new experiments.
- `.venv/`, `site/`, `__pycache__/`, `.ruff_cache/`, `.pytest_cache/`.

Before launching GPU or Ray jobs (anything beyond `mode=help` or CPU tests), confirm with the user.

## Development
- Layout: `tinyexp/` library, `tinyexp/examples/` runnable experiments, `tests/` mirroring the package
  (`tests/utils/`, `tests/tiny_engine/`; the TP accelerator is tested with its example in
  `tests/examples/test_vit_tp_exp.py`), `docs/` + `mkdocs.yml` for MkDocs. Do not commit large files from `output/`.
- `uv` manages `.venv/` and `uv.lock`. Prefer `make` targets: `make install` (env + `pre-commit` hooks),
  `make check` (lockfile + `pre-commit`: ruff/black/isort), `make test` (`pytest --cov`, writes `coverage.xml`),
  `make docs` / `make docs-test` (serve / build MkDocs), `make build` (wheel into `dist/`). `tox` runs the supported
  Python versions. `mypy` is disabled; its dependency and configuration are kept for re-enablement.
- Style: Python 3.9+, 4-space indentation, line length 120, explicit names and type hints.
- 如无必要，勿增实体: do not add new concepts, state, helpers, layers, or dependencies unless they are needed to solve
  the concrete problem. Keep It Simple, Stupid (KISS): prefer the smallest direct implementation that preserves
  behavior and is easy to reason about. Keep patches focused and reviewable.
- Tests: `pytest`, files `tests/**/test_*.py`; fast and deterministic (CPU-first). Gate external services
  (Redis/W&B) behind env vars and provide safe fallbacks/mocks.
- Commits follow Conventional Commits and often include an emoji (`feat: 🎸 add new accelerator`). PRs include a
  clear description, linked issue (if any), tests for behavior changes, and `README.md`/`docs/` updates when
  user-facing behavior changes.
- Keep the agent-facing docs true. `tests/examples/test_examples_help_smoke.py` requires every example to carry a
  run header and statically checks the commands in the run headers, `README.md`, `SKILL.md` and `docs/*.md` (key
  names and types only, so a command can be legal yet stale), and that each run header's `Modes` are handled in
  `run()`. `tests/test_tinyexp_artifacts.py` asserts in a subprocess that `import tinyexp` loads neither torch nor
  wandb, and that every public module has a docstring.
- Invariants the docs rely on: keep the `import torch` in `CheckpointCfg` and the `import wandb` in `WandbCfg` lazy
  (inside methods); keep `from tinyexp.exp_mixins.basic_mixins import *Mixin` importable (released path); do not
  move `tinyexp/__init__.py` and do not add a yaml at the package root (Hydra's `pkg://tinyexp` needs the
  ConfigStore entry `cfg`); code moved out of `examples/` enters coverage, where the patch target is 80%.
