# Repository Guidelines

## Project Structure & Module Organization
- `tinyexp/`: main library code (experiment entrypoints, datasets, accelerators, utilities).
- `tinyexp/examples/`: runnable example experiments (e.g., MNIST, ResNet).
- `tests/`: `pytest` suite (mirrors package areas like `tests/utils/` and `tests/tine_engine/`).
- `docs/` + `mkdocs.yml`: MkDocs documentation sources/config.
- `data/`: local datasets used by examples (e.g., `data/MNIST/`).
- `output/`: generated run artifacts/logs—avoid committing large files.

## Build, Test, and Development Commands
This repo uses `uv` for dependency management (creates `.venv/` and maintains `uv.lock`).
- `make install`: create/sync env and install `pre-commit` hooks.
- `make check`: verify lockfile consistency and run `pre-commit` (ruff/black/isort, etc.).
- `make test`: run unit tests with coverage (`pytest --cov`, emits `coverage.xml`).
- `tox`: run tests across supported Python versions. `mypy` is currently disabled but its dependency and configuration
  are retained for future re-enablement.
- `make docs` / `make docs-test`: serve or build MkDocs docs.
- `make build`: build a wheel into `dist/` (for release preparation).

## Coding Style & Naming Conventions
- Python 3.9+; 4-space indentation; line length target is 120.
- Formatting/linting is enforced via `pre-commit`: Black + isort + Ruff (auto-fix enabled).
- Prefer explicit names and type hints; static type checking is currently disabled and may be re-enabled later.

## Development Principles
- 如无必要，勿增实体: do not add new concepts, state, helpers, layers, or dependencies unless they are needed to solve the concrete problem.
- Keep It Simple, Stupid (KISS): prefer the smallest direct implementation that preserves behavior and is easy to reason about.

## Testing Guidelines
- Framework: `pytest`; place tests under `tests/` and name files `test_*.py`.
- Keep tests fast and deterministic (CPU-first). If a feature needs external services (e.g., Redis/W&B), gate via env vars and provide safe fallbacks/mocks.

## Commit & Pull Request Guidelines
- Commit history follows Conventional Commits (`feat:`, `fix:`) and often includes an emoji.
  Example: `feat: 🎸 add new accelerator`
- PRs should include: a clear description, linked issue (if applicable), tests for behavior changes, and doc updates (`README.md`/`docs/`) when user-facing behavior changes.

## Agent-Specific Notes (Optional)
- Prefer `make` targets over ad-hoc commands and keep patches focused and reviewable.
