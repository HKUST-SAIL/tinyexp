.PHONY: install
install: ## Install core dependencies while preserving machine-selected optional packages
	@echo "🚀 Creating virtual environment using uv"
	@uv sync --locked --inexact
	@uv run pre-commit install

.PHONY: install-pytorch
install-pytorch: ## Install the default PyPI PyTorch, TorchVision, and Accelerate builds
	@uv sync --locked --extra pytorch

.PHONY: install-without-pytorch
install-without-pytorch: ## Install core dependencies while preserving machine-selected accelerator packages
	@uv sync --locked --no-extra pytorch --inexact

.PHONY: check
check: ## Run code quality tools.
	@echo "🚀 Checking lock file consistency with 'pyproject.toml'"
	@uv lock --locked
	@echo "🚀 Linting code: Running pre-commit"
	@uv run pre-commit run -a
	# @echo "🚀 Static type checking: Running mypy"
	# @uv run mypy

.PHONY: test
test: ## Test the code with pytest
	@echo "🚀 Testing code: Running pytest"
	@RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 uv run python -m pytest --cov --cov-config=pyproject.toml --cov-report=xml

.PHONY: build
build: clean-build ## Build wheel file
	@echo "🚀 Creating wheel file"
	@uvx --from build pyproject-build --installer uv

.PHONY: clean-build
clean-build: ## Clean build artifacts
	@echo "🚀 Removing build artifacts"
	@uv run python -c "import shutil; import os; shutil.rmtree('dist') if os.path.exists('dist') else None"

.PHONY: publish
publish: ## Publish a release to PyPI.
	@echo "🚀 Publishing."
	@uvx twine upload --repository pypi dist/*
	# @uvx twine upload --repository-url https://upload.pypi.org/legacy/ dist/*

.PHONY: build-and-publish
build-and-publish: build publish ## Build and publish.

.PHONY: release
release: ## Release with checks, tests, build, publish, commit, tag, and push. Usage: make release VERSION=0.0.4
	@if [ -z "$(VERSION)" ]; then echo "VERSION is required, e.g. make release VERSION=0.0.4"; exit 1; fi
	@./scripts/release.sh $(VERSION)

.PHONY: docs-test
docs-test: ## Test if documentation can be built without warnings or errors
	@uv run mkdocs build -s

.PHONY: docs
docs: ## Build and serve the documentation
	@uv run mkdocs serve

.PHONY: help
help:
	@uv run python -c "import re; \
	[[print(f'\033[36m{m[0]:<20}\033[0m {m[1]}') for m in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open(makefile).read(), re.M)] for makefile in ('$(MAKEFILE_LIST)').strip().split()]"

.DEFAULT_GOAL := help
