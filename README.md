[![Main](https://github.com/zengarden/tinyexp/actions/workflows/main.yml/badge.svg?branch=main)](https://github.com/zengarden/tinyexp/actions/workflows/main.yml)
[![codecov](https://codecov.io/gh/zengarden/tinyexp/branch/main/graph/badge.svg)](https://codecov.io/gh/zengarden/tinyexp)

# TinyExp

Simple experiment management for PyTorch.

TinyExp is built around one idea:
your configured experiment is your entrypoint.

<img src="docs/assets/tinyexp-demo-short.min.gif" alt="TinyExp demo" width="480"/>

Instead of splitting config, launcher, and execution across many files, TinyExp keeps them together in one experiment
definition so iteration stays fast and predictable.

What you get in practice:

- Experiment-centered configuration (Hydra/OmegaConf)
- CLI overrides without rewriting code
- Keep your training loop close to plain PyTorch
- Run the same experiment definition from local debug to distributed launch

## Why TinyExp

TinyExp focuses on simple, maintainable experiment management:

- Your experiment code stays readable.
- Your config stays structured and easy to override.
- Your execution path stays consistent as experiments grow.

## Design Philosophy

TinyExp is intentionally light.

It is not trying to be a heavy trainer framework that owns your epoch loop, callback system, or full runtime
lifecycle. Instead, it focuses on a smaller goal:

- keep the experiment itself as the main entrypoint
- keep the training loop in user space
- make configuration and launch behavior explicit
- expose shared capabilities through focused `XXXCfg` components
- provide thin helpers rather than framework-owned control flow
- treat examples as reusable recipes, not just demos

In short, TinyExp should help you write less experiment plumbing, not less experiment logic.

For a longer explanation, see [`docs/philosophy.md`](docs/philosophy.md).

## Quick Start (1 Minute)

### Option A: Install with pip and use import-based entrypoint

```bash
pip install tinyexp
```

```python
from tinyexp import store_and_run_exp
from tinyexp.examples.mnist_exp import Exp

store_and_run_exp(Exp)
```

```bash
python your_exp.py
python your_exp.py dataloader_cfg.train_batch_size_per_device=16
```

### Option B: Run the bundled example from source (for development)

```bash
git clone https://github.com/HKUST-SAIL/tinyexp.git
cd tinyexp
make install
uv run python tinyexp/examples/mnist_exp.py
```

## Common Commands

Run MNIST with config override:

```bash
uv run python tinyexp/examples/mnist_exp.py dataloader_cfg.train_batch_size_per_device=16
```

Print all available configs:

```bash
uv run python tinyexp/examples/mnist_exp.py mode=help
```

Print all configs plus your overrides:

```bash
uv run python tinyexp/examples/mnist_exp.py mode=help dataloader_cfg.train_batch_size_per_device=16
```

Worker processes are launched according to the `launcher` config: `launcher=ray` spawns Ray workers from the driver
process, while `launcher=mp` runs in the current process. The bundled examples default to `launcher=ray`, so when
using an external multi-process launcher such as `torchrun` or `accelerate launch`, override it explicitly:

```bash
uv run torchrun --standalone --nproc-per-node 2 tinyexp/examples/mnist_exp.py launcher=mp
```

Run a command with TinyExp launch helpers after installing the package:

```bash
tinyexp-run-with-redis -- python your_exp.py redis_cfg.redis_cache_enabled=true
tinyexp-run-with-ray-cluster --node-count 2 --head-addr 10.0.0.1 -- python your_exp.py
```

## Example Experiments

- MNIST baseline: [`tinyexp/examples/mnist_exp.py`](tinyexp/examples/mnist_exp.py)
- ImageNet ResNet-50: [`tinyexp/examples/resnet_exp.py`](tinyexp/examples/resnet_exp.py)

For ImageNet example:

```bash
export IMAGENET_HOME=/path/to/imagenet
uv run python tinyexp/examples/resnet_exp.py
```

## How It Works

1. Define an experiment class by inheriting `TinyExp`.
2. Keep model/data/optimizer/scheduler config in nested dataclasses.
3. Implement `run()` (and train/eval helpers) in the same experiment definition.
4. Launch the script and override config from CLI when needed.

This gives you a single, explicit place to manage experiment behavior.

## Development

Install environment and hooks:

```bash
make install
```

Run checks:

```bash
make check
```

Run tests:

```bash
make test
```

Build docs:

```bash
make docs-test
```

Build package:

```bash
make build
```

Release:

```bash
make release VERSION=0.0.4
```

## Documentation

- Docs site: https://zengarden.github.io/TinyExp/
- API/module overview: [`docs/modules.md`](docs/modules.md)

## Contributing

PRs and issues are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

MIT License. See [`LICENSE`](LICENSE).
