[![Main](https://github.com/zengarden/tinyexp/actions/workflows/main.yml/badge.svg?branch=main)](https://github.com/zengarden/tinyexp/actions/workflows/main.yml)
[![codecov](https://codecov.io/gh/zengarden/tinyexp/branch/main/graph/badge.svg)](https://codecov.io/gh/zengarden/tinyexp)

# TinyExp

Simple experiment management for PyTorch.

TinyExp is built around one idea:
your configured experiment is your entrypoint.

<img src="docs/assets/tinyexp-demo-short.min.gif" alt="Run a TinyExp experiment and override its configuration directly from the terminal" width="720"/>

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

### Option A: Install with pip and run the bundled example

```bash
pip install "tinyexp[pytorch]"
```

```bash
python -m tinyexp.examples.mnist_exp

# or run with override config
python -m tinyexp.examples.mnist_exp dataloader_cfg.train_batch_size_per_device=16
```

### Option B: Run the bundled example from source (for development)

```bash
git clone https://github.com/HKUST-SAIL/tinyexp.git
cd tinyexp
make install-pytorch
source .venv/bin/activate
python -m tinyexp.examples.mnist_exp
```

## Common Commands

The commands below assume that the environment containing TinyExp is active. For a source checkout, run
`source .venv/bin/activate` first.

Run MNIST with config override:

```bash
python tinyexp/examples/mnist_exp.py dataloader_cfg.train_batch_size_per_device=16
```

Print all available configs:

```bash
python tinyexp/examples/mnist_exp.py mode=help
```

Print all configs plus your overrides:

```bash
python tinyexp/examples/mnist_exp.py mode=help dataloader_cfg.train_batch_size_per_device=16
```

Worker processes are selected by both the command and TinyExp's `launcher` config. The base `TinyExp` class defaults to
`launcher=mp`, while the bundled examples default to `launcher=ray`.

| Run style | Command owner | TinyExp launcher |
| --- | --- | --- |
| Plain Python, direct process | `python` | `launcher=mp` |
| Plain Python, local Ray workers | TinyExp and Ray | `launcher=ray` |
| TorchRun | `torchrun` | `launcher=mp` |
| Accelerate launch | `accelerate launch` | `launcher=mp` |
| Static Ray cluster | `tinyexp-run-with-ray-cluster` | `launcher=ray` |

For `launcher=ray`, using the active environment's `python` is intentional. Ray detects a driver launched through
`uv run` and, by default, creates a `uv` runtime environment for its workers instead of reusing the already installed
environment. TinyExp's examples and static cluster helper assume that every participating node already has the
required environment. If a surrounding tool requires `uv run`, disable Ray's automatic `uv` runtime environment for
that command:

```bash
RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 uv run python tinyexp/examples/mnist_exp.py
```

This behavior is determined by the launch command, not by whether TinyExp was installed with `pip` or `uv`.
`pip install "tinyexp[pytorch]"` followed by `python your_exp.py` does not need this environment variable.

`torchrun` and `accelerate launch` create processes externally, so bundled examples must override `launcher=mp`:

```bash
torchrun \
  --nnodes 1 \
  --node-rank 0 \
  --nproc-per-node 2 \
  --master-addr 127.0.0.1 \
  --master-port 29500 \
  tinyexp/examples/mnist_exp.py launcher=mp
accelerate launch --cpu --num-processes 1 -m tinyexp.examples.pi_exp launcher=mp
```

A static Ray cluster must be started on every node with the same node count and head address, and a unique node rank.
The experiment command runs on node rank 0 and should use `launcher=ray`:

```bash
tinyexp-run-with-ray-cluster \
  --node-count 2 \
  --node-rank 0 \
  --head-addr 10.0.0.1 \
  --ray-port 6380 \
  -- \
  python your_exp.py launcher=ray
```

See [Running Modes and Environment Requirements](docs/running-modes.md) for the complete Python, PyTorch, CUDA/NCCL,
Accelerate, Ray multi-node, network, data, Redis, and W&B requirements.

The `mode` config selects what to execute: `train`, `val`, `run`, or `help`. Ray worker resources are explicit: set
`ray_cfg.ray_num_cpus_per_worker` and `ray_cfg.ray_num_gpus_per_worker` for the resources required by one worker.
The fields can be overridden in the experiment's nested `RayCfg` or from the command line.

Run a command with TinyExp's Redis helper after installing the package:

```bash
tinyexp-run-with-redis -- python your_exp.py redis_cfg.redis_cache_enabled=true
```

`tinyexp-run-with-redis` owns and stops only the Redis processes it starts. If a configured port is already served by
another Redis process, startup fails without shutting down or taking ownership of that server. Connect to externally
managed Redis directly through `redis_cfg` instead of wrapping the command with `tinyexp-run-with-redis`.

For multi-node training, the helper's Redis lifecycle follows the local command: each wrapper stops the Redis
resources it owns as soon as its child exits. If one node fails, the whole distributed training job is expected to
fail and restart; the helper does not keep Redis alive for a global finish barrier or implement heartbeat/lease-based
failure coordination. The external launcher or supervisor owns whole-job restart and termination.

## Example Experiments

- MNIST baseline: [`tinyexp/examples/mnist_exp.py`](tinyexp/examples/mnist_exp.py)
- ImageNet ResNet-50: [`tinyexp/examples/resnet_exp.py`](tinyexp/examples/resnet_exp.py)
- Distributed Monte Carlo pi (non-DL, `mode=run`): [`tinyexp/examples/pi_exp.py`](tinyexp/examples/pi_exp.py)

For ImageNet example:

```bash
export IMAGENET_HOME=/path/to/imagenet
python tinyexp/examples/resnet_exp.py
```

For the pi example (Ray workers all-reduce their sample counts, no dataloader involved):

```bash
python -m tinyexp.examples.pi_exp pi_cfg.total_samples=100000000 ray_cfg.ray_num_worker=4
```

## How It Works

1. Define an experiment class by inheriting `TinyExp`.
2. Keep model/data/optimizer/scheduler config in nested dataclasses.
3. Implement `run()` (and train/eval helpers) in the same experiment definition.
4. Launch the script and override config from CLI when needed.

This gives you a single, explicit place to manage experiment behavior. Training helpers can depend on
`tinyexp.tiny_engine.accelerator.AcceleratorProtocol`, so CPU, DDP, and Hugging Face Accelerate backends expose the same
model preparation, reduction, synchronization, and cleanup methods.

## Development

Install the core environment and hooks:

```bash
make install
```

`make install` installs the core environment and hooks without selecting or removing optional accelerator packages. For the default PyPI stack, run `make install-pytorch` before `make test`. On a machine with a preselected CUDA, ROCm, or vendor PyTorch build, `make install` (or the more explicit `make install-without-pytorch`) preserves that environment; install `torch`, `torchvision`, and `accelerate` together according to that machine's package index/backend. Because the PyTorch packages are optional, `uv run` does not remove extraneous packages by default and no repeated `--no-sync` flag is needed. Avoid `uv sync` without `--inexact` in that environment. For `launcher=ray`, activate the environment and use its `python` as described above instead of relying on Ray's automatic `uv` runtime environment.

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
- Running modes and environment requirements: [`docs/running-modes.md`](docs/running-modes.md)
- API/module overview: [`docs/modules.md`](docs/modules.md)

## Contributing

PRs and issues are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

MIT License. See [`LICENSE`](LICENSE).
