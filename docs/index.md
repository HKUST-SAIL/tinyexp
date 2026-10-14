# TinyExp

A minimalist Python project for deep learning experiment management.

TinyExp keeps one idea at the center:
your configured experiment is your entrypoint.

Instead of splitting config, launcher, and execution across many files, TinyExp keeps them together in one experiment
definition so iteration stays fast and predictable.

## What You Get

- Experiment-centered configuration with Hydra/OmegaConf
- CLI overrides without rewriting code
- Training loops that stay close to plain PyTorch
- The same experiment definition from local debug to distributed launch

## Running Experiments

TinyExp supports plain Python, TorchRun, Accelerate launch, and static Ray cluster workflows. These commands use
different process owners and must be paired with the correct TinyExp `launcher` setting and accelerator. See
[Running Modes and Environment Requirements](running-modes.md) for the mode matrix, setup requirements, and verified
command patterns.

## Documentation for Coding Agents

Work that is done by a coding agent reads the docs instead of the source. Three entry points are kept accurate for
that use:

- `AGENTS.md` at the repository root: a task-to-file index for work inside a checkout;
- `SKILL.md`: an index of the same contracts for an installed package, where `AGENTS.md` and `docs/` are absent;
- the run header at the top of every example module (`Modes`, `Run`, `Env`, `Needs`, `Out`), which lists that
  example's commands, data and hardware. Print it with
  `python -c "import tinyexp.examples.mnist_exp as m; print(m.__doc__)"` instead of reading the file.

Their commands are checked by `tests/examples/test_examples_help_smoke.py`: every documented command must be legal
for the config it targets, and every `Modes` value in a run header must be handled in `run()`. The check is static,
so it catches stale option names, not stale advice. Known gaps, deliberately left open:

- `RayCfg`, `RedisCfg` and `CheckpointCfg` docstrings do not list every field default; `mode=help` prints them.
- `tinyexp-run-with-redis` takes its ports and memory limit from `redis_cfg.*` overrides on the command after `--`,
  and appends the resolved connection settings to that command.
- With `launcher=ray` the driver starts and stops Redis by itself, so `tinyexp-run-with-redis` refuses such a
  command; with `launcher=mp` the wrapper owns Redis.
- The MNIST example always trains 3 epochs (not configurable); shorten it with a larger
  `dataloader_cfg.train_batch_size_per_device`.

## Design Philosophy

TinyExp is intentionally light.

It is not trying to be a heavy trainer framework that owns your epoch loop, callback system, or full runtime
lifecycle. Instead, it focuses on a smaller and more explicit goal:

- keep the experiment itself as the main entrypoint
- keep the training loop in user space
- make configuration and launch behavior explicit
- expose shared capabilities through focused `XXXCfg` components
- provide thin helpers instead of framework-owned control flow
- treat examples as reusable recipes, not just demos

In short, TinyExp should help you write less experiment plumbing, not less experiment logic.

For the longer version, see [Design Philosophy](philosophy.md).

## Features

- 🚀 One-click experiment launch: The file you edit becomes the entrypoint to your experiment.
- 🔄 Config-driven experiment management with Hydra.
- 🧩 Thin helpers without taking over your training loop.
- 🧪 Examples that can serve as reusable experiment recipes.
