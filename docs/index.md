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
