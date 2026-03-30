# Phase 1: Minimal Helpers Plan

This document describes the first implementation phase that follows TinyExp's design philosophy.

The goal is not to turn TinyExp into a trainer framework. The goal is to add a small set of reusable helpers that
remove repeated experiment plumbing while keeping training loops in user space.

For the broader principles behind these choices, see [Design Philosophy](philosophy.md).

## Why This Phase Exists

TinyExp already has a clear core direction:

- the experiment is the entrypoint
- configuration is explicit and override-friendly
- launch behavior should stay simple
- users should keep control of their training loop

What is still missing is a minimal layer for common experiment chores that many examples will otherwise repeat by hand.

This phase adds only thin helpers for those chores. It does not add a trainer, runtime, callback engine, or framework
owned lifecycle.

## Goals

Phase 1 should make experiments easier to run and maintain without changing TinyExp's character.

The goals are:

- keep plain PyTorch style intact
- reduce repeated setup code in examples
- improve reproducibility through lightweight artifacts
- make resume/eval workflows easier
- create a stable base for future examples and recipe-style inheritance

## Non-Goals

Phase 1 explicitly does not aim to add:

- a generic trainer abstraction
- a runtime layer that owns epoch or step flow
- a callback or hook engine
- automatic external tracker initialization
- a framework-wide best-model policy system
- a heavy experiment lifecycle API

If a feature starts to own the user workflow instead of helping it, it is out of scope for this phase.

## Minimal Additions to TinyExp

The base `TinyExp` class should remain small. This phase only proposes a few minimal additions.

### New fields

Recommended additions:

- `mode: str = "train"`
- `resume_from: str = ""`

These are intentionally minimal:

- `mode` provides a small, explicit switch for training, validation, and config help flows
- `resume_from` provides a standard path for loading a checkpoint

More policy-driven settings should stay in examples unless they prove broadly reusable.

### New helper methods

The following methods are the proposed Phase 1 surface area:

- `get_run_dir() -> str`
- `ensure_run_dir() -> str`
- `dump_config(path: str | None = None) -> str`
- `log_metrics(metrics: dict, *, step: int | None = None, epoch: int | None = None, filename: str = "metrics.jsonl") -> None`
- `save_checkpoint(...) -> str`
- `load_checkpoint(...) -> dict`
- `maybe_resume(...) -> dict | None`

These are helpers, not control-flow abstractions.

## Artifact Conventions

Phase 1 should establish simple, stable artifact conventions.

The recommended default run layout is:

- `output/<exp_name>/config.yaml`
- `output/<exp_name>/metrics.jsonl`
- `output/<exp_name>/last.ckpt`
- `output/<exp_name>/best.ckpt`
- `output/<exp_name>/log.txt`

This layout is intentionally straightforward. It improves usability and reproducibility without introducing a heavy run
management system.

## Helper Behavior

### Run directory helpers

`get_run_dir()` should return the default run directory for the current experiment.

`ensure_run_dir()` should create that directory if needed and return it.

These helpers should not introduce a large naming or versioning system in Phase 1.

### Config dumping

`dump_config()` should write the effective experiment configuration to YAML.

Expected behavior:

- default path is `<run_dir>/config.yaml`
- output reflects current config state after overrides
- writing should happen only from the main process when running distributed

### Metric logging

`log_metrics()` should append structured records to a local JSONL file.

Expected behavior:

- default file is `<run_dir>/metrics.jsonl`
- each record should include the provided metrics
- helper may also attach lightweight metadata such as timestamp, step, and epoch
- writing should happen only from the main process

This gives TinyExp a useful local record format without introducing a full tracker framework.

### Checkpoint helpers

`save_checkpoint()` and `load_checkpoint()` should provide a standard way to persist and recover experiment state.

Recommended checkpoint content:

- `model_state_dict`
- `optimizer_state_dict` when available
- `scheduler_state_dict` when available
- `epoch`
- `global_step`
- `best_metric`
- `meta`

Recommended metadata:

- `exp_name`
- `exp_class`
- `saved_at`

The helper should only standardize the storage format. It should not decide when checkpoints are written.

### Resume helper

`maybe_resume()` should be a thin convenience layer over `resume_from`.

Expected behavior:

- return `None` when `resume_from` is empty
- otherwise call `load_checkpoint()`
- return the loaded checkpoint state so the example can decide how to resume

This keeps resume logic explicit while reducing repeated boilerplate.

## Boundary Between TinyExp and Examples

This phase depends on keeping a strong boundary between the framework and examples.

### TinyExp should own

- configuration structure and override ergonomics
- launch integration
- thin artifact helpers
- small reusable utilities shared across many experiments

### Examples should own

- model construction
- data loading details
- the training loop
- evaluation logic
- when validation runs
- when checkpoints are saved
- what metric counts as best
- whether and when external integrations are initialized

This boundary is central to TinyExp's design.

## Example Migration Strategy

The first migration target should be `tinyexp/examples/mnist_exp.py`.

It is a good candidate because:

- it is small enough to change safely
- it already represents the intended user-facing workflow
- it can validate whether the helpers are actually reducing useful boilerplate

The migration should:

- keep the training loop inside the example
- replace repeated path/config writing code with helpers
- add checkpoint save/load through helpers
- add `mode=val` using `resume_from`

Only after this works well should TinyExp consider extracting a recipe-style base class from examples.

## Testing Plan

Phase 1 should be backed by lightweight tests.

Recommended test coverage:

- unit tests for run directory creation
- unit tests for config dumping
- unit tests for metric logging
- unit tests for checkpoint save/load
- unit tests for `maybe_resume()`
- a small integration test for `mode=val`

The tests should stay CPU-first and deterministic.

## Implementation Order

Recommended implementation order:

1. add run directory helpers
2. add config dumping
3. add metric logging
4. add checkpoint save/load
5. add `maybe_resume()`
6. migrate `mnist_exp.py`
7. add `mode=val`
8. add tests

This order keeps each change small and easy to validate.

## Success Criteria

Phase 1 is successful if TinyExp can do all of the following while still feeling light:

- keep experiments centered around one explicit entrypoint
- preserve user-owned training loops
- save config and local metrics in a standard way
- save and resume checkpoints with minimal boilerplate
- support a simple validation flow from a checkpoint

In short, Phase 1 should make TinyExp more practical without making it more framework-heavy.
