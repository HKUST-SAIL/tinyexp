# Phase 1: File-by-File Implementation Plan

This document turns the Phase 1 helper direction into a file-by-file implementation plan.

It follows the same constraints described in:

- [Design Philosophy](philosophy.md)
- [Phase 1: Minimal Helpers Plan](phase1-minimal-helpers.md)

The key design rule is unchanged:

- shared capabilities should usually be exposed through focused `XXXCfg` classes
- those config fields should be Hydra-override-friendly
- behavior should execute only when the user explicitly calls a method

This plan is intentionally conservative. It aims to establish one clean first slice, not to solve every future need in
one pass.

## Phase 1 Scope

The first implementation slice should focus on:

- a stable run directory helper
- explicit config dumping
- a new `CheckpointCfg`
- `mode=val` support through explicit checkpoint loading
- a migration of the MNIST example to validate the design

This phase should not introduce:

- a trainer abstraction
- a runtime layer
- callback systems
- automatic checkpoint policy
- automatic resume behavior

## Files to Change

The recommended Phase 1 file set is intentionally small:

- `tinyexp/__init__.py`
- `tinyexp/examples/mnist_exp.py`
- `tests/test_tinyexp.py` or a new artifact-focused test file
- a new checkpoint-focused test file
- optionally one example-level integration test for validation mode

The ResNet example should not be part of the first slice.

## 1. `tinyexp/__init__.py`

This is the main design anchor for Phase 1.

### Why this file changes

This file already defines:

- `TinyExp`
- `LoggerCfg`
- `WandbCfg`
- `RedisCfgMixin`

That makes it the natural place to reinforce the cfg-driven model and add the first checkpoint component.

### What should change

#### Keep `TinyExp` small

`TinyExp` should continue to be the root experiment object, but it should not turn into a feature sink.

It should remain responsible for:

- experiment-level config structure
- launcher-facing fields
- a small number of experiment-wide helpers
- composition of shared `XXXCfg` components

#### Add only minimal experiment-wide fields

Recommended additions or clarifications:

- `mode: str = "train"`
- `resume_from: str = ""`

These belong at the experiment level because they describe run intent rather than one isolated feature subsystem.

#### Add only minimal experiment-wide helpers

Recommended methods on `TinyExp`:

- `get_run_dir() -> str`
- `ensure_run_dir() -> str`
- `dump_config(path: str | None = None) -> str`

These are good fits for `TinyExp` because they are experiment-scoped rather than belonging to a single feature config.

### What should not be added here

Avoid adding many feature-specific top-level methods such as:

- `save_checkpoint(...)`
- `load_checkpoint(...)`
- `maybe_resume(...)`

Those are better expressed through a focused config component.

## 2. Add `CheckpointCfg` in `tinyexp/__init__.py`

For the first slice, `CheckpointCfg` can live in `tinyexp/__init__.py` alongside `LoggerCfg` and `WandbCfg`.

This keeps the initial implementation simple and consistent with the current project structure.

If it grows later, it can be split into a dedicated module.

### Why `CheckpointCfg`

Checkpointing fits the cfg-driven TinyExp pattern well:

- filenames and related defaults are configuration
- save/load methods are explicit actions
- users choose when to call those methods

This is more aligned with TinyExp's style than adding many checkpoint methods directly to `TinyExp`.

### Recommended `CheckpointCfg` scope

Fields:

- `last_ckpt_name: str = "last.ckpt"`
- `best_ckpt_name: str = "best.ckpt"`

Methods:

- `save_checkpoint(...) -> str`
- `load_checkpoint(...) -> dict`

### Recommended responsibilities

`CheckpointCfg` should handle:

- default checkpoint filenames
- run-dir-relative checkpoint path generation when useful
- standard save format
- standard load behavior
- optional loading into model / optimizer / scheduler objects

### What `CheckpointCfg` should not own

Do not put policy into `CheckpointCfg`, including:

- when to save
- whether to save best checkpoints
- how to compare best metrics
- save frequency
- retention policies
- automatic resume behavior

Those decisions belong in the example or user code.

## 3. `tinyexp/examples/mnist_exp.py`

This file should be the first real migration target.

### Why this file changes first

The MNIST example is:

- small enough to change safely
- representative of the intended user workflow
- a good way to validate whether the cfg-driven helper design actually reduces useful boilerplate

### What should change

#### `run()` should adopt experiment-wide helpers

Expected updates:

- call `self.ensure_run_dir()`
- build the logger using `self.logger_cfg.build_logger(...)`
- call `self.dump_config()`
- branch on `self.mode`

#### training should remain explicit

The training loop should stay in the example.

What should change is only the repeated plumbing:

- explicit checkpoint loading when `self.resume_from` is set
- explicit calls to `self.checkpoint_cfg.save_checkpoint(...)`
- explicit best-checkpoint save logic, still decided by the example

#### validation should also stay explicit

For `mode=val`, the example should:

- require a meaningful `resume_from`
- explicitly call `self.checkpoint_cfg.load_checkpoint(...)`
- run evaluation logic in example code

The example remains responsible for evaluation semantics.

### What should not change

Do not try to extract:

- a trainer
- a recipe base class
- generic evaluation policy

Those can be revisited later only if repeated patterns clearly emerge.

## 4. `tinyexp/examples/resnet_exp.py`

This file should not be part of the first implementation slice.

### Why it should wait

The ResNet example includes additional concerns:

- DDP usage
- ImageNet-specific data loading
- Redis-backed caching
- a more complex training setup

It is not the right place to define the first minimal checkpoint and artifact API.

### Recommended Phase 1 stance

- leave it unchanged
- only revisit after the MNIST migration proves the shape of the APIs

## 5. Tests

Phase 1 needs lightweight but meaningful coverage.

### Recommended test additions

#### Artifact tests

Add or expand tests for:

- `get_run_dir()`
- `ensure_run_dir()`
- `dump_config()`

These can live in:

- `tests/test_tinyexp.py`
- or a new `tests/test_tinyexp_artifacts.py`

#### Checkpoint tests

Add a dedicated checkpoint test file, for example:

- `tests/test_tinyexp_checkpoint_cfg.py`

Cover:

- save model-only checkpoint
- save/load with optimizer and scheduler
- standard metadata presence
- correct state restoration

#### Example-level validation test

Add one small integration-style test for:

- `mode=val`
- loading a checkpoint through `resume_from`

This should stay CPU-first and deterministic.

## 6. Files Not Needed in Phase 1

The following files or modules do not need changes in the first slice:

- `tinyexp/utils/ray_utils.py`
- `tinyexp/tiny_engine/accelerator/*`
- `tinyexp/examples/resnet_exp.py`
- Redis-related utilities

This is important.

The first slice should validate the cfg-driven artifact pattern, not broaden the implementation surface.

## Recommended Implementation Order

The order below minimizes risk and keeps the design easy to validate.

1. update `tinyexp/__init__.py` with:
   - `mode`
   - `resume_from`
   - `get_run_dir()`
   - `ensure_run_dir()`
   - `dump_config()`
   - `CheckpointCfg`
2. add checkpoint-focused tests
3. migrate `tinyexp/examples/mnist_exp.py`
4. add validation-mode test coverage
5. only then decide whether any further cfg component is worth introducing

## Stop Point for Phase 1

Phase 1 should stop once the following are true:

- experiment-level artifact basics are available
- checkpointing is exposed through `checkpoint_cfg`
- the MNIST example uses the new pattern successfully
- validation from checkpoint works
- the project still feels light and explicit

That stop point matters.

The goal of Phase 1 is not to fully design TinyExp's long-term helper ecosystem. The goal is to establish one clean,
cfg-driven example of how shared capabilities should be added without drifting toward a trainer framework.
