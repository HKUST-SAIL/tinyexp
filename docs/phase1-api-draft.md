# Phase 1: API Draft

This document proposes a concrete API shape for the first implementation slice.

It should be read together with:

- [Design Philosophy](philosophy.md)
- [Phase 1: Minimal Helpers Plan](phase1-minimal-helpers.md)
- [Phase 1: File-by-File Implementation Plan](phase1-file-by-file-plan.md)

This is still a draft. The goal is to make the intended shape explicit before implementation expands further.

## Drafting Principles

The Phase 1 API should follow these rules:

- keep `TinyExp` small
- keep training and evaluation loops in examples
- expose shared capabilities through focused `XXXCfg` components
- make configuration override-friendly through Hydra
- keep execution explicit through method calls
- avoid introducing trainer-like control flow

## `TinyExp` Draft Surface

Phase 1 keeps the root experiment object intentionally small.

### Fields

Recommended experiment-level fields:

- `mode: str = "train"`
- `resume_from: str = ""`
- `output_root: str = "./output"`
- `exp_name: str = ...`

These fields describe the experiment as a whole rather than one isolated feature subsystem.

### Composed config components

The experiment object should continue to expose capability-specific config components, such as:

- `logger_cfg`
- `wandb_cfg`
- `checkpoint_cfg`

Other feature configs may be added later only if they earn their place.

### Methods

Recommended Phase 1 methods on `TinyExp`:

```python
def get_run_dir(self) -> str:
    ...

def dump_config(self, path: str | None = None) -> str:
    ...
```

These belong on `TinyExp` because they are experiment-scoped, not feature-scoped.

## `CheckpointCfg` Draft

Checkpointing is the main new shared capability in Phase 1.

It should follow the same cfg-driven pattern as logger and W&B integration:

- fields define behavior and defaults
- methods perform explicit actions only when called

### Draft fields

```python
@dataclass
class CheckpointCfg:
    last_ckpt_name: str = "last.ckpt"
    best_ckpt_name: str = "best.ckpt"
```

Phase 1 should keep this deliberately small.

### Draft methods

```python
def save_checkpoint(
    self,
    *,
    run_dir: str,
    name: str,
    model=None,
    optimizer=None,
    scheduler=None,
    epoch: int | None = None,
    global_step: int | None = None,
    best_metric: float | None = None,
    extra_state: dict | None = None,
) -> str:
    ...

def load_checkpoint(
    self,
    path: str,
    *,
    model=None,
    optimizer=None,
    scheduler=None,
    strict: bool = True,
    map_location=None,
) -> dict:
    ...
```

### Responsibilities

`CheckpointCfg` should:

- define default checkpoint filenames
- save a standard checkpoint structure
- load a standard checkpoint structure
- optionally restore state into provided model / optimizer / scheduler objects

### Non-responsibilities

`CheckpointCfg` should not decide:

- when to save
- whether to save best checkpoints
- which metric is considered best
- whether resume is automatic
- how many checkpoints to retain

Those remain example-level or user-level decisions.

## Draft Checkpoint Format

The checkpoint format should be simple and explicit.

Recommended structure:

```python
{
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "scheduler_state_dict": ...,
    "epoch": ...,
    "global_step": ...,
    "best_metric": ...,
    "meta": {
        "exp_name": ...,
        "exp_class": ...,
        "saved_at": ...,
    },
    ...
}
```

Notes:

- `optimizer_state_dict` is optional
- `scheduler_state_dict` is optional
- `meta` should stay lightweight
- `extra_state` can extend the structure without forcing premature abstraction

## Config Dump Draft

Configuration dumping should remain an experiment-level helper.

### Draft method

```python
def dump_config(self, path: str | None = None) -> str:
    ...
```

### Expected behavior

- default path is `<run_dir>/config.yaml`
- output reflects current configuration after Hydra overrides
- dump should be safe to call from examples
- distributed runs should avoid duplicate writes

## Run Directory Draft

Run directory behavior should remain simple in Phase 1.

### Draft methods

```python
def get_run_dir(self) -> str:
    ...
```

### Expected behavior

- `get_run_dir()` returns `os.path.join(self.output_root, self.exp_name)`
- methods that write files should create parent directories when needed

Phase 1 should not add timestamped run folders, version managers, or heavier run registry behavior.

## Example Usage Draft

Below is the intended style for examples after Phase 1.

### Logger setup

```python
run_dir = self.get_run_dir()
logger = self.logger_cfg.build_logger(
    save_dir=run_dir,
    distributed_rank=accelerator.rank,
)
self.dump_config()
```

### Explicit W&B usage

```python
if self.wandb_cfg.enable_wandb:
    wandb = self.wandb_cfg.build_wandb(
        accelerator=accelerator,
        project="Baselines",
        config=cfg_dict,
    )
```

### Explicit checkpoint load

```python
resume_state = None
if self.resume_from:
    resume_state = self.checkpoint_cfg.load_checkpoint(
        self.resume_from,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        map_location=accelerator.device,
    )
```

### Explicit checkpoint save

```python
self.checkpoint_cfg.save_checkpoint(
    run_dir=run_dir,
    name=self.checkpoint_cfg.last_ckpt_name,
    model=accelerator.unwrap_model(model),
    optimizer=optimizer,
    scheduler=scheduler,
    epoch=epoch,
    global_step=global_step,
    best_metric=best_metric,
)
```

### Explicit best checkpoint policy in example code

```python
if best_metric is None or val_metric > best_metric:
    best_metric = val_metric
    self.checkpoint_cfg.save_checkpoint(
        run_dir=run_dir,
        name=self.checkpoint_cfg.best_ckpt_name,
        model=accelerator.unwrap_model(model),
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        global_step=global_step,
        best_metric=best_metric,
    )
```

This is the intended balance:

- framework provides the capability
- example decides when to use it

## API Choices Deferred Beyond Phase 1

The following questions should stay open until the first implementation slice proves itself:

- whether metrics deserve their own `MetricCfg`
- whether config dumping should later move into a dedicated artifact cfg
- whether `CheckpointCfg` should move into its own module
- whether shared recipe base classes are worth introducing

These should not be over-designed before the first slice is working.

## Success Criteria

The Phase 1 API draft is successful if it leads to implementation that feels:

- small
- explicit
- override-friendly
- consistent with the existing `XXXCfg` style
- still close to plain PyTorch

That is the standard Phase 1 should be judged against.
