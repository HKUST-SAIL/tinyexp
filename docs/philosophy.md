# Design Philosophy

TinyExp is intentionally small.

It is not trying to become a heavy training framework, a trainer abstraction, or a system that takes over the
user's training loop. Its goal is much narrower and, for many research and iteration workflows, much more useful:
make experiment code easy to define, easy to launch, and easy to evolve without hiding plain PyTorch.

## What TinyExp Is

TinyExp is best understood as an experiment entry framework with a few lightweight helpers.

It focuses on:

- keeping the experiment definition as the main entrypoint
- making configuration explicit and easy to override
- expressing shared features through focused `XXXCfg` components
- supporting multiple launch styles without changing experiment code too much
- keeping user code close to normal PyTorch
- reducing repeated experiment "plumbing" without owning the full training lifecycle

In practice, TinyExp wants the file you edit to remain the file you run.

## What TinyExp Is Not

TinyExp is intentionally not designed to be:

- a general-purpose trainer framework
- a runtime system that owns epoch/step control flow
- a callback-heavy abstraction layer
- a framework that forces users into a single lifecycle or DSL
- a system that hides the actual training loop behind too many layers

If a feature would make experiments feel less like plain PyTorch and more like framework ceremony, it is usually a
bad fit for TinyExp.

## Core Principles

### 1. The experiment is the entrypoint

The experiment definition should be the center of the workflow.

Users should not need to spread one experiment across many disconnected files just to launch, configure, and run it.
The experiment class should stay readable, local, and easy to reason about.

### 2. Explicit is better than implicit

TinyExp prefers explicit calls over hidden side effects.

For example, integrations with external systems such as W&B should remain explicit. A config object can expose the
ability to build an integration, but the user should still decide when to call it.

This same rule applies to TinyExp features more broadly:

- configuration should live in a small `XXXCfg` class
- config fields should be override-friendly through Hydra
- behavior should only run when the user explicitly calls a method on that config object

This keeps configuration and execution separate while still letting execution be part of the experiment structure.

### 3. Prefer `XXXCfg` components for shared features

When TinyExp grows, new capabilities should usually be introduced as focused config components rather than as many
top-level methods on `TinyExp`.

For example, a feature is often a better fit as:

- `logger_cfg.build_logger(...)`
- `wandb_cfg.build_wandb(...)`
- `checkpoint_cfg.save_checkpoint(...)`

than as a large collection of flat framework methods.

This pattern keeps a feature's configuration and execution close together:

- fields describe the feature and can be overridden through Hydra
- methods execute behavior only when the user explicitly calls them
- `TinyExp` itself stays smaller and easier to understand

### 4. Keep the training loop in user space

The training loop is often the most task-specific part of an experiment. TinyExp should not rush to abstract it into
a universal trainer.

Users should be able to:

- write their own training loop
- define their own evaluation logic
- control when to validate, log, save, or resume
- stay in plain PyTorch as much as possible

### 5. Helpers are good; control frameworks are not

TinyExp should provide thin, reusable helpers for common experiment chores, such as:

- output directory setup
- config dumping
- lightweight metric logging
- checkpoint save/load helpers exposed through focused config components
- launcher compatibility

These helpers reduce repeated boilerplate without dictating how the user structures training.

### 6. Examples are recipes, not just demos

Examples in TinyExp are not only meant to showcase features. They should also serve as reusable recipes and
inheritance-friendly templates.

That means examples should remain understandable and useful as starting points for real projects. When common logic
emerges across multiple examples, it may be worth extracting a small helper or a recipe base class. But that logic
should only move into the framework when it is broadly useful and still keeps the system light.

### 7. Framework-level additions must earn their place

A good question for any new feature is:

Does this reduce repeated experiment plumbing while preserving user control?

If yes, it may belong in TinyExp.

If it starts to own the user workflow, hide core control flow, or push the project toward a heavy trainer-style
architecture, it probably does not belong in TinyExp.

## Recommended Boundary

### TinyExp should own

- configuration structure and CLI overrides
- experiment entry and launch ergonomics
- lightweight utilities shared across many experiments
- small `XXXCfg` components for shared capabilities
- minimal helpers that do not take over control flow

### Examples or user experiments should own

- model construction details
- training and evaluation loops
- task-specific metrics
- validation timing
- checkpointing policy such as what counts as "best"
- external integration timing and usage

This boundary keeps TinyExp small while still making it genuinely useful.

## Design Direction for Future Development

When extending TinyExp, prefer:

- small `XXXCfg` components over large abstractions
- explicit calls over automatic behavior
- recipe-style examples over framework-owned trainers
- local clarity over generic indirection
- composable building blocks over lifecycle machinery

In short:

TinyExp should help users write less experiment plumbing, not less experiment logic.
