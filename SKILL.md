---
name: tinyexp-experiments
description: Write, configure and launch experiments with the TinyExp library (a dataclass Exp per file, CLI overrides, CPU/DDP/FSDP training, Ray workers, torchrun/accelerate). Use when the task involves tinyexp or a class derived from TinyExp.
---

# TinyExp experiments

An experiment is one `@dataclass class Exp(TinyExp, <mixins>)` whose `run()` holds the logic; every config field is a
CLI override (`key=value`). The installed package documents itself, so read docstrings instead of source:
`python -c "import tinyexp; print(tinyexp.__doc__)"`, or for a class
`python -c "from tinyexp.exp_mixins import RayCfgMixin as m; print(m.__doc__)"`.

| Need | Print the `__doc__` of |
| --- | --- |
| New experiment: minimal example, fields, `launcher` pairing | `tinyexp` |
| How overrides are merged and reported | `tinyexp.TinyExp.set_cfg` |
| Training loop API; picking an accelerator | `tinyexp.tiny_engine.accelerator.base_accelerator`; `tinyexp.tiny_engine.accelerator` |
| Code running on Ray workers (minimal example); worker count and resources | `tinyexp.exp_mixins.RayCfgMixin`; `tinyexp.exp_mixins.RayCfgMixin.RayCfg` |
| Checkpoint save / load / resume / eval | `tinyexp.exp_mixins.CheckpointCfgMixin.CheckpointCfg` |
| Logger, W&B, Redis cache mixins | `tinyexp.exp_mixins.basic_mixins` (table, then the nested `XXXCfg`) |
| Infinite sampler, fake dataloader, EMA; errors | `tinyexp.dataset`, `tinyexp.utils`; `tinyexp.exceptions` |
| A bundled example: modes, commands, data, hardware | `tinyexp.examples.<name>` (run header) |

Bundled examples: `mnist_exp` (template for a training experiment: CPU or DDP, checkpoints, resume), `pi_exp`
(non-deep-learning distributed job, fastest smoke test), `resnet_exp` (ImageNet with DDP, FSDP or HF Accelerate and
a Redis cache), `vit_tp_exp` (tensor parallelism, TP-vs-DDP benchmark).

Easy to miss:

- Check a config change before running: `mode=help` prints the resolved config and the overridden values, then
  exits. It checks key names and types, not values.
- `torchrun` and `accelerate launch` need `launcher=mp`; plain `python` takes `mp` (one process) or `ray` (TinyExp
  starts Ray workers); `tinyexp-run-with-ray-cluster` (multi-node, see `--help`) needs `launcher=ray` and runs
  `ray stop --force` on each node, so do not use it where an existing Ray runtime must survive.
- With `launcher=ray`, start with the environment's `python -m package.module`, not `uv run` (or set
  `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0`).
- Install with `pip install "tinyexp[pytorch]"`; for a CUDA/ROCm-specific PyTorch, install `tinyexp` without the
  extra, then `torch`, `torchvision` and `accelerate` from that backend.

```bash
python -m tinyexp.examples.pi_exp mode=help pi_cfg.total_samples=100000
python -m tinyexp.examples.pi_exp launcher=mp pi_cfg.total_samples=100000
torchrun --nproc-per-node=2 -m tinyexp.examples.mnist_exp launcher=mp accelerator_cfg.accelerator=ddp
tinyexp-run-with-ray-cluster --node-count=2 --node-rank=0 --head-addr=10.0.0.1 --ray-port=6380 -- \
  python -m tinyexp.examples.pi_exp launcher=ray
```
