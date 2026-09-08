---
name: tinyexp-experiments
description: Build, run, debug, and scale PyTorch or general experiments with this TinyExp repository. Use for new experiment files, configuration overrides, CPU/GPU training, Ray jobs, and distributed launches.
---

# TinyExp experiments

Use TinyExp when the experiment should keep its configuration and execution in one Python entrypoint. Read the nearest example in `tinyexp/examples/` before adding a new abstraction.

## Core pattern

1. Define `@dataclass class Exp(TinyExp, ...)`.
2. Put model, data, optimizer, scheduler, and runtime settings in nested dataclasses with `field(default_factory=...)`.
3. Implement `run()` and use an accelerator implementing `AcceleratorProtocol` (`prepare`, `backward`, `reduce_*`, `wait_for_everyone`, `destroy`).
4. End the file with `if __name__ == "__main__": store_and_run_exp(Exp)`.

Keep training logic in the experiment. Reuse mixins such as `RayCfgMixin`, `CheckpointCfgMixin`, `LoggerCfgMixin`, `WandbCfgMixin`, and `RedisCfgMixin` only when the feature is needed. Save outputs below `output_root` (default `./output`).

## Configuration and launch

Hydra accepts dotted CLI overrides without editing code. First inspect a new experiment:

```bash
python -m tinyexp.examples.mnist_exp mode=help
```

`launcher` selects TinyExp's process owner: `mp` runs `Exp.run()` in the current process; `ray` makes the driver create Ray workers from `ray_cfg`. `torchrun`, `accelerate launch`, and `tinyexp-run-with-ray-cluster` are external launchers and must be paired with `launcher=mp`, `launcher=mp`, and `launcher=ray` respectively.

Typical commands:

```bash
# Local CPU/debug, one process
python -m tinyexp.examples.pi_exp launcher=mp pi_cfg.total_samples=100000

# Local Ray workers (CPU in this example)
python -m tinyexp.examples.pi_exp ray_cfg.ray_num_worker=4 pi_cfg.total_samples=1000000

# Single-node GPU/DDP (one process per visible GPU)
torchrun --nproc-per-node=2 -m tinyexp.examples.resnet_exp \
  launcher=mp accelerator_cfg.accelerator=ddp redis_cfg.redis_cache_enabled=false

# Accelerate owns the processes; TinyExp remains in mp mode
accelerate launch --cpu --num-processes=1 -m tinyexp.examples.pi_exp \
  launcher=mp pi_cfg.total_samples=100000

# Existing multi-node Ray cluster
tinyexp-run-with-ray-cluster --node-count=2 --node-rank=0 \
  --head-addr=10.0.0.1 --ray-port=6380 -- \
  python -m tinyexp.examples.pi_exp launcher=ray
```

For GPU jobs, verify CUDA visibility and use `DDPAccelerator` with NCCL. For `launcher=ray`, every node needs the same code, Python environment, dependencies, and dataset paths; set `ray_cfg.ray_num_cpus_per_worker` and `ray_cfg.ray_num_gpus_per_worker` for each worker's resources. Do not use the static Ray helper on a node whose existing Ray runtime must be preserved.

## Examples to reuse

- `tinyexp/examples/pi_exp.py`: general distributed workload and a fast smoke test (`mode=run`).
- `tinyexp/examples/mnist_exp.py`: CPU or DDP image training, checkpoints, logging, and CLI overrides.
- `tinyexp/examples/resnet_exp.py`: ImageNet training; set `IMAGENET_HOME` or override `dataloader_cfg.data_root`; defaults to DDP and Ray.

The examples are included in the published package. The bundled examples use PyTorch, so a bare `pip install tinyexp`
does not install all of their dependencies. For a normal install, use the PyTorch extra and run them as modules:

```bash
python -m pip install "tinyexp[pytorch]"
python -m tinyexp.examples.pi_exp pi_cfg.total_samples=100000
```

If the machine needs a CUDA/ROCm/vendor-specific PyTorch build, install `tinyexp` without the extra and install a compatible `torch`, `torchvision`, and `accelerate` set using that backend's instructions.

## Development checks

For repository changes from a source checkout, use `make install-pytorch`, then run `make check` and focused `pytest` tests after changing library behavior.
