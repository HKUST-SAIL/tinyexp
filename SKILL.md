---
name: tinyexp
description: Write, configure, debug and launch TinyExp experiments (`Exp(TinyExp, ...)` dataclasses, `key=value` overrides, `mode=help`, accelerators, Ray workers, torchrun/accelerate, multi-node Ray). Use when the task mentions tinyexp, imports `tinyexp`, or subclasses `TinyExp`.
---

# TinyExp experiments

An experiment is one file: `@dataclass(repr=False) class Exp(TinyExp, <mixins>)` with nested `XXXCfg` dataclasses, a
`run()` that dispatches on `self.mode`, and `store_and_run_exp(Exp)` under `if __name__ == "__main__":`. Every field
is a CLI override by dotted path (`train_cfg.lr=0.01`).

## 1. Read docstrings, not source

The installed package documents itself. If `python -c "import tinyexp; print(tinyexp.__doc__)"` prints `None`, the
install is too old; upgrade it (`[vit]` for `vit_tp_exp`; for a CUDA/ROCm PyTorch, drop the extra and install
`torch`/`torchvision` from that backend's index):

```bash
python -m pip install -U "tinyexp[pytorch] @ git+https://github.com/HKUST-SAIL/tinyexp.git"
```

Print the documentation of a module, class, nested class or method by dotted name (an overriding method shows its
base class's docstring):

```bash
python -c "import inspect,pydoc,sys; print(inspect.getdoc(pydoc.locate(sys.argv[1])))" tinyexp.exp_mixins.RayCfgMixin.RayCfg
```

| Need | Dotted name |
| --- | --- |
| New experiment: minimal file, `TinyExp` fields, override syntax, `launcher` pairing | `tinyexp` |
| How overrides are merged and reported | `tinyexp.TinyExp.set_cfg` |
| Training-loop API (`prepare`, `backward`, `optimizer_step`, `reduce`, ...) | `tinyexp.tiny_engine.accelerator.base_accelerator` |
| Picking an accelerator (CPU, DDP, FSDP, TP, HF Accelerate) | `tinyexp.tiny_engine.accelerator` |
| Code that runs on Ray workers (contract, complete CPU example) | `tinyexp.exp_mixins.RayCfgMixin` |
| Ray worker count, resources, multi-node paths | `tinyexp.exp_mixins.RayCfgMixin.RayCfg` |
| Checkpoint save / load / resume / eval | `tinyexp.exp_mixins.CheckpointCfgMixin.CheckpointCfg` |
| Logger, W&B, Trackio, Redis cache: which mixin | `tinyexp.exp_mixins.basic_mixins` (table) |
| ... then its settings, e.g. W&B | `tinyexp.exp_mixins.WandbCfgMixin.WandbCfg` (likewise `LoggerCfg`, `TrackioCfg`, `RedisCfg`) |
| Infinite sampler, fake dataloader; EMA | `tinyexp.dataset`; `tinyexp.utils` |
| Exceptions and what raises them | `tinyexp.exceptions` |

Each bundled example's docstring holds a run header (`Modes`, `Run`, `Env`, `Needs`, `Out`): its valid commands,
data, hardware and outputs. Print it without importing the example, whose extras may be missing:

```bash
python -c "import ast,importlib.util as u,sys; print(ast.get_docstring(ast.parse(open(u.find_spec(sys.argv[1]).origin).read())))" tinyexp.examples.mnist_exp
```

Examples (files to copy as templates are in `python -c "import tinyexp.examples as e; print(e.__path__[0])"`):

- `mnist_exp`: template for a training experiment (checkpoints, resume, W&B/Trackio); CPU, or
  `accelerator_cfg.accelerator=ddp` on CUDA.
- `pi_exp`: non-deep-learning distributed job, no data; the fastest smoke test.
- `resnet_exp`: ImageNet; `accelerator_cfg.accelerator` is `ddp`, `fsdp`, `cpu`, or `hf` (Hugging Face Accelerate,
  with `accelerate launch`); Redis cache.
- `vit_tp_exp`: tensor parallelism, TP-vs-DDP benchmark; needs `tinyexp[vit]`.

## 2. Rules that are easy to miss

- Check every new command with `mode=help` first: it prints the resolved config and the overridden values, then
  exits without starting workers. It checks key names and types, not values or resources; it imports the
  experiment (missing extras fail here) and creates `output_root`. If it fails with
  `FileExistsError: ... 'output'`, pass `output_root=/tmp/tinyexp`.
- `TinyExp` defaults to `launcher=mp`, but every bundled example defaults to `launcher=ray` and starts local Ray
  workers. Pass `launcher=mp` for one plain process; `torchrun` and `accelerate launch` also need `launcher=mp`, or
  each of their processes starts its own Ray workers.
- Before a `launcher=ray` run, check `ray_cfg` in the `mode=help` output: workers x CPUs/GPUs per worker must fit
  (`ray_num_worker=-1` takes what is free; `mnist_exp` needs 8 CPUs, `resnet_exp` 12 CPUs and 1 GPU per worker).
  Too large fails at once (`InsufficientCPUError`, also for GPUs; `InvalidWorkerCountError` when `-1` finds
  nothing); busy resources fail after `ray_placement_timeout_s`. The base `RayCfg` asks for 1 GPU per worker, so a
  CPU-only experiment overrides it as in the `RayCfgMixin` example.
- With `launcher=ray`, start as `python -m pkg.module`, not `python path/to/file.py`, so workers import the class by
  path. Run the environment's `python` directly: under `uv run`, Ray also starts every worker through `uv`, unless
  prefixed as `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 uv run python -m ...`.
- To use a running cluster, set `RAY_ADDRESS`: `launcher=ray` joins it and never starts or stops Ray.
  `tinyexp-run-with-ray-cluster` (command 6, `--help`) is for dedicated nodes only: with `--node-count` above 1 it
  runs `ray stop --force` on each node at start and exit. All nodes need the same environment and code, and paths
  every node can reach (`RayCfg`).
- Change behavior with overrides, a subclass of an example's `Exp` or a new file, not by editing the installed
  package.

## 3. Commands, in the order to try them

```bash
# 1. Inspect only: resolved config plus overridden values, nothing runs.
python -m tinyexp.examples.pi_exp mode=help pi_cfg.total_samples=100000

# 2. One CPU process, no Ray: the fastest end-to-end check.
python -m tinyexp.examples.pi_exp launcher=mp pi_cfg.total_samples=100000

# 3. Local Ray workers started by TinyExp (plain python, module form).
python -m tinyexp.examples.pi_exp launcher=ray ray_cfg.ray_num_worker=2 ray_cfg.ray_num_cpus_per_worker=1

# 4. torchrun owns the processes, so launcher=mp. CPU (gloo) first, then GPU DDP (2 CUDA GPUs, NCCL).
torchrun --nproc-per-node=2 -m tinyexp.examples.pi_exp launcher=mp pi_cfg.total_samples=100000
torchrun --nproc-per-node=2 -m tinyexp.examples.mnist_exp launcher=mp accelerator_cfg.accelerator=ddp

# 5. Your own file my_exp.py, run from its directory.
python my_exp.py mode=help train_cfg.lr=0.01
python -m my_exp launcher=mp train_cfg.lr=0.01
torchrun --nproc-per-node=2 my_exp.py launcher=mp

# 6. Two dedicated nodes; 10.0.0.1 = node 0's IP, reachable from node 1. Same arguments on every node except a
#    unique --node-rank; only node 0 runs the command after --. SPREAD puts the workers on both nodes.
#    node 0 (head, runs the job):
tinyexp-run-with-ray-cluster --node-count=2 --node-rank=0 --head-addr=10.0.0.1 -- \
  python -m tinyexp.examples.pi_exp launcher=ray ray_cfg.ray_num_worker=2 ray_cfg.ray_placement_strategy=SPREAD
#    node 1 (joins, then waits until the head exits):
tinyexp-run-with-ray-cluster --node-count=2 --node-rank=1 --head-addr=10.0.0.1 -- \
  python -m tinyexp.examples.pi_exp launcher=ray ray_cfg.ray_num_worker=2 ray_cfg.ray_placement_strategy=SPREAD
```
