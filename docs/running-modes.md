# Running Modes and Environment Requirements

TinyExp separates three concerns that are easy to confuse:

- The command that creates processes: plain `python`, `torchrun`, `accelerate launch`, or the Ray cluster helper.
- The TinyExp `launcher` setting: `mp` or `ray`.
- The accelerator built by the experiment, such as `CPUAccelerator`, `DDPAccelerator`, or a custom use of `HFAccelerator`.

The command alone does not select all three. Check the experiment defaults with `mode=help` before launching a job.

## Mode Matrix

| Run style | Typical command | Required TinyExp launcher | Process owner | Main use |
| --- | --- | --- | --- | --- |
| Plain Python | `python exp.py` | `mp` for one direct process, or `ray` for Ray workers | Current Python process or TinyExp's Ray driver | Local debug and local Ray runs |
| TorchRun | `torchrun ... exp.py launcher=mp` | `mp` | PyTorch `torchrun` | Explicit PyTorch distributed launch |
| Accelerate launch | `accelerate launch ... exp.py launcher=mp` | `mp` | Hugging Face Accelerate CLI | External process launch using Accelerate configuration |
| Static Ray cluster | `tinyexp-run-with-ray-cluster ... -- python exp.py launcher=ray` | `ray` | Ray head plus Ray workers | Multi-node Ray scheduling |

`torchrun` and `accelerate launch` are not valid values for TinyExp's `launcher` field. They create processes externally, so each process must enter TinyExp with `launcher=mp`. Conversely, `launcher=ray` makes the driver call `ray.init()` and create Ray actors according to `ray_cfg`.

The base `TinyExp` class defaults to `launcher=mp`. The bundled MNIST, ResNet, and pi examples override it to `launcher=ray`.

## Common Requirements

All modes require:

- Python `>=3.9,<4.0`.
- TinyExp and the experiment's Python dependencies installed in the active environment.
- A PyTorch build compatible with the host operating system and, for GPU jobs, the installed NVIDIA driver and CUDA runtime.
- Read access to experiment code and datasets, and write access to `output_root`.
- Consistent code, configuration, and package versions in every process or cluster node participating in a job.

Install the source development environment with:

```bash
make install
```

Or install the published package with:

```bash
pip install tinyexp
```

TinyExp declares PyTorch, Ray, Accelerate, and their Python-level dependencies, but it does not choose a CUDA-specific PyTorch wheel or manage GPU drivers. Follow the PyTorch installation guidance for the target machine when a GPU build is required.

Check the active environment before launching:

```bash
python -c "import sys, torch, ray, accelerate; print(sys.version); print(torch.__version__); print(ray.__version__); print(accelerate.__version__)"
command -v torchrun
command -v accelerate
command -v ray
```

For GPU jobs, also check:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

## Plain Python

Plain Python has two behaviors, selected by the experiment configuration.

### Direct single process

With `launcher=mp`, TinyExp runs `Exp.run()` in the current process. No external launcher creates additional processes:

```bash
uv run python -m tinyexp.examples.pi_exp \
  launcher=mp \
  pi_cfg.total_samples=100000
```

This is the smallest environment for debugging. The experiment's accelerator still controls whether it uses CPU or GPU.

### Local Ray workers

With `launcher=ray`, the Python process becomes a Ray driver. TinyExp calls `ray.init()`, creates a placement group, and launches Ray actors:

```bash
uv run python -m tinyexp.examples.pi_exp \
  launcher=ray \
  ray_cfg.ray_num_worker=2 \
  ray_cfg.ray_num_cpus_per_worker=1 \
  ray_cfg.ray_num_gpus_per_worker=0 \
  pi_cfg.total_samples=100000
```

Requirements:

- The `ray` Python package must import successfully.
- The machine must have enough resources for every worker bundle.
- `ray_cfg.ray_num_worker` must be `-1` or a positive integer. `-1` fills the available CPU or GPU capacity.
- `ray_cfg.ray_num_cpus_per_worker` must be positive.
- `ray_cfg.ray_num_gpus_per_worker` must be non-negative.
- `ray_cfg.ray_placement_timeout_s` controls how long TinyExp waits for the placement group; the default is 120 seconds.
- Requests that exceed the cluster's total CPU or GPU capacity fail before placement starts. If the total capacity is sufficient but currently busy, placement waits up to the configured timeout.
- GPU workers require a CUDA-enabled PyTorch installation and visible GPUs.
- TinyExp reads the placement-group bundle-to-node topology before creating Ray worker actors, then derives `RANK` and `LOCAL_RANK` from that topology. Ray workers must be homogeneous: every participating node must host the same number of workers, so every node has the same local-rank range. When bundles are interleaved across nodes, global ranks are reassigned so ranks remain contiguous within each node (for example, node-local workers receive `0..N-1`, then the next node receives the following range).
- For multi-worker runs, TinyExp starts a zero-CPU TCPStore actor in placement-group bundle 0. The actor binds and holds a dynamic port before worker creation, and all ranks connect to it as clients. The Ray head therefore does not need to host rank 0 or provide a GPU.

If `RAY_ADDRESS` already points to a reachable Ray cluster, `ray.init()` can attach to that cluster. Otherwise, Ray starts a local runtime.

The MNIST example uses `seed=42` by default for repeatable model initialization and dropout behavior. Override it explicitly when comparing runs across launchers or cluster topologies.

For configuration inspection without starting workers:

```bash
uv run python tinyexp/examples/mnist_exp.py \
  mode=help \
  ray_cfg.ray_num_worker=1 \
  ray_cfg.ray_num_gpus_per_worker=0
```

## TorchRun

`torchrun` is installed with PyTorch. It creates worker processes and supplies the distributed environment variables consumed by TinyExp accelerators, including `RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `MASTER_ADDR`, and `MASTER_PORT`.

Always override bundled examples with `launcher=mp` so each `torchrun` process executes the experiment directly:

```bash
uv run torchrun \
  --nnodes=1 \
  --node-rank=0 \
  --nproc-per-node=2 \
  --master-addr=127.0.0.1 \
  --master-port=29500 \
  -m tinyexp.examples.pi_exp \
  launcher=mp \
  pi_cfg.total_samples=100000
```

CPU distributed jobs use `CPUAccelerator` and PyTorch's Gloo backend. When `WORLD_SIZE > 1`, `CPUAccelerator` wraps the model with `DistributedDataParallel`, so gradients and model parameters are synchronized across processes. GPU distributed jobs using TinyExp's `DDPAccelerator` require:

- CUDA-enabled PyTorch.
- NVIDIA GPUs visible to each process.
- NCCL support, because `DDPAccelerator` initializes the `nccl` backend.
- A process count that does not exceed the intended visible GPU count for one-process-per-GPU DDP.

For a single-node GPU example:

```bash
uv run torchrun \
  --nnodes=1 \
  --node-rank=0 \
  --nproc-per-node=2 \
  --master-addr=127.0.0.1 \
  --master-port=29500 \
  tinyexp/examples/resnet_exp.py \
  launcher=mp \
  accelerator_cfg.accelerator=ddp \
  redis_cfg.redis_cache_enabled=false
```

For multi-node TorchRun, every node needs the same environment, source code, and data view. The rendezvous address and port must be reachable between nodes. Configure `--nnodes`, `--node-rank`, and the rendezvous arguments according to the PyTorch `torchrun` contract.

## Accelerate Launch

`accelerate launch` is another external process owner. Use `launcher=mp` for the same reason as TorchRun:

The following portable smoke test verifies the Accelerate entry path with one CPU process:

```bash
uv run accelerate launch \
  --cpu \
  --num-processes=1 \
  --num-machines=1 \
  --num-cpu-threads-per-process=1 \
  --mixed-precision=no \
  --dynamo-backend=no \
  -m tinyexp.examples.pi_exp \
  launcher=mp \
  pi_cfg.total_samples=100000
```

A multi-GPU launch can use the existing DDP accelerator when CUDA and NCCL are available:

```bash
uv run accelerate launch \
  --multi-gpu \
  --num-processes=2 \
  tinyexp/examples/resnet_exp.py \
  launcher=mp \
  accelerator_cfg.accelerator=ddp \
  redis_cfg.redis_cache_enabled=false
```

Accelerate multi-CPU execution depends on the distributed configuration and installed CPU communication backend. A
`--cpu --num-processes=2` command is not a portable guarantee that two processes will be created. Use the verified
TorchRun CPU recipe when explicit local CPU process counts are required.

Accelerate settings can come from explicit CLI arguments or a file created by `accelerate config`. Prefer explicit arguments in reproducible scripts so machine-local defaults do not silently change the topology.

There are two distinct Accelerate integrations:

1. `accelerate launch` can create processes for an experiment that still builds TinyExp's `CPUAccelerator` or `DDPAccelerator`.
2. `HFAccelerator` adapts `accelerate.Accelerator` to TinyExp's accelerator protocol, but an experiment must construct it explicitly.

The bundled MNIST and ResNet examples currently accept only `accelerator_cfg.accelerator=cpu` and `accelerator_cfg.accelerator=ddp`. They do not accept `accelerator_cfg.accelerator=hf`; using `accelerate launch` does not automatically select `HFAccelerator`.

For multi-machine Accelerate launches, all machines need matching code and environments, and the main process IP and port must be reachable. Supply `--num-machines`, `--machine-rank`, `--main-process-ip`, and `--main-process-port`, or provide equivalent Accelerate configuration.

## Static Ray Cluster

`tinyexp-run-with-ray-cluster` starts a Ray head on node rank 0, joins Ray workers from the remaining nodes, and runs the command only on the head after the requested nodes are alive. The command should use `launcher=ray` so the TinyExp driver attaches to the cluster through the injected `RAY_ADDRESS`.

On every node, provide the same node count and head address, with a unique node rank. For example, on the head:

```bash
tinyexp-run-with-ray-cluster \
  --node-count=2 \
  --node-rank=0 \
  --head-addr=10.0.0.1 \
  --ray-port=6380 \
  -- \
  python -m tinyexp.examples.pi_exp \
  launcher=ray \
  ray_cfg.ray_num_worker=2 \
  ray_cfg.ray_num_cpus_per_worker=1 \
  ray_cfg.ray_num_gpus_per_worker=0
```

On the worker node:

```bash
tinyexp-run-with-ray-cluster \
  --node-count=2 \
  --node-rank=1 \
  --head-addr=10.0.0.1 \
  --ray-port=6380 \
  -- \
  python -m tinyexp.examples.pi_exp \
  launcher=ray
```

The command after `--` is required on every node for a uniform invocation, but only node rank 0 executes it.

Multi-node requirements:

- `ray` and Python executables must be available on every node. Use `--ray-bin` and `--python-bin` when they are not on `PATH`.
- The helper owns the Ray runtime on each participating node: it runs `ray stop --force` before startup and during cleanup. Do not use it on a node whose existing Ray runtime must remain active.
- Each node must use compatible Python, TinyExp, PyTorch, Ray, and experiment dependency versions.
- Custom experiment modules must be importable on every node. The helper does not distribute source code or create environments.
- Dataset paths used by a scheduled worker must exist on that worker, either through a shared filesystem or equivalent per-node data layout.
- The head address must be reachable from every worker and must not resolve to loopback for a multi-node job.
- Firewalls and security groups must allow Ray's head port and Ray's node-to-node runtime traffic. The head port defaults to `6379`; choose an unused `--ray-port` when that port is occupied. The selected head port must also be outside Ray's configured worker port range. Dashboard, metrics, client, and other Ray runtime ports may also need explicit network policy.
- Aggregate Ray cluster resources must satisfy the requested placement group. A job waits if resources exist in theory but cannot be placed with the requested bundle shape or placement strategy.
- The Ray head may be CPU-only. GPU worker bundles and the distributed TCPStore are placed on eligible worker nodes according to the resolved placement group.

When `--node-count=1`, the helper executes the command unchanged and does not start a static Ray cluster. A command with `launcher=ray` may still start a local Ray runtime itself.

Check every node before launch:

```bash
ray --version
python -c "import tinyexp, torch, ray; print(tinyexp.__file__); print(torch.__version__); print(ray.__version__)"
```

## Optional Services

Redis and W&B requirements are independent of the four launch styles.

- Enabling Redis cache requires a reachable Redis service.
- TinyExp-managed standalone Redis requires `redis-server` on the host that starts it.
- TinyExp-managed Redis Cluster also requires `redis-cli` on the coordinating host.
- The ResNet example enables Redis cache by default. Set `redis_cfg.redis_cache_enabled=false` when Redis is not installed or desired.
- Enabling W&B requires suitable credentials and network access, or an explicitly configured offline mode.

Check Redis system commands when cache management is enabled:

```bash
command -v redis-server
command -v redis-cli
```

## Choosing a Mode

Use direct `python` with `launcher=mp` for the smallest local debug loop. Use local `launcher=ray` when the experiment should exercise TinyExp's Ray resource orchestration. Use TorchRun when PyTorch's distributed launcher is the deployment contract. Use Accelerate launch when its machine configuration or launch integrations are required. Use the static Ray cluster helper when workers must be scheduled across an existing set of reachable machines.
