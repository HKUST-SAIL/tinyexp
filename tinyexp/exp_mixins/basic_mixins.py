"""Opt-in ``XXXCfgMixin`` components; mix only the ones an experiment needs into ``TinyExp``.

Each mixin adds one config field whose nested dataclass holds both settings and a ``build_*`` helper:

| mixin | field | use |
| --- | --- | --- |
| `LoggerCfgMixin` | `logger_cfg` | `build_logger(save_dir, distributed_rank)` -> loguru logger |
| `RayCfgMixin` | `ray_cfg` | `launcher=ray`: worker contract and minimal example in `RayCfgMixin`, count/resources in `RayCfg` |
| `CheckpointCfgMixin` | `checkpoint_cfg` | `save_checkpoint` / `load_checkpoint` / RNG state helpers |
| `WandbCfgMixin` | `wandb_cfg` | `build_wandb(accelerator, **wandb_init_kwargs)` if enabled, else `None` |
| `RedisCfgMixin` | `redis_cfg` | Redis byte-cache settings consumed by datasets and launchers |

Usage: ``class Exp(TinyExp, RayCfgMixin, CheckpointCfgMixin, LoggerCfgMixin)``. To change defaults,
subclass the nested config and redeclare the field, e.g.::

    @dataclass
    class RayCfg(RayCfgMixin.RayCfg):
        ray_num_gpus_per_worker: float = 0.0

    ray_cfg: RayCfg = field(default_factory=RayCfg)

``torch`` and ``wandb`` are imported lazily inside methods so ``import tinyexp`` stays torch-free.
"""

import os
import random
import tempfile
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import ray
from loguru._logger import Logger
from omegaconf import DictConfig
from omegaconf.listconfig import ListConfig

from ..exceptions import (
    InsufficientCPUError,
    InvalidWorkerCountError,
    UnsupportedCheckpointFormatError,
)
from ..utils.log_utils import tiny_logger_setup
from ..utils.ray_utils import (
    _maybe_start_ray_redis_cache,
    get_network_config,
    get_num_worker_options,
    get_placement_group,
    get_placement_group_node_ids,
    start_ray_rendezvous_store,
)


@dataclass
class LoggerCfgMixin:
    @dataclass
    class LoggerCfg:
        """``build_logger(save_dir, distributed_rank=0, filename="log.txt", mode="a")`` -> loguru logger.

        Creates ``save_dir`` (usually ``self.get_run_dir()``). Every rank writes to ``<save_dir>/<filename>``
        and stderr: rank 0 at INFO, other ranks at WARNING and above. ``mode="o"`` overwrites the file.
        """

        def build_logger(
            self,
            save_dir: str,
            distributed_rank: int = 0,
            filename: str = "log.txt",
            mode: str = "a",
        ) -> Logger:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            logger = tiny_logger_setup(save_dir, distributed_rank, filename, mode)
            logger.info(f"==> log file: {os.path.join(save_dir, filename)}")
            return logger

    logger_cfg: LoggerCfg = field(default_factory=LoggerCfg)


def _resolve_ray_num_worker(
    ray_num_worker: int,
    num_cpus_per_worker: int,
    num_gpus_per_worker: float,
    cluster_resources: dict[str, float],
) -> int:
    if ray_num_worker != -1:
        return ray_num_worker

    cpu_worker_count = int(cluster_resources.get("CPU", 0) // num_cpus_per_worker)
    if num_gpus_per_worker > 0:
        gpu_worker_count = int(cluster_resources.get("GPU", 0) // num_gpus_per_worker)
        resolved_worker_count = min(cpu_worker_count, gpu_worker_count)
        resource_name = "GPU and CPU"
    else:
        resolved_worker_count = cpu_worker_count
        resource_name = "CPU"

    if resolved_worker_count <= 0:
        raise InvalidWorkerCountError(resolved_worker_count)
    print(
        f"==> ray_num_worker is -1, using available {resource_name} resources: {resolved_worker_count}",
        flush=True,
    )
    return resolved_worker_count


@dataclass
class RayCfgMixin:
    """Run ``Exp.run()`` on a group of Ray workers when ``launcher=ray``; worker count/resources: ``RayCfg``.

    Every worker calls ``run()`` with ``RANK``/``WORLD_SIZE``/``LOCAL_RANK``/``MASTER_ADDR``/``MASTER_PORT`` set,
    so an accelerator built inside ``run()`` is one rank of a ready process group (``rank``, ``world_size``,
    ``reduce``). Worker stdout reaches the driver prefixed and interleaved; for one final line, return it from
    ``get_ray_run_result`` on rank 0 and the driver prints it after every worker has finished. The same file runs
    unchanged with ``launcher=mp`` (one process, ``world_size == 1``), where that method is never called::

        @dataclass(repr=False)
        class Exp(TinyExp, RayCfgMixin):
            launcher: str = "ray"

            def run(self) -> None:
                from tinyexp.tiny_engine.accelerator import CPUAccelerator

                accelerator = CPUAccelerator()
                try:
                    part = torch.arange(accelerator.rank, 1000, accelerator.world_size).sum().double()
                    total = accelerator.reduce(part, reduction="sum").item()
                    self._result = f"total={total:.0f}" if accelerator.is_main_process else None
                    if self._result and self.launcher != "ray":
                        print(self._result, flush=True)
                finally:
                    accelerator.destroy()

            def get_ray_run_result(self):
                return self._result

    Start it as ``python -m package.module`` rather than ``python path/to/file.py``: the class then reaches the
    workers by import path. From ``__main__`` it is pickled by value, which works locally but is fragile
    ("not the same object as ..." pickling errors), and its imports must still resolve on every node.
    """

    def get_ray_run_result(self) -> Optional[str]:
        """Rank 0's return value is printed by the Ray driver after all workers finish (default ``None``: nothing).

        Only called with ``launcher=ray``; see ``RayCfgMixin`` for the pattern.
        """
        return None

    @dataclass
    class RayCfg:
        """Ray worker group used when ``launcher=ray``.

        - ``ray_num_worker``: total workers across the whole Ray cluster, not per node. Each worker is
          one rank (``RANK``/``WORLD_SIZE``/``LOCAL_RANK``/``MASTER_ADDR``/``MASTER_PORT`` are set).
          ``-1`` fills the currently available resources.
        - ``ray_num_cpus_per_worker`` / ``ray_num_gpus_per_worker``: resources of one worker (one
          placement-group bundle). Use ``ray_num_gpus_per_worker=0`` for CPU-only runs.
        - ``ray_placement_strategy``: Ray placement-group strategy (``PACK``, ``SPREAD``, ...).
        - ``ray_placement_timeout_s``: how long to wait for the placement group before failing.

        On a static multi-node cluster (``tinyexp-run-with-ray-cluster``) the same total is spread over
        the nodes by Ray; e.g. 2 nodes x 4 GPUs -> ``ray_cfg.ray_num_worker=8 ray_cfg.ray_num_gpus_per_worker=1``
        (GPU examples additionally take ``accelerator_cfg.accelerator=ddp``). Workers may land on any node,
        so ``dataloader_cfg.data_root``, ``output_root`` and ``resume_from`` must be paths every node can
        reach (shared filesystem or identical per-node layout).
        """

        # ---------------- luancher configuration ---------------- #
        ray_num_worker: int = -1  # Number of Ray workers, -1 means fill available worker resources.
        ray_num_cpus_per_worker: int = 1
        ray_num_gpus_per_worker: float = 1.0
        ray_placement_strategy: str = "PACK"
        ray_placement_timeout_s: float = 120.0

        @classmethod
        def run(cls, exp_class: type[Any], experiment_cfg: DictConfig) -> None:  # noqa: C901
            """
            Run the Ray driver-side orchestration for ``launcher=ray``.

            Steps (each marked by a ``# ---- n/5`` comment below):
            1. validate ``ray_cfg`` and start Ray if no runtime exists yet;
            2. resolve the worker count (``-1`` -> available resources) and check cluster capacity;
            3. start the Ray-managed Redis cache if ``redis_cfg`` asks for it;
            4. reserve one placement-group bundle per worker and pick the rendezvous address;
            5. create one actor per worker, ``set_cfg`` + ``run`` on all, print rank 0's run result.
            Cleanup releases the rendezvous actor, placement group, Redis and the Ray runtime it started.

            Args:
                exp_class: Experiment class instantiated by each Ray worker.
                experiment_cfg: Complete Hydra configuration for exp_class.
            """
            # ---- 1/5 validate ray_cfg and own the Ray runtime ------------------------------------ #
            ray_cfg = experiment_cfg.ray_cfg
            num_cpus_per_worker = int(ray_cfg.ray_num_cpus_per_worker)
            num_gpus_per_worker = float(ray_cfg.ray_num_gpus_per_worker)
            if num_cpus_per_worker <= 0:
                raise ValueError("ray_num_cpus_per_worker must be greater than 0")  # noqa: TRY003
            if num_gpus_per_worker < 0:
                raise ValueError("ray_num_gpus_per_worker must not be negative")  # noqa: TRY003

            placement_timeout_s = float(getattr(ray_cfg, "ray_placement_timeout_s", 120.0))
            if placement_timeout_s <= 0:
                raise ValueError("ray_placement_timeout_s must be greater than 0")  # noqa: TRY003

            if ray_cfg.ray_num_worker < -1 or ray_cfg.ray_num_worker == 0:
                raise InvalidWorkerCountError(ray_cfg.ray_num_worker)
            owns_ray_runtime = not ray.is_initialized()
            if owns_ray_runtime:
                ray.init()
            pg = None
            redis_manager = None
            rendezvous_actor = None
            worker_group = []

            try:
                # ---- 2/5 resolve the worker count and check cluster capacity ---------------------- #
                cluster_resources = ray.cluster_resources()
                sizing_resources = ray.available_resources() if ray_cfg.ray_num_worker == -1 else cluster_resources
                ray_num_worker = _resolve_ray_num_worker(
                    ray_cfg.ray_num_worker,
                    num_cpus_per_worker,
                    num_gpus_per_worker,
                    sizing_resources,
                )
                if ray_cfg.ray_num_worker == -1:
                    ray_cfg.ray_num_worker = ray_num_worker

                remote_exp = ray.remote(exp_class)

                needed_num_cpus_per_worker = num_cpus_per_worker

                needed_cpu = ray_num_worker * needed_num_cpus_per_worker
                needed_gpu = ray_num_worker * num_gpus_per_worker
                total_cpu = int(cluster_resources.get("CPU", 0))
                total_gpu = float(cluster_resources.get("GPU", 0))

                if needed_cpu > total_cpu or needed_gpu > total_gpu:
                    raise InsufficientCPUError(
                        total_cpu=total_cpu,
                        needed_cpu=needed_cpu,
                        total_gpu=total_gpu,
                        needed_gpu=needed_gpu,
                    )

                # ---- 3/5 start the Ray-managed Redis cache (no-op unless redis_cfg enables it) ---- #
                redis_manager = _maybe_start_ray_redis_cache(experiment_cfg)
                # ---- 4/5 reserve placement-group bundles and choose the rendezvous address -------- #

                pg = get_placement_group(
                    num_worker=ray_cfg.ray_num_worker,
                    num_gpus_per_worker=num_gpus_per_worker,
                    num_cpus_per_worker=needed_num_cpus_per_worker,
                    strategy=ray_cfg.ray_placement_strategy,
                    timeout_s=placement_timeout_s,
                )
                worker_node_ids = get_placement_group_node_ids(pg, ray_cfg.ray_num_worker)
                if ray_cfg.ray_num_worker > 1:
                    rendezvous_actor, master_addr, master_port = start_ray_rendezvous_store(
                        pg,
                        ray_cfg.ray_num_worker,
                        placement_timeout_s,
                    )
                else:
                    master_addr, master_port = get_network_config(worker_node_ids[0])
                options_list = get_num_worker_options(
                    pg,
                    ray_cfg.ray_num_worker,
                    gpu_ratio=num_gpus_per_worker,
                    num_cpus_per_worker=needed_num_cpus_per_worker,
                    master_addr=master_addr,
                    master_port=master_port,
                    node_ids=worker_node_ids,
                )
                runtime_envs = [options["runtime_env"]["env_vars"] for options in options_list]
                # ---- 5/5 print topology, launch workers, run them, report rank 0's result -------- #
                print("==> Ray worker topology:", flush=True)
                node_ranks = {node_id: node_rank for node_rank, node_id in enumerate(dict.fromkeys(worker_node_ids))}
                topology = sorted(
                    zip(worker_node_ids, runtime_envs),
                    key=lambda item: int(item[1]["RANK"]),
                )
                for node_id, env_vars in topology:
                    print(
                        f"    rank={env_vars['RANK']}/{ray_cfg.ray_num_worker} node={node_id} "
                        f"node_rank={node_ranks[node_id]} "
                        f"local_rank={env_vars['LOCAL_RANK']}",
                        flush=True,
                    )
                worker_group = [remote_exp.options(**options).remote() for options in options_list]
                ray.get([worker.set_cfg.remote(experiment_cfg) for worker in worker_group])
                ray.get([worker.run.remote() for worker in worker_group])
                main_worker_index = next(
                    index for index, env_vars in enumerate(runtime_envs) if env_vars["RANK"] == "0"
                )
                result = ray.get(worker_group[main_worker_index].get_ray_run_result.remote())
                if result is not None:
                    print(result, flush=True)
            finally:
                # ---- cleanup: release everything this call created, in reverse order -------------- #
                if rendezvous_actor is not None:
                    with suppress(Exception):
                        ray.kill(rendezvous_actor, no_restart=True)

                if pg is not None:
                    with suppress(Exception):
                        ray.util.remove_placement_group(pg)

                if redis_manager is not None:
                    with suppress(Exception):
                        redis_manager.stop()

                if owns_ray_runtime and ray.is_initialized():
                    with suppress(Exception):
                        # Drain Ray's asynchronous worker logs before tearing down the runtime.
                        ray.shutdown(_exiting_interpreter=True)

    ray_cfg: RayCfg = field(default_factory=RayCfg)


@dataclass
class WandbCfgMixin:
    @dataclass
    class WandbCfg:
        """Optional Weights & Biases logging, off by default (``wandb_cfg.enable_wandb=true`` turns it on).

        ``build_wandb(accelerator=None, **kwargs)`` returns ``None`` when disabled. When enabled it returns the
        ``wandb`` module on every rank, but calls ``wandb.init(**kwargs)`` (``project=``, ``name=``, ``config=``, ...)
        only when ``accelerator.rank == 0`` (or when no accelerator is given), so log from rank 0 only
        (``cfg_dict`` below is what ``self.print_cfg(logger)`` returns)::

            wandb = self.wandb_cfg.build_wandb(accelerator=accelerator, project="p", name=self.exp_name, config=cfg_dict)
            if wandb is not None and accelerator.is_main_process:
                wandb.log({"loss": loss})
        """

        enable_wandb: bool = False

        def build_wandb(self, accelerator=None, **kwargs):
            if self.enable_wandb:
                import wandb

                if accelerator is None or accelerator.rank == 0:
                    wandb.init(**kwargs)
                return wandb

    wandb_cfg: WandbCfg = field(default_factory=WandbCfg)


@dataclass
class RedisCfgMixin:
    """Adds ``redis_cfg``: settings of the Redis byte cache (see ``RedisCfg``)."""

    @dataclass
    class RedisCfg:
        """Settings of the Redis byte cache used by datasets (e.g. cached ImageNet files in ``resnet_exp``).

        - ``redis_cache_enabled`` (default ``True``): whether the experiment should use the cache.
        - ``redis_cluster_host`` / ``redis_cluster_ports``: where Redis listens; standalone mode runs one
          ``redis-server`` per port and shards keys across them.
        - ``redis_cache_max_memory``: total memory limit in **GB** (default 160, sized for ImageNet), split
          evenly across the Redis processes.
        - ``redis_cluster_startup_timeout_s``: how long to wait for Redis to come up.
        - ``redis_rendezvous_world_size``: ``1`` standalone Redis (with ``launcher=ray``: one per Ray node);
          ``-1`` Ray-managed Redis Cluster over the alive Ray nodes; ``>1`` externally managed Redis Cluster.

        TinyExp never starts Redis from inside ``run()``; the launcher owns its lifecycle:
        ``launcher=ray`` starts and stops it automatically, ``tinyexp-run-with-redis -- <command>`` does so for
        any other launch (and injects the matching ``redis_cfg.*`` overrides), or point these fields at a
        Redis you manage yourself. The wrapper refuses a ``launcher=ray`` command, which already owns its
        Redis. Inside the experiment connect with::

            from tinyexp.utils.redis_utils import RedisClientManager
            cache = RedisClientManager(cfg.redis_cluster_host, list(cfg.redis_cluster_ports),
                                       redis_world_size=int(cfg.redis_rendezvous_world_size))
            if cache.safe_get(key) is None:        # miss, or Redis unavailable
                cache.safe_set(key, value_bytes)   # False when it could not be stored
        """

        redis_cache_enabled: bool = True
        redis_cluster_host: str = "127.0.0.1"
        redis_cluster_ports: ListConfig = field(
            default_factory=lambda: ListConfig([7000, 7001, 7002, 7003, 7004, 7005])
        )
        redis_cache_max_memory: int = 160  # Maximum memory is 160GB, according to the ImageNet dataset size
        redis_cluster_startup_timeout_s: float = 30.0

        # 1: standalone Redis. In Ray launches, this starts one local Redis per alive Ray node.
        # -1: Ray-managed Redis Cluster using the alive Ray nodes.
        # >1: externally managed Redis Cluster rendezvous world size.
        redis_rendezvous_world_size: int = 1

    redis_cfg: RedisCfg = field(default_factory=RedisCfg)


@dataclass
class CheckpointCfgMixin:
    @dataclass
    class CheckpointCfg:
        """Atomic ``torch.save``/``torch.load`` of training state; the caller decides where and when.

        Payload written by ``save_checkpoint`` to ``<run_dir>/<name>``::

            {
                "epoch": int | None, "global_step": int | None, "best_metric": float | None,
                "meta": {"exp_name": str, "exp_class": str, "saved_at": ISO-8601 UTC str},  # names are "" unless passed
                # present only when the object was passed:
                "model_state_dict", "optimizer_state_dict", "scheduler_state_dict",
                "scaler_state_dict", "extra_state": dict,
            }

        ``load_checkpoint(path, model=..., optimizer=..., scheduler=..., scaler=...)`` loads every object
        you pass (missing key -> ``KeyError``) and returns the whole dict, so read ``epoch``/``best_metric``/
        ``extra_state`` from it. A dict with none of ``epoch``/``global_step``/``best_metric``/``meta``/
        ``extra_state``/``model_state_dict`` raises ``UnsupportedCheckpointFormatError``
        (``tinyexp.exceptions``); a payload that is not a dict raises ``TypeError``. To inspect a file by
        hand use ``torch.load(path, weights_only=False)``.

        Conventions used by the examples::

            if accelerator.is_main_process:                       # save on rank 0 only
                # RNG state is per process: only a single-process run can restore it faithfully
                rng = {"rng_state": self.checkpoint_cfg.capture_rng_state()} if accelerator.world_size == 1 else None
                self.checkpoint_cfg.save_checkpoint(
                    run_dir=self.get_run_dir(), name=self.checkpoint_cfg.last_ckpt_name,
                    model=accelerator.unwrap_model(module),       # DDP/CPU: unwrap first
                    optimizer=optimizer, scheduler=lr_scheduler, epoch=epoch,
                    exp_name=self.exp_name, exp_class=self.exp_class,  # fills ckpt["meta"]
                    extra_state=rng,
                )
            ckpt = self.checkpoint_cfg.load_checkpoint(           # every rank loads
                self.resume_from, model=accelerator.unwrap_model(module), optimizer=optimizer,
                scheduler=lr_scheduler, map_location=accelerator.device,
            )
            start_epoch, best_metric = ckpt["epoch"] + 1, ckpt["best_metric"]
            rng_state = (ckpt.get("extra_state") or {}).get("rng_state")
            if rng_state is not None and accelerator.world_size == 1:
                self.checkpoint_cfg.restore_rng_state(rng_state)  # python/numpy/torch(/cuda) RNG

        FSDP (``FSDPAccelerator``, see ``examples/resnet_exp.py``):

        - Save: every rank calls the collective ``model_sd, optim_sd =
          accelerator.collect_checkpoint_state(module, optimizer)``; rank 0 receives full CPU state dicts
          and the other ranks receive empty dicts. Rank 0 then passes them as ``model=``/``optimizer=``
          wrapped in any object whose ``state_dict()`` returns them (only ``state_dict()`` is called).
        - Load: every rank calls ``load_checkpoint`` with the *prepared* (still wrapped) module and the
          optimizer returned by ``accelerator.prepare(model, optimizer)``; their ``load_state_dict`` is patched
          to accept the full, unsharded state dicts, so the same file works for DDP and FSDP.

        The checkpoint path is therefore ``<output_root>/<exp_name>/{last,best}.ckpt``. For an evaluation
        mode, require ``resume_from`` yourself (raise if empty) and call ``load_checkpoint`` with ``model=`` only.
        """

        last_ckpt_name: str = "last.ckpt"
        best_ckpt_name: str = "best.ckpt"

        def save_checkpoint(
            self,
            *,
            run_dir: str,
            name: str,
            model=None,
            optimizer=None,
            scheduler=None,
            scaler=None,
            epoch: Optional[int] = None,
            global_step: Optional[int] = None,
            best_metric: Optional[float] = None,
            exp_name: str = "",
            exp_class: str = "",
            extra_state: Optional[dict[str, Any]] = None,
        ) -> str:
            import torch

            save_path = Path(run_dir) / name
            save_path.parent.mkdir(parents=True, exist_ok=True)

            checkpoint: dict[str, Any] = {
                "epoch": epoch,
                "global_step": global_step,
                "best_metric": best_metric,
                "meta": {
                    "exp_name": exp_name,
                    "exp_class": exp_class,
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                },
            }
            if model is not None:
                checkpoint["model_state_dict"] = model.state_dict()
            if optimizer is not None:
                checkpoint["optimizer_state_dict"] = optimizer.state_dict()
            if scheduler is not None:
                checkpoint["scheduler_state_dict"] = scheduler.state_dict()
            if scaler is not None:
                checkpoint["scaler_state_dict"] = scaler.state_dict()
            if extra_state is not None:
                checkpoint["extra_state"] = extra_state

            temp_path: Optional[Path] = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=save_path.parent,
                    prefix=f".{save_path.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as temp_file:
                    temp_path = Path(temp_file.name)
                    torch.save(checkpoint, temp_file)
                    temp_file.flush()
                    os.fsync(temp_file.fileno())
                os.replace(temp_path, save_path)
            except Exception:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)
                raise
            return str(save_path)

        def _validate_checkpoint_payload(self, path: str, checkpoint: Any) -> dict[str, Any]:
            if not isinstance(checkpoint, dict):
                raise TypeError(  # noqa: TRY003
                    f"Checkpoint at {path} must be a dict, got {type(checkpoint).__name__}."
                )

            if (
                not any(
                    key in checkpoint
                    for key in (
                        "epoch",
                        "global_step",
                        "best_metric",
                        "meta",
                        "extra_state",
                    )
                )
                and "model_state_dict" not in checkpoint
            ):
                raise UnsupportedCheckpointFormatError(path)

            return checkpoint

        def _load_required_state(
            self,
            checkpoint: dict[str, Any],
            *,
            model=None,
            optimizer=None,
            scheduler=None,
            scaler=None,
            strict: bool = True,
        ) -> None:
            if model is not None:
                if "model_state_dict" not in checkpoint:
                    raise KeyError("model_state_dict")
                model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
            if optimizer is not None:
                if "optimizer_state_dict" not in checkpoint:
                    raise KeyError("optimizer_state_dict")
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if scheduler is not None:
                if "scheduler_state_dict" not in checkpoint:
                    raise KeyError("scheduler_state_dict")
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if scaler is not None:
                if "scaler_state_dict" not in checkpoint:
                    raise KeyError("scaler_state_dict")
                scaler.load_state_dict(checkpoint["scaler_state_dict"])

        @staticmethod
        def capture_rng_state() -> dict[str, Any]:
            import torch

            state: dict[str, Any] = {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
            }
            if torch.cuda.is_available():
                state["torch_cuda"] = torch.cuda.get_rng_state_all()
            return state

        @staticmethod
        def restore_rng_state(state: dict[str, Any]) -> None:
            import torch

            random.setstate(state["python"])
            np.random.set_state(state["numpy"])
            torch.set_rng_state(state["torch"])
            if torch.cuda.is_available() and "torch_cuda" in state:
                torch.cuda.set_rng_state_all(state["torch_cuda"])

        def load_checkpoint(
            self,
            path: str,
            *,
            model=None,
            optimizer=None,
            scheduler=None,
            scaler=None,
            strict: bool = True,
            map_location=None,
        ) -> dict[str, Any]:
            import torch

            checkpoint = self._validate_checkpoint_payload(
                path,
                torch.load(path, map_location=map_location, weights_only=False),
            )
            self._load_required_state(
                checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                strict=strict,
            )

            return checkpoint

    checkpoint_cfg: CheckpointCfg = field(default_factory=CheckpointCfg)
