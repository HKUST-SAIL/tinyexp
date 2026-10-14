"""Interchangeable training backends implementing ``AcceleratorProtocol`` (see ``base_accelerator``).

| class | backend / device | mixed_precision | when to use |
| --- | --- | --- | --- |
| `CPUAccelerator` | gloo, CPU | none, bf16 | debugging, CI, CPU-only training (DDP when world_size > 1) |
| `DDPAccelerator` | nccl, CUDA required | none, fp16, bf16 | standard multi-GPU data parallel |
| `FSDPAccelerator` | nccl, CUDA required | none, fp16, bf16 | shard params/grads/optimizer state (FULL_SHARD) |
| `TPAccelerator` | nccl on CUDA, else gloo | n/a | tensor parallel, tp_size == world_size (`vit_tp_exp`) |
| `HFAccelerator` | Hugging Face Accelerate | accelerate's args | when launching with `accelerate launch` |

All read ``RANK``/``WORLD_SIZE``/``LOCAL_RANK`` from the environment, so construct them inside the
worker (inside ``Exp.run()``), never in the driver. ``HFAccelerator`` exists only when ``accelerate`` is
installed. Experiments usually select one from config, e.g. ``accelerator_cfg.accelerator=ddp``.

Exports: the five classes above, ``AcceleratorProtocol``, ``BaseAccelerator``.
"""

# flake8: noqa: F401, F403
import contextlib as _contextlib

from .base_accelerator import AcceleratorProtocol, BaseAccelerator
from .cpu_accelerator import CPUAccelerator
from .ddp_accelerator import DDPAccelerator
from .fsdp_accelerator import FSDPAccelerator
from .tp_accelerator import TPAccelerator

with _contextlib.suppress(ImportError):
    from .hf_accelerator import HFAccelerator

_EXCLUDE = {}
__all__ = [k for k in globals() if k not in _EXCLUDE and not k.startswith("_")]
