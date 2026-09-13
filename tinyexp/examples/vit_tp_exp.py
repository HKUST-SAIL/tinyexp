"""DeiT-S experiment with tensor parallelism, ported against the official DeiT.

This module is written by strict cross-checking (对拍) against upstream sources; only
formatting-level changes are allowed (see ``docs/vit_tp.md``). The ported sources are:

- Model classes (``Mlp``/``Block``/``PatchEmbed``/``VisionTransformer``): timm's
  ``vision_transformer`` implementation, configured to retain the timm 0.3.2 DeiT
  behavior. ``Attention`` is the only model layer adapted locally for TP: its fused
  projection is split into ``q``/``k``/``v``.
- ``deit_small_patch16_224`` factory hyper-parameters and checkpoint URL:
  facebookresearch/deit ``models.py`` at commit ``7e160fe43f0252d17191b71cbb5826254114ea5b``.
- Train/eval loops, recipe defaults, transforms and RASampler wiring: deit ``engine.py``,
  ``main.py``, ``datasets.py`` (same commit).
- Logging statistics (``SmoothedValue``/``MetricLogger``): deit ``utils.py`` (detr lineage).

Sanctioned deviation (docs/vit_tp.md D1): the official fused ``attn.qkv`` Linear is split
into separate ``attn.q``/``attn.k``/``attn.v`` Linears so that ``ColwiseParallel`` can
shard attention on head boundaries; ``convert_fused_qkv_to_split`` /
``convert_split_qkv_to_fused`` translate weights between the two layouts, and the unit
tests prove numerical equivalence of the split forward against the fused original.

Sanctioned deviation (docs/vit_tp.md D2): ``Attention.forward`` reshapes use ``-1``
(local-shape aware) to stay correct under colwise sharding; at tp=1 the resolved shapes
are exactly the official ones (guarded by the fused-oracle unit tests).

Sanctioned deviation (docs/vit_tp.md D7): the training set can optionally be served
through a Redis raw-byte cache (``redis_cfg.redis_cache_enabled``, default off) for
slow shared filesystems; samples stay bit-identical to ``datasets.ImageFolder``.

Cross-check results (full record in ``docs/vit_tp.md``): TP=2 is numerically
equivalent to single-rank (CI); the official checkpoint evaluates to 79.82/79.81
top-1 at tp=1/tp=2 (official 79.8); the full 300-epoch recipe trained from scratch
(8xH200 DDP via rjob, Redis cache on) reached **79.84% top-1 / 94.99% top-5**
(paper: 79.8 / ~95.0).

Usage (full recipe, 2 GPUs, TP=2)::

    # ImageNet must use the standard ImageFolder layout with train/ and val/
    # class directories.  The same variable is used by the ResNet example.
    export IMAGENET_HOME=/path/to/imagenet

    python -m tinyexp.examples.vit_tp_exp

Eval-only against the official checkpoint (accuracy cross-check)::

    export IMAGENET_HOME=/path/to/imagenet
    python -m tinyexp.examples.vit_tp_exp mode=eval module_cfg.pretrained_from=<ckpt-or-url>

Full 300-epoch training on 8 GPUs with DDP and the optional Redis byte cache::

    export IMAGENET_HOME=/path/to/imagenet
    python -m tinyexp.examples.vit_tp_exp accelerator_cfg.accelerator=ddp \\
        ray_cfg.ray_num_worker=8 redis_cfg.redis_cache_enabled=true redis_cfg.redis_cache_max_memory=300

Throughput / memory benchmark, e.g. TP vs DDP::

    python -m tinyexp.examples.vit_tp_exp mode=bench
    python -m tinyexp.examples.vit_tp_exp mode=bench accelerator_cfg.accelerator=ddp

CPU smoke (no ImageNet on disk, ray 2 workers, gloo TP=2)::

    python -m tinyexp.examples.vit_tp_exp dataloader_cfg.fake_data=true dataloader_cfg.input_size=32 \\
        module_cfg.img_size=32 module_cfg.num_classes=10 epochs=1 max_train_steps=4 \\
        ray_cfg.ray_num_gpus_per_worker=0.0 ray_cfg.ray_num_cpus_per_worker=4

Not ported: ``HybridEmbed``, distilled models, distillation, ``--cosub``,
``--ThreeAugment``, ``--bce-loss``, ``--attn-only``, INAT/CIFAR datasets, submitit,
position-embedding interpolation for non-224 finetuning.
"""

from __future__ import annotations

import datetime
import io
import json
import math
import os
import sys
import time
import weakref
from collections import defaultdict, deque
from dataclasses import dataclass, field
from functools import partial
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, Mixup, create_transform
from timm.layers import trunc_normal_
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models.vision_transformer import Attention as TimmAttention
from timm.models.vision_transformer import Block as TimmBlock
from timm.models.vision_transformer import VisionTransformer as TimmVisionTransformer
from timm.scheduler import create_scheduler
from timm.utils import ModelEma, NativeScaler, accuracy, get_state_dict
from torch.distributed.tensor.parallel import ColwiseParallel, ParallelStyle, RowwiseParallel
from torchvision import datasets, transforms

from tinyexp import TinyExp, store_and_run_exp
from tinyexp.dataset.ra_sampler import RASampler
from tinyexp.exceptions import UnknownAcceleratorTypeError
from tinyexp.exp_mixins import CheckpointCfgMixin, LoggerCfgMixin, RayCfgMixin, RedisCfgMixin, WandbCfgMixin
from tinyexp.tiny_engine.accelerator import AcceleratorProtocol, TPAccelerator

# Official DeiT-S checkpoint (facebookresearch/deit models.py).
DEIT_SMALL_PATCH16_224_CKPT_URL = "https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth"


class _SplitQkv(nn.Module):
    """Compose split projections into the fused tensor expected by timm Attention."""

    def __init__(self, attention: Attention) -> None:
        super().__init__()
        self._attention = weakref.ref(attention)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self._attention()
        q, k, v = attention.q(x), attention.k(x), attention.v(x)
        # ColwiseParallel returns local plain tensors. timm's inherited forward
        # then reshapes this local QKV tensor using the local head count.
        attention.attn_dim = q.shape[-1]
        attention.num_heads = attention.attn_dim // attention.head_dim
        return torch.cat((q, k, v), dim=-1)


class Attention(TimmAttention):
    """timm Attention with split projections for tensor parallelism."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        proj_bias: bool = True,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("qk_norm", None)
        kwargs.pop("scale_norm", None)
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=False,
            scale_norm=False,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            **kwargs,
        )
        del self.qkv
        self.q = nn.Linear(dim, self.attn_dim, bias=qkv_bias)
        self.k = nn.Linear(dim, self.attn_dim, bias=qkv_bias)
        self.v = nn.Linear(dim, self.attn_dim, bias=qkv_bias)
        self.qkv = _SplitQkv(self)
        if qk_scale is not None:
            self.scale = qk_scale


class Block(TimmBlock):
    """timm block that keeps the old Attention call signature when unmasked."""

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        if attn_mask is None and not is_causal:
            x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        else:
            x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class VisionTransformer(TimmVisionTransformer):
    """DeiT-compatible timm ViT with the TP attention adapter above."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            global_pool="token",
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_path_rate=drop_path_rate,
            pos_drop_rate=drop_rate,
            proj_drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            norm_layer=norm_layer,
            block_fn=Block,
            attn_layer=Attention,
            weight_init="skip",
            fc_norm=False,
            **kwargs,
        )
        if qk_scale is not None:
            for block in self.blocks:
                block.attn.scale = qk_scale
        self._init_old_weights()

    def _init_old_weights(self) -> None:
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self) -> set[str]:
        return {"pos_embed", "cls_token"}


def deit_small_patch16_224(pretrained: bool = False, **kwargs: Any) -> VisionTransformer:
    """Build DeiT-S with the official hyper-parameters (facebookresearch/deit models.py).

    With ``pretrained=True`` the official checkpoint is loaded through the qkv-split
    converter of docs/vit_tp.md D1.
    """
    model = VisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url=DEIT_SMALL_PATCH16_224_CKPT_URL, map_location="cpu", check_hash=True
        )
        model.load_state_dict(convert_fused_qkv_to_split(checkpoint["model"]))
    return model


def build_tp_parallelize_plan(module: VisionTransformer, tp_size: int) -> dict[str, ParallelStyle]:
    """Build the Megatron-style colwise/rowwise plan for the ported VisionTransformer (docs/vit_tp.md).

    Head divisibility is the hard structural requirement of TP attention: ColwiseParallel
    shards the ``q``/``k``/``v`` output dim, and a shard boundary must not fall inside a
    head (that is exactly why the fused ``qkv`` Linear had to be split, docs/vit_tp.md D1).
    """
    num_heads = module.blocks[0].attn.num_heads
    if tp_size < 1:
        raise ValueError(f"tp_size must be >= 1, got {tp_size}")
    if num_heads % tp_size != 0:
        raise ValueError(f"num_heads={num_heads} must be divisible by tp_size={tp_size} to shard on head boundaries")
    plan: dict[str, ParallelStyle] = {}
    for i in range(len(module.blocks)):
        plan.update(
            {
                f"blocks.{i}.attn.q": ColwiseParallel(),
                f"blocks.{i}.attn.k": ColwiseParallel(),
                f"blocks.{i}.attn.v": ColwiseParallel(),
                # Rowwise proj consumes the Shard(-1) head chunk and all-reduces back to
                # a replicated output: one forward all-reduce per attention.
                f"blocks.{i}.attn.proj": RowwiseParallel(),
                f"blocks.{i}.mlp.fc1": ColwiseParallel(),
                f"blocks.{i}.mlp.fc2": RowwiseParallel(),
            }
        )
    return plan


def convert_fused_qkv_to_split(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert an official (fused-qkv) DeiT/ViT state dict into the split layout of ``Attention``.

    The fused ``attn.qkv`` weight has shape ``(3 * dim, dim)`` with rows ordered
    ``[q, k, v]`` (timm 0.3.2 reshapes the ``(B, N, 3 * dim)`` output with the factor of
    3 leading), so contiguous thirds map to the three split Linears.
    """
    converted: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if ".attn.qkv." in key:
            prefix, suffix = key.split(".attn.qkv.")
            third = value.shape[0] // 3
            converted[f"{prefix}.attn.q.{suffix}"] = value[:third].clone()
            converted[f"{prefix}.attn.k.{suffix}"] = value[third : 2 * third].clone()
            converted[f"{prefix}.attn.v.{suffix}"] = value[2 * third :].clone()
        else:
            converted[key] = value
    return converted


def convert_split_qkv_to_fused(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Inverse of :func:`convert_fused_qkv_to_split`: emit official checkpoint layout."""
    converted = dict(state_dict)
    for key in state_dict:
        if not key.endswith(".attn.q.weight"):
            continue
        prefix = key[: -len(".attn.q.weight")]
        for stem in ("weight", "bias"):
            q = state_dict[f"{prefix}.attn.q.{stem}"]
            k = state_dict[f"{prefix}.attn.k.{stem}"]
            v = state_dict[f"{prefix}.attn.v.{stem}"]
            converted[f"{prefix}.attn.qkv.{stem}"] = torch.cat([q, k, v], dim=0)
            del converted[f"{prefix}.attn.q.{stem}"]
            del converted[f"{prefix}.attn.k.{stem}"]
            del converted[f"{prefix}.attn.v.{stem}"]
    return converted


class _FusedQkvModelProxy:
    """Adapter that reads/writes the model in the official fused-qkv checkpoint layout.

    tinyexp checkpoints therefore stay byte-compatible with facebookresearch/deit
    checkpoints (``model_state_dict`` holds fused ``attn.qkv`` keys) regardless of the
    accelerator in use.
    """

    def __init__(self, module: nn.Module, accelerator: AcceleratorProtocol | None = None) -> None:
        self.module = module
        self.accelerator = accelerator

    def state_dict(self) -> dict[str, torch.Tensor]:
        dump = getattr(self.accelerator, "dump_model_to_state_dict", None) if self.accelerator is not None else None
        if dump is not None:
            state = dump(self.module)
        else:  # CPUAccelerator / plain module: state_dict is already whole.
            state = {key: value.cpu() for key, value in self.module.state_dict().items()}
        return convert_split_qkv_to_fused(state)

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True) -> Any:
        # Match the official finetune behavior: ignore classifier weights when
        # the checkpoint and configured class count differ.
        state_dict = dict(state_dict)
        model_state = self.module.state_dict()
        for key in ("head.weight", "head.bias"):
            if key in state_dict and key in model_state and state_dict[key].shape != model_state[key].shape:
                print(f"Removing key {key} from pretrained checkpoint")
                del state_dict[key]
        return self.module.load_state_dict(convert_fused_qkv_to_split(state_dict), strict=strict)


def _tensors_to_plain(state: Any) -> Any:
    """Recursively convert DTensors to plain whole cpu tensors.

    ``full_tensor()`` is a collective: under TP every rank must call this on the same
    object graph at the same time.
    """
    from torch.distributed.tensor import DTensor

    if isinstance(state, DTensor):
        return state.full_tensor().cpu()
    if isinstance(state, dict):
        return {key: _tensors_to_plain(value) for key, value in state.items()}
    if isinstance(state, list):
        return [_tensors_to_plain(item) for item in state]
    if isinstance(state, tuple):
        return tuple(_tensors_to_plain(item) for item in state)
    if isinstance(state, torch.Tensor):
        return state.cpu()
    return state


class _PrecomputedStateDict:
    """Duck-typed stand-in so CheckpointCfg.save_checkpoint stores an already-dumped dict."""

    def __init__(self, state_dict: dict) -> None:
        self._state_dict = state_dict

    def state_dict(self) -> dict:
        return self._state_dict


# ---------------------- metric logging (deit utils.py, verbatim) ---------------------- #


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size: int = 20, fmt: str | None = None) -> None:
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque: deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value: float, n: int = 1) -> None:
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self) -> float:
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self) -> float:
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self) -> float:
        return self.total / self.count

    @property
    def max(self) -> float:
        return max(self.deque)

    @property
    def value(self) -> float:
        return self.deque[-1]

    def __str__(self) -> str:
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )


class MetricLogger:
    def __init__(self, delimiter: str = "\t", log_fn: Callable[..., None] = print) -> None:
        self.meters: defaultdict = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.log_fn = log_fn

    def update(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            self.meters[key].update(value)

    def __getattr__(self, attr: str) -> SmoothedValue:
        if attr in self.meters:
            return self.meters[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self) -> str:
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {meter!s}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self, accelerator: AcceleratorProtocol) -> None:
        """detr-style sync: all-reduce (total, count) per meter, then average."""
        if getattr(accelerator, "world_size", 1) < 2:
            return
        for meter in self.meters.values():
            pair = accelerator.reduce_sum(torch.tensor([meter.total, meter.count], dtype=torch.float64))
            meter.total = float(pair[0].item()) / accelerator.world_size
            meter.count = float(pair[1].item()) / accelerator.world_size

    def add_meter(self, name: str, meter: SmoothedValue) -> None:
        self.meters[name] = meter

    def log_every(self, iterable: Any, print_freq: int, header: str | None = None) -> Any:
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                ]
            )
        i = 0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    self.log_fn(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                        )
                    )
                else:
                    self.log_fn(
                        log_msg.format(
                            i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time)
                        )
                    )
            i += 1  # noqa: SIM113  (official utils.py counter)
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.log_fn(f"{header} Total time: {total_time_str} ({total_time / max(len(iterable), 1):.4f} s / it)")


class RedisCachedImageFolder:
    """ImageFolder wrapper caching raw JPEG bytes in Redis (resnet_exp pattern).

    Authorized deviation (docs/vit_tp.md D7): IO layer only — cache stores the exact
    file bytes and the decode/transform pipeline is unchanged, so samples are
    bit-identical to ``datasets.ImageFolder``. First epoch fills the cache from
    disk (misses), later epochs read from Redis (hits).
    """

    def __init__(
        self,
        redis_host: str,
        redis_ports: list[int],
        root: str,
        transform=None,
        target_transform=None,
        redis_world_size: int = 1,
    ):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.dataset = datasets.ImageFolder(root)

        self.cache_misses = 0
        self.cache_hits = 0
        from tinyexp.utils.redis_utils import RedisClientManager

        self.redis_client_manager = RedisClientManager(redis_host, redis_ports, redis_world_size)

    def __getitem__(self, index):
        path, target = self.dataset.samples[index]
        cache_key = index

        file_data = self.redis_client_manager.safe_get(cache_key)
        if file_data is None:
            self.cache_misses += 1
            with open(path, "rb") as f:
                file_data = f.read()
            self.redis_client_manager.safe_set(cache_key, file_data)
        else:
            self.cache_hits += 1

        image = Image.open(io.BytesIO(file_data)).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target

    def __len__(self):
        return len(self.dataset)


# ------------------------------ experiment (docs/vit_tp.md) ------------------------------ #
# The Exp wiring below maps facebookresearch/deit main.py/engine.py/datasets.py onto the
# tinyexp Cfg-component layout; recipe defaults are the official argparse defaults.


def _set_cudnn_benchmark(value: bool = True) -> None:
    """official main.py: ``import torch.backends.cudnn as cudnn; cudnn.benchmark = True``.

    The local import is also what makes this picklable for Ray: a direct
    ``torch.backends.cudnn`` attribute chain is captured by cloudpickle and
    ``CudnnModule`` cannot be pickled.
    """
    import torch.backends.cudnn as cudnn

    cudnn.benchmark = value


@dataclass(repr=False)
class VitTpExp(TinyExp, RayCfgMixin, RedisCfgMixin, CheckpointCfgMixin, WandbCfgMixin, LoggerCfgMixin):
    mode: str = "train"  # train / eval / bench
    launcher: str = "ray"
    epochs: int = 300  # official --epochs
    max_train_steps: int = -1  # smoke runs only; official has no step cap
    seed: int = 0  # official --seed
    # DDP offsets the seed per rank (official: seed + rank) for data shuffling. TP ranks
    # consume the *same* batches and must keep identical RNG state, so all ranks share
    # the seed under accelerator="tp" (docs/vit_tp.md D6).

    @dataclass
    class RayCfg(RayCfgMixin.RayCfg):
        ray_num_worker: int = 2
        ray_num_gpus_per_worker: float = 1.0
        ray_num_cpus_per_worker: int = 12  # main process plus 10 dataloader workers

    ray_cfg: RayCfg = field(default_factory=RayCfg)

    @dataclass
    class RedisCfg(RedisCfgMixin.RedisCfg):
        # Opt-in (resnet_exp defaults to True): the official deit pipeline is the
        # verbatim default; the Redis byte cache is enabled per run for slow
        # filesystems (docs/vit_tp.md D7). No effect on fake_data runs.
        redis_cache_enabled: bool = False

    redis_cfg: RedisCfg = field(default_factory=RedisCfg)

    @dataclass
    class AcceleratorCfg:
        accelerator: str = "tp"

        def build_accelerator(self) -> AcceleratorProtocol:
            from tinyexp.tiny_engine.accelerator import CPUAccelerator, DDPAccelerator

            if self.accelerator == "cpu":
                return CPUAccelerator()
            if self.accelerator == "ddp":
                return DDPAccelerator()
            if self.accelerator == "tp":
                return TPAccelerator()
            raise UnknownAcceleratorTypeError(self.accelerator)

    accelerator_cfg: AcceleratorCfg = field(default_factory=AcceleratorCfg)

    @dataclass
    class ModuleCfg:
        # factory kwargs from facebookresearch/deit models.py (depth 12 shared by all)
        model_name: str = "deit_small_patch16_224"  # also: deit_tiny_/deit_base_patch16_224
        img_size: int = 224
        num_classes: int = 1000
        drop_rate: float = 0.0  # official --drop
        drop_path_rate: float = 0.1  # official --drop-path
        # official checkpoint (URL or path) loaded through the D1 converter; mismatched
        # classifier heads are dropped (official --finetune branch)
        pretrained_from: str = ""

        def build_module(self) -> VisionTransformer:
            factory = {
                "deit_tiny_patch16_224": 192,
                "deit_small_patch16_224": 384,
                "deit_base_patch16_224": 768,
            }
            if self.model_name not in factory:
                raise ValueError(f"unknown model_name {self.model_name}")
            heads = {"deit_tiny_patch16_224": 3, "deit_small_patch16_224": 6, "deit_base_patch16_224": 12}[
                self.model_name
            ]
            return VisionTransformer(
                img_size=self.img_size,
                patch_size=16,
                embed_dim=factory[self.model_name],
                depth=12,
                num_heads=heads,
                mlp_ratio=4,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                drop_rate=self.drop_rate,
                drop_path_rate=self.drop_path_rate,
                num_classes=self.num_classes,
            )

    module_cfg: ModuleCfg = field(default_factory=ModuleCfg)

    @dataclass
    class DataloaderCfg:
        data_root: str = os.environ.get("IMAGENET_HOME", "./data/imagenet/")
        input_size: int = 224
        # 256/rank * 2 ranks = 512 global batch -> lr 5e-4 via the official linear rule
        # (docs/vit_tp.md), the same scaling point as the official 1024 @ lr 1e-3.
        train_batch_size_per_device: int = 256
        val_batch_size_per_device: int = 384  # int(1.5 * batch), official main.py
        num_workers: int = 10
        color_jitter: float = 0.3
        auto_augment: str = "rand-m9-mstd0.5-inc1"
        train_interpolation: str = "bicubic"
        reprob: float = 0.25
        remode: str = "pixel"
        recount: int = 1
        repeated_aug: bool = True
        num_repeats: int = 3  # RASampler default
        dist_eval: bool = False  # official default; forced off under TP (D5)
        eval_crop_ratio: float = 0.875
        # synthetic tensors instead of ImageNet: smoke tests and mode=bench, no disk data
        fake_data: bool = False
        fake_data_len: int = 64
        pin_mem: bool = True

        def _build_transform(self, is_train: bool) -> transforms.Compose:
            # official datasets.py build_transform (IMNET branch)
            resize_im = self.input_size > 32
            if is_train:
                # this should always dispatch to transforms_imagenet_train
                transform = create_transform(
                    input_size=self.input_size,
                    is_training=True,
                    color_jitter=self.color_jitter,
                    auto_augment=self.auto_augment,
                    interpolation=self.train_interpolation,
                    re_prob=self.reprob,
                    re_mode=self.remode,
                    re_count=self.recount,
                )
                if not resize_im:
                    # replace RandomResizedCropAndInterpolation with RandomCrop
                    transform.transforms[0] = transforms.RandomCrop(self.input_size, padding=4)
                return transform
            t = []
            if resize_im:
                size = int(self.input_size / self.eval_crop_ratio)
                t.append(
                    # official passes interpolation=3 (BICUBIC)
                    transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
                )
                t.append(transforms.CenterCrop(self.input_size))
            t.append(transforms.ToTensor())
            t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
            return transforms.Compose(t)

        def _build_dataset(self, is_train: bool, redis_cfg=None) -> torch.utils.data.Dataset:
            if self.fake_data:
                return torch.utils.data.TensorDataset(
                    torch.randn(self.fake_data_len, 3, self.input_size, self.input_size),
                    torch.randint(0, 2, (self.fake_data_len,)),
                )
            # official datasets.py build_dataset IMNET branch
            root = os.path.join(self.data_root, "train" if is_train else "val")
            transform = self._build_transform(is_train)
            if is_train and redis_cfg is not None and redis_cfg.redis_cache_enabled:
                # docs/vit_tp.md D7: Redis byte cache for slow filesystems; samples are
                # bit-identical to datasets.ImageFolder (bytes are cached verbatim).
                return RedisCachedImageFolder(
                    redis_host=redis_cfg.redis_cluster_host,
                    redis_ports=list(redis_cfg.redis_cluster_ports),
                    root=root,
                    transform=transform,
                    redis_world_size=int(redis_cfg.redis_rendezvous_world_size),
                )
            return datasets.ImageFolder(root, transform=transform)

        def build_train_dataloader(
            self, accelerator: AcceleratorProtocol, replicate_data: bool, redis_cfg=None
        ) -> torch.utils.data.DataLoader:
            dataset = self._build_dataset(is_train=True, redis_cfg=redis_cfg)
            num_replicas = 1 if replicate_data else accelerator.world_size
            rank = 0 if replicate_data else accelerator.rank
            if num_replicas > 1 and self.repeated_aug:
                sampler: torch.utils.data.Sampler = RASampler(
                    dataset, num_replicas=num_replicas, rank=rank, shuffle=True, num_repeats=self.num_repeats
                )
            elif num_replicas > 1:
                sampler = torch.utils.data.DistributedSampler(
                    dataset, num_replicas=num_replicas, rank=rank, shuffle=True
                )
            else:
                sampler = torch.utils.data.RandomSampler(dataset)
            return torch.utils.data.DataLoader(
                dataset,
                sampler=sampler,
                batch_size=self.train_batch_size_per_device,
                num_workers=self.num_workers,
                pin_memory=self.pin_mem,
                drop_last=True,
            )

        def build_val_dataloader(
            self, accelerator: AcceleratorProtocol, replicate_data: bool
        ) -> torch.utils.data.DataLoader:
            dataset = self._build_dataset(is_train=False)
            if self.dist_eval and not replicate_data:
                sampler: torch.utils.data.Sampler = torch.utils.data.DistributedSampler(
                    dataset, num_replicas=accelerator.world_size, rank=accelerator.rank, shuffle=False
                )
            else:
                # official non-dist-eval: every rank iterates the full validation set
                sampler = torch.utils.data.SequentialSampler(dataset)
            return torch.utils.data.DataLoader(
                dataset,
                sampler=sampler,
                batch_size=self.val_batch_size_per_device,
                num_workers=self.num_workers,
                pin_memory=self.pin_mem,
                drop_last=False,
            )

    dataloader_cfg: DataloaderCfg = field(default_factory=DataloaderCfg)

    @dataclass
    class LossCfg:
        # official mixup/smoothing argparse defaults
        mixup: float = 0.8
        cutmix: float = 1.0
        mixup_prob: float = 1.0
        mixup_switch_prob: float = 0.5
        mixup_mode: str = "batch"
        smoothing: float = 0.1

        def build_mixup_fn(self, num_classes: int) -> Mixup | None:
            mixup_active = self.mixup > 0 or self.cutmix > 0.0
            if not mixup_active:
                return None
            return Mixup(
                mixup_alpha=self.mixup,
                cutmix_alpha=self.cutmix,
                cutmix_minmax=None,  # official default; not exposed (cloudpickle/OmegaConf annotation limits)
                prob=self.mixup_prob,
                switch_prob=self.mixup_switch_prob,
                mode=self.mixup_mode,
                label_smoothing=self.smoothing,
                num_classes=num_classes,
            )

        def build_criterion(self, num_classes: int) -> nn.Module:
            mixup_active = self.mixup > 0 or self.cutmix > 0.0
            criterion = LabelSmoothingCrossEntropy()
            if mixup_active:
                # smoothing is handled with mixup label transform
                criterion = SoftTargetCrossEntropy()
            elif self.smoothing:
                criterion = LabelSmoothingCrossEntropy(smoothing=self.smoothing)
            else:
                criterion = nn.CrossEntropyLoss()
            return criterion

    loss_cfg: LossCfg = field(default_factory=LossCfg)

    @dataclass
    class OptimizerCfg:
        lr: float = 5e-4  # official --lr, linearly scaled below (official --unscale-lr to disable)
        weight_decay: float = 0.05
        opt_eps: float = 1e-8
        clip_grad: float = 0.0  # official default is None (no clipping); <=0 disables here

        def build_optimizer(
            self, module: nn.Module, dataloader, accelerator: AcceleratorProtocol
        ) -> torch.optim.Optimizer:
            linear_scaled_lr = self.lr * dataloader.batch_size * accelerator.world_size / 512.0
            # timm create_optimizer adamw path: params listed in module.no_weight_decay()
            # get zero weight decay
            skip = module.no_weight_decay() if hasattr(module, "no_weight_decay") else set()
            parameters = [
                {
                    "params": [p for n, p in module.named_parameters() if n not in skip and p.requires_grad],
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": [p for n, p in module.named_parameters() if n in skip and p.requires_grad],
                    "weight_decay": 0.0,
                },
            ]
            return torch.optim.AdamW(parameters, lr=linear_scaled_lr, eps=self.opt_eps)

    optimizer_cfg: OptimizerCfg = field(default_factory=OptimizerCfg)

    @dataclass
    class LrSchedulerCfg:
        # official scheduler argparse defaults; fed to timm create_scheduler verbatim
        sched: str = "cosine"
        warmup_epochs: int = 5
        warmup_lr: float = 1e-6
        min_lr: float = 1e-5
        cooldown_epochs: int = 10
        decay_epochs: float = 30
        decay_rate: float = 0.1
        patience_epochs: int = 10
        lr_noise_pct: float = 0.67
        lr_noise_std: float = 1.0

        def build_lr_scheduler(self, optimizer: torch.optim.Optimizer, epochs: int):
            sched_args = SimpleNamespace(
                sched=self.sched,
                epochs=epochs,
                min_lr=self.min_lr,
                warmup_lr=self.warmup_lr,
                warmup_epochs=self.warmup_epochs,
                cooldown_epochs=self.cooldown_epochs,
                decay_epochs=self.decay_epochs,
                decay_rate=self.decay_rate,
                patience_epochs=self.patience_epochs,
                lr_noise=None,  # official default; not exposed (cloudpickle/OmegaConf annotation limits)
                lr_noise_pct=self.lr_noise_pct,
                lr_noise_std=self.lr_noise_std,
                seed=0,
                cycle_mul=1.0,
                cycle_limit=1,
                t_in_epochs=True,
                scale_mode="cycle",
            )
            lr_scheduler, _ = create_scheduler(sched_args, optimizer)
            return lr_scheduler

    lr_scheduler_cfg: LrSchedulerCfg = field(default_factory=LrSchedulerCfg)

    @dataclass
    class EmaCfg:
        model_ema: bool = True  # official default on; forced off at tp>1 (docs/vit_tp.md D4)
        model_ema_decay: float = 0.99996

    ema_cfg: EmaCfg = field(default_factory=EmaCfg)

    @dataclass
    class BenchCfg:
        # docs/vit_tp.md D3: throughput/memory check on synthetic batches (no dataset IO)
        warmup_steps: int = 10
        measure_steps: int = 50

    bench_cfg: BenchCfg = field(default_factory=BenchCfg)

    # ------------------------------ execution part ------------------------------ #

    def run(self) -> None:
        accelerator = self.accelerator_cfg.build_accelerator()
        try:
            self._run(accelerator)
        finally:
            accelerator.destroy()

    def _run(self, accelerator: AcceleratorProtocol) -> None:
        run_dir = self.get_run_dir()
        logger = self.logger_cfg.build_logger(
            save_dir=run_dir, distributed_rank=accelerator.rank, filename="deit_train.log"
        )
        cfg_dict = self.print_cfg(logger)

        if accelerator.device.type == "cuda":
            _set_cudnn_benchmark()  # official main.py
        # official seeds seed + rank; TP keeps all ranks on one seed (docs/vit_tp.md D6)
        seed = self.seed + (accelerator.rank if self.accelerator_cfg.accelerator == "ddp" else 0)
        torch.manual_seed(seed)
        np.random.seed(seed)

        if self.mode == "train":
            self._train(accelerator=accelerator, logger=logger, cfg_dict=cfg_dict, run_dir=run_dir)
        elif self.mode == "eval":
            self._run_eval(accelerator=accelerator, logger=logger)
        elif self.mode == "bench":
            self._bench(accelerator=accelerator, logger=logger)
        else:
            raise NotImplementedError(f"Mode {self.mode} is not implemented")

    def _prepare_for_train_or_eval(self, accelerator):
        """Components shared by train and eval, in official main.py order."""
        replicate_data = self.accelerator_cfg.accelerator == "tp"  # TP ranks eat identical batches (D5/D6)
        dataloader_val = self.dataloader_cfg.build_val_dataloader(accelerator, replicate_data)

        # official args.nb_classes (IMNET branch hard-codes 1000)
        nb_classes = self.module_cfg.num_classes
        mixup_fn = self.loss_cfg.build_mixup_fn(nb_classes)
        criterion = self.loss_cfg.build_criterion(nb_classes)

        model = self.module_cfg.build_module()
        if self.module_cfg.pretrained_from:
            path_or_url = self.module_cfg.pretrained_from
            if path_or_url.startswith("https"):
                checkpoint = torch.hub.load_state_dict_from_url(path_or_url, map_location="cpu", check_hash=True)
            else:
                checkpoint = torch.load(path_or_url, map_location="cpu", weights_only=False)
            checkpoint_model = checkpoint["model"] if "model" in checkpoint else checkpoint.get("model_state_dict")
            _FusedQkvModelProxy(model).load_state_dict(checkpoint_model, strict=False)
        model.to(accelerator.device)

        return replicate_data, dataloader_val, mixup_fn, criterion, model

    def _parallelize(self, accelerator, model, optimizer=None):
        if isinstance(accelerator, TPAccelerator):
            plan = build_tp_parallelize_plan(accelerator.unwrap_model(model), tp_size=accelerator.world_size)
            return accelerator.prepare(model, optimizer, parallelize_plan=plan)
        return accelerator.prepare(model, optimizer)

    def _train(self, accelerator, logger, cfg_dict, run_dir: str) -> None:  # noqa: C901
        replicate_data, data_loader_val, mixup_fn, criterion, model = self._prepare_for_train_or_eval(accelerator)
        data_loader_train = self.dataloader_cfg.build_train_dataloader(accelerator, replicate_data, self.redis_cfg)

        model_ema = None
        if self.ema_cfg.model_ema:
            if accelerator.world_size > 1 and self.accelerator_cfg.accelerator == "tp":
                logger.info("model_ema disabled under tp>1 (docs/vit_tp.md D4)")
            else:
                # Important to create EMA model after cuda() but before TP/DDP wrapper
                model_ema = ModelEma(model, decay=self.ema_cfg.model_ema_decay, device="", resume="")

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"number of params: {n_parameters}")

        optimizer = self.optimizer_cfg.build_optimizer(model, data_loader_train, accelerator)
        loss_scaler = NativeScaler(device=accelerator.device.type)  # GradScaler auto-disables on cpu
        lr_scheduler = self.lr_scheduler_cfg.build_lr_scheduler(optimizer, epochs=self.epochs)

        model, optimizer = self._parallelize(accelerator, model, optimizer)
        model_proxy = _FusedQkvModelProxy(accelerator.unwrap_model(model), accelerator)

        start_epoch = 0
        max_accuracy = 0.0
        if self.resume_from:
            checkpoint = self.checkpoint_cfg.load_checkpoint(
                self.resume_from,
                model=model_proxy,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                scaler=loss_scaler,
                map_location=accelerator.device,
            )
            start_epoch = int(checkpoint.get("epoch", -1)) + 1
            max_accuracy = float(checkpoint.get("best_metric") or 0.0)
            extra_state = checkpoint.get("extra_state") or {}
            if model_ema is not None and extra_state.get("model_ema_state_dict") is not None:
                model_ema.ema.load_state_dict(extra_state["model_ema_state_dict"])
            lr_scheduler.step(start_epoch)

        if self.wandb_cfg.enable_wandb and accelerator.is_main_process:
            self.wandb_cfg.build_wandb(accelerator=accelerator, project="TinyExp", config=cfg_dict, name="vit_tp_exp")

        logger.info(f"Start training for {self.epochs} epochs")
        start_training_time = time.time()
        global_step = 0
        for epoch in range(start_epoch, self.epochs):
            train_sampler = getattr(data_loader_train, "sampler", None)
            if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)

            train_stats = self._train_one_epoch(
                accelerator,
                logger,
                model,
                criterion,
                data_loader_train,
                optimizer,
                accelerator.device,
                epoch,
                loss_scaler,
                self.optimizer_cfg.clip_grad if self.optimizer_cfg.clip_grad > 0 else None,
                model_ema,
                mixup_fn,
            )
            global_step += len(data_loader_train)
            lr_scheduler.step(epoch)

            # official order: last checkpoint every epoch, then evaluate and keep best
            self._save_checkpoint(
                accelerator,
                run_dir,
                self.checkpoint_cfg.last_ckpt_name,
                model_proxy,
                optimizer,
                lr_scheduler,
                loss_scaler,
                model_ema,
                epoch,
                global_step,
                max_accuracy,
            )

            test_stats = self._evaluate(accelerator, logger, model, accelerator.device, data_loader_val)
            logger.info(
                f"Accuracy of the network on the {len(data_loader_val.dataset)} test images: {test_stats['acc1']:.1f}%"
            )

            if max_accuracy < test_stats["acc1"]:
                max_accuracy = test_stats["acc1"]
                self._save_checkpoint(
                    accelerator,
                    run_dir,
                    self.checkpoint_cfg.best_ckpt_name,
                    model_proxy,
                    optimizer,
                    lr_scheduler,
                    loss_scaler,
                    model_ema,
                    epoch,
                    global_step,
                    max_accuracy,
                )
                logger.info(f"Max accuracy: {max_accuracy:.2f}%")

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }
            if accelerator.is_main_process:
                with open(os.path.join(run_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")
                if self.wandb_cfg.enable_wandb:
                    import wandb

                    wandb.log(log_stats)

            if 0 < self.max_train_steps <= global_step:
                break

        total_time = time.time() - start_training_time
        logger.info(f"Training time {datetime.timedelta(seconds=int(total_time))}")

    def _save_checkpoint(
        self,
        accelerator,
        run_dir,
        name,
        model_proxy,
        optimizer,
        lr_scheduler,
        loss_scaler,
        model_ema,
        epoch,
        global_step,
        max_accuracy,
    ) -> None:
        # DTensor -> full_tensor gathers are collectives: every rank must run the dumps,
        # only the main process writes the file (otherwise TP ranks deadlock).
        model_state = model_proxy.state_dict()
        optimizer_state = _tensors_to_plain(optimizer.state_dict())
        if not accelerator.is_main_process:
            return
        self.checkpoint_cfg.save_checkpoint(
            run_dir=run_dir,
            name=name,
            model=_PrecomputedStateDict(model_state),
            optimizer=_PrecomputedStateDict(optimizer_state),
            scheduler=lr_scheduler,
            epoch=epoch,
            global_step=global_step,
            best_metric=max_accuracy,
            exp_name=self.exp_name,
            exp_class=self.exp_class,
            extra_state={
                "scaler_state_dict": loss_scaler.state_dict(),
                # official main.py: get_state_dict(model_ema); ema only exists at world 1
                "model_ema_state_dict": get_state_dict(model_ema) if model_ema is not None else None,
            },
        )

    def _train_one_epoch(
        self,
        accelerator,
        logger,
        model,
        criterion,
        data_loader,
        optimizer,
        device,
        epoch,
        loss_scaler,
        clip_grad=None,
        model_ema=None,
        mixup_fn=None,
    ):
        """Verbatim port of deit engine.train_one_epoch (cosub/bce branches cut, V1)."""
        model.train()
        metric_logger = MetricLogger(log_fn=logger.info)
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        header = f"Epoch: [{epoch}]"
        print_freq = 10

        for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if mixup_fn is not None:
                samples, targets = mixup_fn(samples, targets)

            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                outputs = model(samples)
                loss = criterion(outputs, targets)

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                logger.error(f"Loss is {loss_value}, stopping training")
                sys.exit(1)

            optimizer.zero_grad()

            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, "is_second_order") and optimizer.is_second_order
            loss_scaler(
                loss,
                optimizer,
                clip_grad=clip_grad,
                parameters=model.parameters(),
                create_graph=is_second_order,
            )

            if model_ema is not None:
                model_ema.update(model)

            metric_logger.update(loss=loss_value)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        # gather the stats from all processes
        metric_logger.synchronize_between_processes(accelerator)
        logger.info(f"Averaged stats: {metric_logger}")
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    @torch.no_grad()
    def _evaluate(self, accelerator, logger, model, device, val_dataloader) -> dict:
        """Verbatim port of deit engine.evaluate."""
        criterion = nn.CrossEntropyLoss()

        metric_logger = MetricLogger(log_fn=logger.info)
        header = "Test:"

        # switch to evaluation mode
        model.eval()

        for images, target in metric_logger.log_every(val_dataloader, 10, header):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                output = model(images)
                loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            batch_size = images.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        # gather the stats from all processes
        metric_logger.synchronize_between_processes(accelerator)
        logger.info(
            f"* Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f} loss {metric_logger.loss.global_avg:.3f}"
        )

        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    def _run_eval(self, accelerator, logger) -> float:
        source = self.resume_from or self.module_cfg.pretrained_from
        if not source:
            raise ValueError("mode=eval requires resume_from (tinyexp ckpt) or module_cfg.pretrained_from")
        _, data_loader_val, _, _, model = self._prepare_for_train_or_eval(accelerator)
        if self.resume_from and not self.module_cfg.pretrained_from:
            proxy = _FusedQkvModelProxy(model, accelerator)
            self.checkpoint_cfg.load_checkpoint(self.resume_from, model=proxy, map_location=accelerator.device)
        model = self._parallelize(accelerator, model)
        test_stats = self._evaluate(accelerator, logger, model, accelerator.device, data_loader_val)
        logger.info(
            f"Accuracy of the network on the {len(data_loader_val.dataset)} test images: {test_stats['acc1']:.1f}%"
        )
        self._run_result = f"eval acc@1={test_stats['acc1']:.2f}% acc@5={test_stats['acc5']:.2f}%"
        return test_stats["acc1"]

    def _bench(self, accelerator, logger) -> None:
        """Synthetic throughput/memory check (docs/vit_tp.md L4). No dataset involved."""
        model = self.module_cfg.build_module()
        model.to(accelerator.device)
        if accelerator.device.type == "cuda":
            _set_cudnn_benchmark()
        model = self._parallelize(accelerator, model)
        criterion = nn.CrossEntropyLoss()

        batch_size = self.dataloader_cfg.train_batch_size_per_device
        images = torch.randn(batch_size, 3, self.dataloader_cfg.input_size, self.dataloader_cfg.input_size)
        target = torch.randint(0, self.module_cfg.num_classes, (batch_size,))

        model.train()
        for _ in range(self.bench_cfg.warmup_steps):
            images_d = images.to(accelerator.device, non_blocking=True)
            target_d = target.to(accelerator.device, non_blocking=True)
            with torch.amp.autocast(accelerator.device.type, enabled=accelerator.device.type == "cuda"):
                loss = criterion(model(images_d), target_d)
            model.zero_grad(set_to_none=True)
            accelerator.backward(loss)

        if accelerator.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(self.bench_cfg.measure_steps):
            images_d = images.to(accelerator.device, non_blocking=True)
            target_d = target.to(accelerator.device, non_blocking=True)
            with torch.amp.autocast(accelerator.device.type, enabled=accelerator.device.type == "cuda"):
                loss = criterion(model(images_d), target_d)
            model.zero_grad(set_to_none=True)
            accelerator.backward(loss)
        if accelerator.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        steps = self.bench_cfg.measure_steps
        images_per_s_per_rank = batch_size * steps / elapsed
        # DDP ranks process disjoint batches (global throughput = rank * world), TP
        # ranks cooperate on the same batch (global throughput = one rank's rate).
        effective_global = images_per_s_per_rank * (
            accelerator.world_size if self.accelerator_cfg.accelerator != "tp" else 1
        )
        peak_mem_gb = 0.0
        if accelerator.device.type == "cuda":
            peak_mem_gb = torch.cuda.max_memory_allocated() / 1024**3
        result = (
            f"bench[{self.accelerator_cfg.accelerator} world={accelerator.world_size}] "
            f"batch/rank={batch_size} step={elapsed / steps * 1000:.1f}ms "
            f"img/s/rank={images_per_s_per_rank:.0f} effective_global_img/s={effective_global:.0f} "
            f"peak_mem/rank={peak_mem_gb:.2f}GB"
        )
        logger.info(result)
        self._run_result = result
        accelerator.wait_for_everyone()

    def get_ray_run_result(self) -> str | None:
        return getattr(self, "_run_result", None)


if __name__ == "__main__":
    store_and_run_exp(VitTpExp)
