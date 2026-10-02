"""MAE ViT-B pretraining and evaluation, ported from facebookresearch/mae.

This module is written by strict cross-checking (对拍) against upstream sources; only
formatting-level changes are allowed (see ``docs/mae.md``). The ported sources are:

- ``MaskedAutoencoderViT`` and ``mae_vit_base_patch16`` factory: mae ``models_mae.py``.
- Pretraining loop and recipe: mae ``main_pretrain.py`` (seeding, transform, optimizer,
  checkpoint cadence, per-epoch log stats) and ``engine_pretrain.py``
  ``train_one_epoch`` (per-iteration LR, autocast, loss scaler, grad accumulation).
- LR schedule: mae ``util/lr_sched.py`` ``adjust_learning_rate`` (linear warmup +
  half-cycle cosine, per-iteration, ``lr_scale`` aware).
- Weight-decay grouping: timm ``optim_factory.add_weight_decay`` (timm >= 1.0 spelling:
  ``timm.optim.param_groups_weight_decay``, same grouping semantics).
- ``VisionTransformer`` (global-pool classifier) and ``vit_base_patch16`` factory: mae
  ``models_vit.py``, which subclasses timm's VisionTransformer. Upstream pins
  ``timm==0.3.2``; the timm base here is >= 1.0, configured to reproduce the 0.3.2
  parameter layout and initialization (docs/mae.md D1/D2).
- Checkpoint loading: mae ``main_finetune.py`` finetune branch (drop mismatched head,
  ``interpolate_pos_embed``, ``strict=False``, missing-key assert, manual head init)
  merged with the ``util/misc.py`` ``load_model`` resume branch (URL/local path,
  ``checkpoint['model']``) — the FINETUNE.md sanity check runs the resume branch
  (docs/mae.md D3).
- ``get_2d_sincos_pos_embed`` and ``interpolate_pos_embed``: mae ``util/pos_embed.py``.
- Transforms: mae ``main_pretrain.py`` inline ``transform_train``; ``util/datasets.py``
  ``build_transform(is_train=False)``.
- Eval loop ``_evaluate``: mae ``engine_finetune.py``. ``SmoothedValue``/``MetricLogger``
  and ``_set_cudnn_benchmark`` are reused from ``vit_tp_exp``, which ported them from the
  same detr-lineage upstream (deit/mae ``utils.py``/``misc.py``).
- Train-set IO: the ``RedisCachedImageFolder`` byte cache (resnet_exp pattern, also
  reused from ``vit_tp_exp``) — first epoch fills the cache from disk, later epochs read
  raw file bytes from Redis. Samples stay bit-identical to ``datasets.ImageFolder``
  (docs/mae.md D11). Enabled by default like resnet_exp; turn off with
  ``redis_cfg.redis_cache_enabled=false``.

Cross-check targets:

- Eval (mae FINETUNE.md sanity check, ViT-B)::

      python main_finetune.py --eval --resume mae_finetuned_vit_base.pth \
          --model vit_base_patch16 --batch_size 16
      # Acc@1 83.664  Acc@5 96.530  loss 0.731

  Result (RTX 4080, batch 64): Acc@1 83.746 Acc@5 96.540 loss 0.731 — loss matches to
  three decimals, the accuracy delta (+0.08/+0.01) is fp16-autocast reduction-order
  variance across GPU architectures (full record in ``docs/mae.md``).

- Pretraining: the official PRETRAIN.md ViT-B recipe (batch 64/GPU, mask ratio 0.75,
  800 epochs, 40 warmup epochs, blr 1.5e-4, weight decay 0.05, norm_pix_loss). Local
  multi-GPU runs are smoke checks only (跑通); the full recipe belongs on a cluster.

Usage (ImageNet must use the standard ImageFolder layout with train/ and val/ class
directories; set ``IMAGENET_HOME``)::

    export IMAGENET_HOME=/path/to/imagenet

Pretrain on 2 GPUs (Ray workers, the default launcher)::

    python -m tinyexp.examples.mae_exp ray_cfg.ray_num_worker=2

Pretrain under an external launcher such as torchrun (launcher=mp)::

    torchrun --standalone --nproc-per-node=2 \
      -m tinyexp.examples.mae_exp launcher=mp

Resume from the last checkpoint (``output/<exp_name>/last.ckpt`` is rewritten every
epoch and restores model/optimizer/scaler plus the epoch counter)::

    python -m tinyexp.examples.mae_exp ray_cfg.ray_num_worker=2 \
      resume_from=output/mae_exp/last.ckpt

Quick real-data training smoke on 2 GPUs (one epoch, 20 steps)::

    python -m tinyexp.examples.mae_exp ray_cfg.ray_num_worker=2 epochs=1 max_train_steps=20

Eval cross-check against the official finetuned checkpoint (downloaded once into
``~/.cache/torch/hub/checkpoints/``)::

    python -m tinyexp.examples.mae_exp mode=eval \
      module_cfg.pretrained_from=https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_base.pth

CPU smoke (no ImageNet, no download, random weights; img_size/input_size must stay
divisible by the patch size 16)::

    python -m tinyexp.examples.mae_exp accelerator_cfg.accelerator=cpu \
        module_cfg.img_size=32 dataloader_cfg.fake_data=true dataloader_cfg.input_size=32 \
        dataloader_cfg.train_num_workers=0 dataloader_cfg.val_num_workers=0 \
        dataloader_cfg.fake_data_len=8 epochs=1 max_train_steps=4 redis_cfg.redis_cache_enabled=false \
        ray_cfg.ray_num_gpus_per_worker=0.0 ray_cfg.ray_num_cpus_per_worker=4

Not ported (this round): the finetuning and linear-probing training loops,
``mae_vit_large_patch16``/``mae_vit_huge_patch14``/``vit_large_patch16``/``vit_huge_patch14``,
layer-wise lr decay, submitit, and tensorboard (``wandb_cfg.enable_wandb=true`` covers
remote logging instead).
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import trunc_normal_
from timm.models.vision_transformer import Block, PatchEmbed
from timm.models.vision_transformer import VisionTransformer as TimmVisionTransformer
from timm.optim import param_groups_weight_decay
from timm.utils import NativeScaler, accuracy
from torchvision import datasets, transforms

from tinyexp import TinyExp, store_and_run_exp
from tinyexp.examples.vit_tp_exp import MetricLogger, RedisCachedImageFolder, SmoothedValue, _set_cudnn_benchmark
from tinyexp.exceptions import UnknownAcceleratorTypeError
from tinyexp.exp_mixins import CheckpointCfgMixin, LoggerCfgMixin, RayCfgMixin, RedisCfgMixin, WandbCfgMixin
from tinyexp.tiny_engine.accelerator import AcceleratorProtocol

# Official MAE fine-tuned ViT-B checkpoint (facebookresearch/mae FINETUNE.md).
MAE_FINETUNED_VIT_BASE_CKPT_URL = "https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_base.pth"


# ---------------------- classifier (mae models_vit.py) ---------------------- #


class VisionTransformer(TimmVisionTransformer):
    """MAE ViT with support for global average pooling (facebookresearch/mae models_vit.py)."""

    def __init__(self, global_pool: bool = True, **kwargs: Any) -> None:
        super().__init__(
            # official models_vit.py takes a bool global_pool; timm>=1.0 spells it "avg"/"token".
            # "avg" makes timm build fc_norm and replace norm with nn.Identity(), which is the
            # official ``del self.norm`` layout (docs/mae.md D2).
            global_pool="avg" if global_pool else "token",
            weight_init="skip",  # timm 0.3.2 defaults are applied by _init_old_weights (docs/mae.md D1)
            **kwargs,
        )
        self._init_old_weights()

    def _init_weights(self, module: nn.Module) -> None:
        # timm 0.3.2 default init (vit_tp_exp precedent)
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def _init_old_weights(self) -> None:
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool == "avg":  # official: ``if self.global_pool:`` (a bool there)
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # timm>=1.0 forward() passes attention kwargs into forward_features, which the official
        # signature does not accept; the official head call is spelled out instead. Proven
        # identical to timm's native path by the unit tests (docs/mae.md D2).
        return self.head(self.forward_features(x))


def vit_base_patch16(**kwargs: Any) -> VisionTransformer:
    """Build ViT-B/16 with the official hyper-parameters (mae models_vit.py)."""
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


# ---------------------- masked autoencoder (mae models_mae.py) ---------------------- #


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = False) -> np.ndarray:
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=float)  # official: np.float (removed in numpy >= 1.24, docs/mae.md D7)
    grid_w = np.arange(grid_size, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)  # official: np.float (docs/mae.md D7)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone (facebookresearch/mae models_mae.py)."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        norm_pix_loss: bool = False,
    ) -> None:
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos

        # official passes qk_scale=None positionally; timm>=1.0 Block has no such parameter
        # and the value equals the default head_dim**-0.5 scale (docs/mae.md D1)
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for _ in range(depth)]
        )
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )  # fixed sin-cos

        self.decoder_blocks = nn.ModuleList(
            [
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self) -> None:
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.patch_embed.num_patches**0.5), cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**0.5), cls_token=True
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x: torch.Tensor, mask_ratio: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x: torch.Tensor, mask_ratio: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs: torch.Tensor, mask_ratio: float = 0.75) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def mae_vit_base_patch16_dec512d8b(**kwargs: Any) -> MaskedAutoencoderViT:
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks


# ---------------------- checkpoint loading (mae main_finetune.py / util/misc.py) ---------------------- #


def interpolate_pos_embed(model: nn.Module, checkpoint_model: dict[str, torch.Tensor]) -> None:
    """Interpolate patch position embeddings to the model's grid (mae util/pos_embed.py, verbatim).

    Class/extra tokens are kept unchanged; only the patch tokens are bicubic-interpolated.
    The in-place ``checkpoint_model['pos_embed']`` update is part of the official behavior.
    """
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches**0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print(f"Position interpolate from {orig_size}x{orig_size} to {new_size}x{new_size}")
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model["pos_embed"] = new_pos_embed


def load_pretrained_checkpoint(model: nn.Module, source: str) -> None:
    """Load an official MAE checkpoint (URL or local path) into ``model``.

    Merge of the mae ``main_finetune.py`` finetune branch and ``util/misc.py`` ``load_model``
    resume branch: for the finetuned ViT-B checkpoint both official paths assign identical
    weights (docs/mae.md D3). Works for both model families: ``head.*`` keys that the model
    does not have (e.g. a classifier checkpoint into the autoencoder) are simply unexpected.
    """
    if source.startswith("https"):
        checkpoint = torch.hub.load_state_dict_from_url(source, map_location="cpu", check_hash=True)
    else:
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    print(f"Load pre-trained checkpoint from: {source}")
    checkpoint_model = checkpoint["model"]
    state_dict = model.state_dict()

    dropped_head = False
    for key in ("head.weight", "head.bias"):
        if key in checkpoint_model and key in state_dict and checkpoint_model[key].shape != state_dict[key].shape:
            print(f"Removing key {key} from pretrained checkpoint")
            del checkpoint_model[key]
            dropped_head = True

    # interpolate position embedding
    interpolate_pos_embed(model, checkpoint_model)

    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)
    expected = {"head.weight", "head.bias"}
    if getattr(model, "global_pool", "") == "avg":  # official: ``if args.global_pool``
        expected |= {"fc_norm.weight", "fc_norm.bias"}
    # official asserts equality against the pretrain-checkpoint set; the merged loader accepts
    # its subsets so the finetuned checkpoint (missing == set()) also passes (docs/mae.md D3)
    assert set(msg.missing_keys) <= expected, f"Unexpected missing keys: {msg.missing_keys}"
    if dropped_head:
        # manually initialize fc layer (official finetune branch)
        trunc_normal_(model.head.weight, std=2e-5)


# ---------------------- lr schedule (mae util/lr_sched.py) ---------------------- #


def adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    epoch: float,
    *,
    lr: float,
    min_lr: float,
    warmup_epochs: int,
    epochs: int,
) -> float:
    """Decay the learning rate with half-cycle cosine after warmup (mae util/lr_sched.py, verbatim math).

    ``epoch`` is fractional — the official schedule steps per iteration with
    ``step / len(data_loader) + epoch``. Honors the official ``lr_scale`` param-group
    extension consumed by layer-wise lr decay (not ported, kept so the math stays verbatim).
    """
    if epoch < warmup_epochs:
        lr_now = lr * epoch / warmup_epochs
    else:
        lr_now = min_lr + (lr - min_lr) * 0.5 * (
            1.0 + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs))
        )
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr_now * param_group["lr_scale"]
        else:
            param_group["lr"] = lr_now
    return lr_now


# ------------------------------ experiment (docs/mae.md) ------------------------------ #
# The Exp wiring below maps facebookresearch/mae main_pretrain.py (training) and the eval
# slice of main_finetune.py onto the tinyexp Cfg-component layout; recipe defaults are the
# official argparse / PRETRAIN.md defaults.


@dataclass(repr=False)
class MaeExp(TinyExp, RayCfgMixin, RedisCfgMixin, CheckpointCfgMixin, WandbCfgMixin, LoggerCfgMixin):
    mode: str = "train"  # train (MAE pretraining) / eval (finetuned-classifier cross-check)
    launcher: str = "ray"
    epochs: int = 800  # PRETRAIN.md recipe (argparse default is 400)
    max_train_steps: int = -1  # smoke runs only; official has no step cap
    accum_iter: int = 1  # official --accum_iter (effective batch = batch * accum * world)
    seed: int = 0  # official --seed; _run adds the rank (official: seed + get_rank())

    @dataclass
    class RayCfg(RayCfgMixin.RayCfg):
        ray_num_worker: int = 1  # official sanity checks are single GPU; scale per cluster
        ray_num_gpus_per_worker: float = 1.0
        ray_num_cpus_per_worker: int = 12  # main process plus 10 dataloader workers

    ray_cfg: RayCfg = field(default_factory=RayCfg)

    @dataclass
    class AcceleratorCfg:
        accelerator: str = "ddp"

        def build_accelerator(self) -> AcceleratorProtocol:
            from tinyexp.tiny_engine.accelerator import CPUAccelerator, DDPAccelerator

            if self.accelerator == "cpu":
                return CPUAccelerator()
            if self.accelerator == "ddp":
                return DDPAccelerator()
            raise UnknownAcceleratorTypeError(self.accelerator)

    accelerator_cfg: AcceleratorCfg = field(default_factory=AcceleratorCfg)

    @dataclass
    class ModuleCfg:
        # shared factory kwargs (official --input_size)
        img_size: int = 224
        # pretraining model kwargs (official --mask_ratio / --norm_pix_loss; PRETRAIN.md recipe)
        mask_ratio: float = 0.75
        norm_pix_loss: bool = True  # official argparse default False; PRETRAIN.md recipe sets it
        # classifier (eval) kwargs (official --nb_classes / --global_pool / --drop_path)
        num_classes: int = 1000
        global_pool: bool = True  # must match the checkpoint
        drop_path_rate: float = 0.1  # eval-irrelevant, official default
        # official MAE checkpoint (URL or local path) to warm-start training from (official
        # --finetune branch) or to evaluate (mode=eval); empty trains/evals from scratch
        pretrained_from: str = ""

        def build_module(self) -> MaskedAutoencoderViT:
            """Pretraining model (official ``--model mae_vit_base_patch16``)."""
            return mae_vit_base_patch16(img_size=self.img_size, norm_pix_loss=self.norm_pix_loss)

        def build_classifier(self) -> VisionTransformer:
            """Eval classifier (official ``vit_base_patch16`` of main_finetune.py)."""
            return vit_base_patch16(
                img_size=self.img_size,
                num_classes=self.num_classes,
                global_pool=self.global_pool,
                drop_path_rate=self.drop_path_rate,
            )

    module_cfg: ModuleCfg = field(default_factory=ModuleCfg)

    @dataclass
    class DataloaderCfg:
        data_root: str = os.environ.get("IMAGENET_HOME", "./data/imagenet/")
        input_size: int = 224
        train_batch_size_per_device: int = 64  # official --batch_size (per GPU)
        train_num_workers: int = 10  # official --num_workers
        val_batch_size_per_device: int = 64  # official --batch_size default (FINETUNE.md uses 16)
        val_num_workers: int = 10
        dist_eval: bool = False  # official default: every rank iterates the full val set
        pin_mem: bool = True  # official --pin_mem default
        # synthetic tensors instead of ImageNet: CPU smoke tests, no disk data
        fake_data: bool = False
        fake_data_len: int = 64

        def _build_transform(self, is_train: bool) -> transforms.Compose:
            # train branch: main_pretrain.py inline transform_train ("simple augmentation");
            # eval branch: util/datasets.py build_transform(is_train=False)
            if is_train:
                return transforms.Compose(
                    [
                        # official passes interpolation=3 (bicubic)
                        transforms.RandomResizedCrop(
                            self.input_size, scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BICUBIC
                        ),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
                    ]
                )
            t = []
            crop_pct = 224 / 256 if self.input_size <= 224 else 1.0
            size = int(self.input_size / crop_pct)
            t.append(
                # official passes interpolation=PIL.Image.BICUBIC (docs/mae.md D5)
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
            # official build_dataset IMNET branch / main_pretrain.py ImageFolder(train)
            root = os.path.join(self.data_root, "train" if is_train else "val")
            transform = self._build_transform(is_train)
            if is_train and redis_cfg is not None and redis_cfg.redis_cache_enabled:
                # docs/mae.md D11: Redis byte cache for slow shared filesystems; samples are
                # bit-identical to datasets.ImageFolder (the cache stores raw file bytes).
                # The val split is read once per run and stays uncached.
                return RedisCachedImageFolder(
                    redis_host=redis_cfg.redis_cluster_host,
                    redis_ports=list(redis_cfg.redis_cluster_ports),
                    root=root,
                    transform=transform,
                    redis_world_size=int(redis_cfg.redis_rendezvous_world_size),
                )
            return datasets.ImageFolder(root, transform=transform)

        def build_train_dataloader(
            self, accelerator: AcceleratorProtocol, redis_cfg=None
        ) -> torch.utils.data.DataLoader:
            dataset = self._build_dataset(is_train=True, redis_cfg=redis_cfg)
            if accelerator.world_size > 1:
                sampler: torch.utils.data.Sampler = torch.utils.data.DistributedSampler(
                    dataset, num_replicas=accelerator.world_size, rank=accelerator.rank, shuffle=True
                )
            else:
                # official else-branch (not distributed)
                sampler = torch.utils.data.RandomSampler(dataset)
            return torch.utils.data.DataLoader(
                dataset,
                sampler=sampler,
                batch_size=self.train_batch_size_per_device,
                num_workers=self.train_num_workers,
                pin_memory=self.pin_mem,
                drop_last=True,
            )

        def build_val_dataloader(self, accelerator: AcceleratorProtocol) -> torch.utils.data.DataLoader:
            dataset = self._build_dataset(is_train=False)
            if self.dist_eval and accelerator.world_size > 1:
                sampler: torch.utils.data.Sampler = torch.utils.data.DistributedSampler(
                    dataset, num_replicas=accelerator.world_size, rank=accelerator.rank, shuffle=True
                )  # official: shuffle=True to reduce monitor bias
            else:
                # official non-dist-eval: every rank iterates the full validation set
                sampler = torch.utils.data.SequentialSampler(dataset)
            return torch.utils.data.DataLoader(
                dataset,
                sampler=sampler,
                batch_size=self.val_batch_size_per_device,
                num_workers=self.val_num_workers,
                pin_memory=self.pin_mem,
                drop_last=False,
            )

    dataloader_cfg: DataloaderCfg = field(default_factory=DataloaderCfg)

    @dataclass
    class OptimizerCfg:
        # official PRETRAIN.md recipe / main_pretrain.py defaults
        blr: float = 1.5e-4  # base lr; actual lr = blr * eff_batch_size / 256 (linear rule)
        weight_decay: float = 0.05

        def scaled_lr(self, dataloader, accelerator: AcceleratorProtocol, accum_iter: int = 1) -> float:
            eff_batch_size = dataloader.batch_size * accum_iter * accelerator.world_size
            return self.blr * eff_batch_size / 256.0

        def build_optimizer(
            self, module: nn.Module, dataloader, accelerator: AcceleratorProtocol, accum_iter: int = 1
        ) -> torch.optim.Optimizer:
            # official: timm optim_factory.add_weight_decay — no wd for bias/norm (ndim<=1),
            # frozen pos embeds skipped; timm>=1.0 spelling param_groups_weight_decay (D9)
            param_groups = param_groups_weight_decay(module, self.weight_decay)
            return torch.optim.AdamW(
                param_groups,
                lr=self.scaled_lr(dataloader, accelerator, accum_iter),
                betas=(0.9, 0.95),
            )

    optimizer_cfg: OptimizerCfg = field(default_factory=OptimizerCfg)

    @dataclass
    class LrSchedulerCfg:
        # official main_pretrain.py argparse defaults; consumed per-iteration by
        # adjust_learning_rate (no scheduler object to build)
        min_lr: float = 0.0
        warmup_epochs: int = 40

    lr_scheduler_cfg: LrSchedulerCfg = field(default_factory=LrSchedulerCfg)

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
            save_dir=run_dir, distributed_rank=accelerator.rank, filename="mae_train.log"
        )
        cfg_dict = self.print_cfg(logger)

        if accelerator.device.type == "cuda":
            _set_cudnn_benchmark()  # official main.py
        # official seeds seed + rank (main_finetune.py / main_pretrain.py)
        seed = self.seed + accelerator.rank
        torch.manual_seed(seed)
        np.random.seed(seed)

        if self.mode == "train":
            self._train(accelerator=accelerator, logger=logger, cfg_dict=cfg_dict, run_dir=run_dir)
        elif self.mode == "eval":
            self._run_eval(accelerator=accelerator, logger=logger)
        else:
            raise NotImplementedError(f"Mode {self.mode} is not implemented")

    # ------------------------------ pretraining (main_pretrain.py) ------------------------------ #

    def _train(self, accelerator, logger, cfg_dict, run_dir: str) -> None:
        data_loader_train = self.dataloader_cfg.build_train_dataloader(accelerator, self.redis_cfg)

        # define the model
        model = self.module_cfg.build_module()
        if self.module_cfg.pretrained_from:
            # official --finetune branch: warm-start the autoencoder weights only
            load_pretrained_checkpoint(model, self.module_cfg.pretrained_from)
        model.to(accelerator.device)

        lr = self.optimizer_cfg.scaled_lr(data_loader_train, accelerator, self.accum_iter)
        optimizer = self.optimizer_cfg.build_optimizer(model, data_loader_train, accelerator, self.accum_iter)
        loss_scaler = NativeScaler(device=accelerator.device.type)  # GradScaler auto-disables on cpu

        eff_batch_size = self.dataloader_cfg.train_batch_size_per_device * self.accum_iter * accelerator.world_size
        logger.info(f"base lr: {lr * 256 / eff_batch_size:.2e}")
        logger.info(f"actual lr: {lr:.2e}")
        logger.info(f"accumulate grad iterations: {self.accum_iter}")
        logger.info(f"effective batch size: {eff_batch_size}")

        start_epoch = 0
        global_step = 0
        if self.resume_from:
            # official load_model: restore model/optimizer/scaler and continue at epoch+1
            checkpoint = self.checkpoint_cfg.load_checkpoint(
                self.resume_from,
                model=model,
                optimizer=optimizer,
                scaler=loss_scaler,
                map_location=accelerator.device,
            )
            start_epoch = int(checkpoint.get("epoch", -1)) + 1
            global_step = int(checkpoint.get("global_step", 0) or 0)

        model, optimizer = accelerator.prepare(model, optimizer)

        if self.wandb_cfg.enable_wandb and accelerator.is_main_process:
            self.wandb_cfg.build_wandb(accelerator=accelerator, project="TinyExp", config=cfg_dict, name="mae_exp")

        logger.info(f"Start training for {self.epochs} epochs")
        for epoch in range(start_epoch, self.epochs):
            train_sampler = getattr(data_loader_train, "sampler", None)
            if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)

            train_stats, global_step = self._train_one_epoch(
                accelerator,
                logger,
                model,
                data_loader_train,
                optimizer,
                accelerator.device,
                epoch,
                loss_scaler,
                lr,
                global_step,
            )

            # official order: checkpoint, then per-epoch log stats (rank 0 only)
            if accelerator.is_main_process:
                self.checkpoint_cfg.save_checkpoint(
                    run_dir=run_dir,
                    name=self.checkpoint_cfg.last_ckpt_name,
                    model=accelerator.unwrap_model(model),
                    optimizer=optimizer,
                    scaler=loss_scaler,
                    epoch=epoch,
                    global_step=global_step,
                    exp_name=self.exp_name,
                    exp_class=self.exp_class,
                )
                log_stats = {**{f"train_{k}": v for k, v in train_stats.items()}, "epoch": epoch}
                with open(os.path.join(run_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")
                if self.wandb_cfg.enable_wandb:
                    import wandb

                    wandb.log(log_stats)

            if 0 < self.max_train_steps <= global_step:
                break

    def _train_one_epoch(
        self,
        accelerator,
        logger,
        model,
        data_loader,
        optimizer,
        device,
        epoch: int,
        loss_scaler,
        lr: float,
        global_step: int = 0,
    ) -> tuple[dict, int]:
        """Verbatim port of mae engine_pretrain.train_one_epoch (tensorboard branch cut).

        tinyexp smoke extension (resnet_exp precedent): the running ``global_step`` is
        counted per iteration and the loop stops early once ``max_train_steps`` is reached
        (official has no step cap; every rank computes the same count, so the early stop
        stays DDP-safe).
        """
        model.train(True)
        metric_logger = MetricLogger(log_fn=logger.info)
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        header = f"Epoch: [{epoch}]"
        print_freq = 20

        accum_iter = self.accum_iter

        optimizer.zero_grad()

        adjust = partial(
            adjust_learning_rate,
            optimizer,
            lr=lr,
            min_lr=self.lr_scheduler_cfg.min_lr,
            warmup_epochs=self.lr_scheduler_cfg.warmup_epochs,
            epochs=self.epochs,
        )

        for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            # we use a per iteration (instead of per epoch) lr scheduler
            if data_iter_step % accum_iter == 0:
                adjust(data_iter_step / len(data_loader) + epoch)

            samples = samples.to(device, non_blocking=True)

            # compute output; official: torch.cuda.amp.autocast() (docs/mae.md D4)
            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                loss, _, _ = model(samples, mask_ratio=self.module_cfg.mask_ratio)
                loss_value = loss.item()

            if not math.isfinite(loss_value):
                logger.error(f"Loss is {loss_value}, stopping training")
                sys.exit(1)

            loss /= accum_iter
            # official timm 0.3.2 kwarg was update_grad; timm>=1.0 renamed it need_update (D9)
            loss_scaler(
                loss,
                optimizer,
                parameters=model.parameters(),
                need_update=(data_iter_step + 1) % accum_iter == 0,
            )
            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()

            if device.type == "cuda":
                torch.cuda.synchronize()  # official call (guarded for CPU smoke, docs/mae.md D4)

            metric_logger.update(loss=loss_value)

            lr_now = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr_now)

            global_step += 1
            if 0 < self.max_train_steps <= global_step:
                break

        # gather the stats from all processes
        metric_logger.synchronize_between_processes(accelerator)
        logger.info(f"Averaged stats: {metric_logger}")
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, global_step

    # ------------------------------ evaluation (main_finetune.py eval slice) ------------------------------ #

    def _prepare_for_eval(self, accelerator, logger):
        """Eval slice of official main_finetune.py, in official order."""
        data_loader_val = self.dataloader_cfg.build_val_dataloader(accelerator)

        model = self.module_cfg.build_classifier()
        if self.resume_from:
            # tinyexp checkpoint (future finetune port); takes precedence over pretrained_from
            self.checkpoint_cfg.load_checkpoint(self.resume_from, model=model, map_location="cpu")
        elif self.module_cfg.pretrained_from:
            load_pretrained_checkpoint(model, self.module_cfg.pretrained_from)
        elif not self.dataloader_cfg.fake_data:
            raise ValueError(
                "mode=eval requires resume_from (tinyexp ckpt) or module_cfg.pretrained_from "
                f"(e.g. {MAE_FINETUNED_VIT_BASE_CKPT_URL})"
            )
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"number of params (M): {n_parameters / 1.0e6:.2f}")  # official print format
        model.to(accelerator.device)
        return data_loader_val, model

    def _run_eval(self, accelerator, logger) -> float:
        data_loader_val, model = self._prepare_for_eval(accelerator, logger)
        model = accelerator.prepare(model)
        test_stats = self._evaluate(accelerator, logger, model, accelerator.device, data_loader_val)
        logger.info(
            f"Accuracy of the network on the {len(data_loader_val.dataset)} test images: {test_stats['acc1']:.1f}%"
        )
        self._run_result = f"eval acc@1={test_stats['acc1']:.2f}% acc@5={test_stats['acc5']:.2f}%"
        return test_stats["acc1"]

    @torch.no_grad()
    def _evaluate(self, accelerator, logger, model, device, val_dataloader) -> dict:
        """Verbatim port of mae engine_finetune.evaluate."""
        criterion = nn.CrossEntropyLoss()

        metric_logger = MetricLogger(log_fn=logger.info)
        header = "Test:"

        # switch to evaluation mode
        model.eval()

        for images, target in metric_logger.log_every(val_dataloader, 10, header):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output; official: torch.cuda.amp.autocast() (docs/mae.md D4)
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
            f"* Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f} "
            f"loss {metric_logger.loss.global_avg:.3f}"
        )

        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    def get_ray_run_result(self) -> str | None:
        return getattr(self, "_run_result", None)


if __name__ == "__main__":
    store_and_run_exp(MaeExp)
