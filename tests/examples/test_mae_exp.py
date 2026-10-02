"""Focused correctness tests for the MAE ViT-B example."""

from __future__ import annotations

import json
import math
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("timm", reason="tinyexp vit extra (timm) is required for the mae_exp example")

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from tinyexp.examples.mae_exp import (
    MaeExp,
    MaskedAutoencoderViT,
    VisionTransformer,
    adjust_learning_rate,
    interpolate_pos_embed,
    load_pretrained_checkpoint,
    mae_vit_base_patch16,
    vit_base_patch16,
)


def _tiny_model(seed: int | None = None, **overrides: Any) -> VisionTransformer:
    """A small ViT classifier shaped like the MAE factory but cheap enough for CPU tests."""
    if seed is not None:
        torch.manual_seed(seed)
    kwargs: dict = {
        "img_size": 64,
        "patch_size": 16,
        "embed_dim": 192,
        "depth": 2,
        "num_heads": 3,
        "mlp_ratio": 4,
        "qkv_bias": True,
        "norm_layer": partial(nn.LayerNorm, eps=1e-6),
        "num_classes": 10,
    }
    kwargs.update(overrides)
    return VisionTransformer(**kwargs).eval()


def _tiny_mae(seed: int | None = None, **overrides: Any) -> MaskedAutoencoderViT:
    """A small masked autoencoder shaped like mae_vit_base_patch16_dec512d8b."""
    if seed is not None:
        torch.manual_seed(seed)
    kwargs: dict = {
        "img_size": 64,
        "patch_size": 16,
        "embed_dim": 192,
        "depth": 2,
        "num_heads": 3,
        "decoder_embed_dim": 128,
        "decoder_depth": 2,
        "decoder_num_heads": 4,
        "mlp_ratio": 4,
        "norm_layer": partial(nn.LayerNorm, eps=1e-6),
    }
    kwargs.update(overrides)
    return MaskedAutoencoderViT(**kwargs).eval()


def _timm_native_forward(model: VisionTransformer, x: torch.Tensor) -> torch.Tensor:
    """Reference composition of timm's own forward_features + forward_head stages."""
    feats = model.patch_embed(x)
    feats = model._pos_embed(feats)
    feats = model.patch_drop(feats)
    feats = model.norm_pre(feats)
    feats = model.blocks(feats)
    feats = model.norm(feats)
    out = model.fc_norm(feats[:, model.num_prefix_tokens :].mean(dim=1)) if model.global_pool == "avg" else feats[:, 0]
    return model.head(out)


# ---------------------- classifier (eval model) ---------------------- #


def test_vit_base_param_count_and_key_layout() -> None:
    model = vit_base_patch16()
    assert sum(p.numel() for p in model.parameters()) == 86_567_656  # official "86.57M"
    assert model.global_pool == "avg"
    assert isinstance(model.fc_norm, nn.LayerNorm)
    assert isinstance(model.norm, nn.Identity)  # the official ``del self.norm`` layout

    state_dict = model.state_dict()
    assert len(state_dict) == 152
    assert not any(key.startswith("norm.") for key in state_dict)
    assert "fc_norm.weight" in state_dict and "head.weight" in state_dict
    assert state_dict["pos_embed"].shape == (1, 197, 768)


@pytest.mark.parametrize("global_pool", [True, False])
def test_mae_forward_matches_timm_native_pooling(global_pool: bool) -> None:
    model = _tiny_model(seed=0, global_pool=global_pool)
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        torch.testing.assert_close(model(x), _timm_native_forward(model, x), atol=0, rtol=0)


def test_load_pretrained_checkpoint_round_trip(tmp_path: Path) -> None:
    model_a = _tiny_model(seed=0, num_classes=100)
    checkpoint_path = tmp_path / "tiny_ckpt.pth"
    torch.save({"model": model_a.state_dict()}, checkpoint_path)

    model_b = _tiny_model(seed=1, num_classes=100)
    load_pretrained_checkpoint(model_b, str(checkpoint_path))
    assert set(model_b.state_dict()) == set(model_a.state_dict())
    for key, value in model_a.state_dict().items():
        torch.testing.assert_close(model_b.state_dict()[key], value, rtol=0, atol=0)


def test_load_pretrained_checkpoint_drops_mismatched_head(tmp_path: Path) -> None:
    model_a = _tiny_model(seed=0, num_classes=100)
    checkpoint_path = tmp_path / "tiny_ckpt_100cls.pth"
    torch.save({"model": model_a.state_dict()}, checkpoint_path)

    model_b = _tiny_model(seed=1, num_classes=10)
    load_pretrained_checkpoint(model_b, str(checkpoint_path))
    checkpoint_head_weight = model_a.head.weight
    assert model_b.head.weight.shape == (10, 192)
    assert model_b.head.weight.shape != checkpoint_head_weight.shape  # the dropped head cannot be restored
    for key, value in model_b.state_dict().items():
        if key.startswith("head."):
            continue
        torch.testing.assert_close(value, model_a.state_dict()[key], rtol=0, atol=0)


def test_load_pretrained_checkpoint_mae_round_trip(tmp_path: Path) -> None:
    """The train-mode warm-start path (official --finetune branch) restores autoencoder weights."""
    model_a = _tiny_mae(seed=0)
    checkpoint_path = tmp_path / "tiny_mae_ckpt.pth"
    torch.save({"model": model_a.state_dict()}, checkpoint_path)

    model_b = _tiny_mae(seed=1)
    load_pretrained_checkpoint(model_b, str(checkpoint_path))
    for key, value in model_a.state_dict().items():
        torch.testing.assert_close(model_b.state_dict()[key], value, rtol=0, atol=0)


def test_interpolate_pos_embed_downsizes_grid() -> None:
    model = _tiny_model(seed=0, img_size=64)  # 4x4 patch grid
    checkpoint_model = {"pos_embed": torch.randn(1, 257, 192)}  # 16x16 grid + cls token
    interpolate_pos_embed(model, checkpoint_model)
    interpolated = checkpoint_model["pos_embed"]
    assert interpolated.shape == (1, 17, 192)
    # class token row is kept unchanged
    torch.testing.assert_close(interpolated[0, 0], checkpoint_model["pos_embed"][0, 0], rtol=0, atol=0)


def test_interpolate_pos_embed_is_noop_for_matching_grid() -> None:
    model = _tiny_model(seed=0, img_size=64)  # 4x4 grid
    pos_embed = torch.randn(1, 17, 192)  # matching 4x4 grid + cls token
    checkpoint_model = {"pos_embed": pos_embed}
    interpolate_pos_embed(model, checkpoint_model)
    assert checkpoint_model["pos_embed"] is pos_embed


def test_eval_transform_matches_official() -> None:
    cfg = MaeExp.DataloaderCfg(input_size=224)
    transform = cfg._build_transform(is_train=False)
    resize, center_crop, to_tensor, normalize = transform.transforms
    assert type(resize).__name__ == "Resize" and resize.size == 256  # int(224 / (224 / 256))
    assert resize.interpolation == transforms.InterpolationMode.BICUBIC
    assert type(center_crop).__name__ == "CenterCrop" and center_crop.size == (224, 224)
    assert type(to_tensor).__name__ == "ToTensor"
    assert type(normalize).__name__ == "Normalize"
    image = Image.new("RGB", (256, 336))
    assert transform(image).shape == (3, 224, 224)

    big = MaeExp.DataloaderCfg(input_size=384)._build_transform(is_train=False).transforms[0]
    assert big.size == 384  # crop_pct 1.0 branch


def test_train_transform_matches_official() -> None:
    transform = MaeExp.DataloaderCfg(input_size=224)._build_transform(is_train=True)
    crop, flip, to_tensor, normalize = transform.transforms
    assert type(crop).__name__ == "RandomResizedCrop" and crop.scale == (0.2, 1.0)  # official "simple augmentation"
    assert crop.interpolation == transforms.InterpolationMode.BICUBIC  # official interpolation=3
    assert type(flip).__name__ == "RandomHorizontalFlip"
    assert type(to_tensor).__name__ == "ToTensor" and type(normalize).__name__ == "Normalize"


# ---------------------- masked autoencoder (pretraining model) ---------------------- #


def test_mae_vit_base_param_count() -> None:
    model = mae_vit_base_patch16()
    assert sum(p.numel() for p in model.parameters()) == 111_907_840
    # fixed sin-cos position embeddings are buffers-by-choice (frozen Parameters)
    assert not model.pos_embed.requires_grad and not model.decoder_pos_embed.requires_grad


def test_mae_forward_shapes_and_masking() -> None:
    model = _tiny_mae(seed=0, norm_pix_loss=True)
    x = torch.randn(3, 3, 64, 64)
    loss, pred, mask = model(x, mask_ratio=0.75)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert pred.shape == (3, 16, 768)  # L patches of p*p*3 = 16*16*3
    assert mask.shape == (3, 16)
    assert set(mask.unique().tolist()) <= {0.0, 1.0}
    assert mask.sum(dim=1).tolist() == [12.0, 12.0, 12.0]  # exactly int(16 * 0.75) removed per sample
    # patchify/unpatchify are inverse operations
    torch.testing.assert_close(model.unpatchify(model.patchify(x)), x, rtol=0, atol=0)


def test_mae_norm_pix_loss_value() -> None:
    model = _tiny_mae(seed=0, norm_pix_loss=True)
    x = torch.randn(2, 3, 64, 64)
    loss, pred, mask = model(x, mask_ratio=0.5)

    target = model.patchify(x)
    mean = target.mean(dim=-1, keepdim=True)
    var = target.var(dim=-1, keepdim=True)
    target = (target - mean) / (var + 1.0e-6) ** 0.5
    expected = (((pred - target) ** 2).mean(dim=-1) * mask).sum() / mask.sum()
    torch.testing.assert_close(loss, expected, rtol=0, atol=1e-6)


def test_adjust_learning_rate_warmup_and_cosine() -> None:
    p1, p2 = nn.Parameter(torch.zeros(1)), nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([{"params": [p1]}, {"params": [p2], "lr_scale": 0.5}], lr=1.0)
    common = {"lr": 1.0, "min_lr": 0.0, "warmup_epochs": 40, "epochs": 100}

    assert adjust_learning_rate(optimizer, 10, **common) == pytest.approx(0.25)  # linear warmup
    assert [g["lr"] for g in optimizer.param_groups] == pytest.approx([0.25, 0.125])  # lr_scale honored
    assert adjust_learning_rate(optimizer, 40, **common) == pytest.approx(1.0)  # end of warmup
    assert adjust_learning_rate(optimizer, 70, **common) == pytest.approx(0.5)  # cosine midpoint (cos(pi/2)=0)
    assert adjust_learning_rate(optimizer, 100, **common) == pytest.approx(0.0)  # end of cosine


def test_optimizer_param_groups_and_scaled_lr() -> None:
    model = _tiny_mae(seed=0)
    cfg = MaeExp.OptimizerCfg()
    dataloader = SimpleNamespace(batch_size=8)
    accelerator = SimpleNamespace(world_size=2)
    optimizer = cfg.build_optimizer(model, dataloader, accelerator, accum_iter=4)

    # official linear rule: blr * (batch * accum * world) / 256
    expected_lr = 1.5e-4 * 8 * 4 * 2 / 256
    assert optimizer.param_groups[0]["lr"] == pytest.approx(expected_lr)
    assert optimizer.param_groups[0]["betas"] == (0.9, 0.95)

    names = dict(model.named_parameters())
    decay_group = next(g for g in optimizer.param_groups if g["weight_decay"] > 0)
    no_decay_group = next(g for g in optimizer.param_groups if g["weight_decay"] == 0)
    assert decay_group["weight_decay"] == pytest.approx(0.05)
    assert no_decay_group["weight_decay"] == 0.0
    assert any(p is names["patch_embed.proj.weight"] for p in decay_group["params"])
    assert any(p is names["cls_token"] for p in decay_group["params"])
    assert any(p is names["blocks.0.norm1.weight"] for p in no_decay_group["params"])
    # frozen sin-cos pos embeds are excluded from optimization entirely (official behavior)
    grouped = {id(p) for g in optimizer.param_groups for p in g["params"]}
    assert id(names["pos_embed"]) not in grouped and id(names["decoder_pos_embed"]) not in grouped


# ---------------------- experiment run path ---------------------- #


def _smoke_exp(tmp_path: Path, exp_name: str = "mae_smoke") -> MaeExp:
    exp = MaeExp(output_root=str(tmp_path), exp_name=exp_name)
    exp.accum_iter = 1  # the 1-step smoke must exercise the optimizer-update path
    exp.module_cfg.img_size = 32
    exp.dataloader_cfg.fake_data = True
    exp.dataloader_cfg.input_size = 32
    exp.dataloader_cfg.fake_data_len = 8
    exp.dataloader_cfg.train_batch_size_per_device = 8  # drop_last=True: batch must not exceed fake_data_len
    exp.dataloader_cfg.val_num_workers = 0
    return exp


def _patch_run_dependencies(monkeypatch: pytest.MonkeyPatch, exp: MaeExp) -> None:
    dummy_accelerator = SimpleNamespace(
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
        is_main_process=True,
        prepare=lambda model, optimizer=None: model if optimizer is None else (model, optimizer),
        unwrap_model=lambda model: model,
        destroy=lambda: None,
    )
    dummy_logger = SimpleNamespace(info=lambda *args, **kwargs: None, error=lambda *args, **kwargs: None)
    monkeypatch.setattr(exp.accelerator_cfg, "build_accelerator", lambda: dummy_accelerator)
    monkeypatch.setattr(exp.logger_cfg, "build_logger", lambda **kwargs: dummy_logger)


def test_train_smoke_fake_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exp = _smoke_exp(tmp_path)
    exp.epochs = 2
    exp.max_train_steps = 2  # 1 step per epoch on 8 fake samples at batch 8
    exp.eval_every_n_epochs = 1  # exercise the periodic val reconstruction monitor
    _patch_run_dependencies(monkeypatch, exp)

    exp.run()

    run_dir = tmp_path / "mae_smoke"
    checkpoint = torch.load(run_dir / "last.ckpt", map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == 1  # two epochs of one step each ran
    assert checkpoint["global_step"] == 2
    assert "model_state_dict" in checkpoint and "cls_token" in checkpoint["model_state_dict"]
    assert "optimizer_state_dict" in checkpoint and "scaler_state_dict" in checkpoint
    stats = [json.loads(line) for line in (run_dir / "log.txt").read_text().splitlines()]
    assert [entry["epoch"] for entry in stats] == [0, 1]
    assert all(math.isfinite(entry["train_loss"]) for entry in stats)
    assert all(math.isfinite(entry["val_loss"]) for entry in stats)  # periodic monitor fired


def test_train_resume_continues_epochs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exp = _smoke_exp(tmp_path)
    exp.epochs = 2
    exp.max_train_steps = 2
    _patch_run_dependencies(monkeypatch, exp)
    exp.run()

    resumed = _smoke_exp(tmp_path)
    resumed.epochs = 3
    resumed.resume_from = str(tmp_path / "mae_smoke" / "last.ckpt")
    _patch_run_dependencies(monkeypatch, resumed)
    resumed.run()

    checkpoint = torch.load(tmp_path / "mae_smoke" / "last.ckpt", map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == 2
    assert checkpoint["global_step"] == 3
    stats = [json.loads(line) for line in (tmp_path / "mae_smoke" / "log.txt").read_text().splitlines()]
    assert [entry["epoch"] for entry in stats] == [0, 1, 2]  # resumed run appended epoch 2 only


def test_eval_smoke_fake_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exp = _smoke_exp(tmp_path, exp_name="mae_eval_smoke")
    exp.mode = "eval"
    exp.module_cfg.num_classes = 10
    _patch_run_dependencies(monkeypatch, exp)

    exp.run()

    result = exp.get_ray_run_result()
    assert result is not None and result.startswith("eval acc@1=")


def test_eval_requires_checkpoint_unless_fake_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exp = MaeExp(output_root=str(tmp_path), exp_name="mae_no_ckpt")
    exp.mode = "eval"
    _patch_run_dependencies(monkeypatch, exp)
    # the real-data path would scan ImageNet; the checkpoint guard fires first in logic
    monkeypatch.setattr(exp.dataloader_cfg, "build_val_dataloader", lambda accelerator: object())

    with pytest.raises(ValueError, match="pretrained_from"):
        exp.run()


def test_redis_cache_is_train_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from torchvision import datasets as tv_datasets

    from tinyexp.examples.vit_tp_exp import RedisCachedImageFolder

    for split in ("train", "val"):
        for class_name in ("a", "b"):
            class_dir = tmp_path / split / class_name
            class_dir.mkdir(parents=True)
            Image.new("RGB", (8, 8)).save(class_dir / "x.jpg")

    experiment = MaeExp(output_root=str(tmp_path), exp_name="mae_redis")
    experiment.dataloader_cfg.data_root = str(tmp_path)
    assert experiment.redis_cfg.redis_cache_enabled  # resnet_exp-style default: on

    class FakeRedisManager:
        def __init__(self, *args, **kwargs) -> None:
            pass

    monkeypatch.setattr("tinyexp.utils.redis_utils.RedisClientManager", FakeRedisManager)
    assert isinstance(experiment.dataloader_cfg._build_dataset(True, experiment.redis_cfg), RedisCachedImageFolder)
    # the val split is read once per run and stays uncached
    assert type(experiment.dataloader_cfg._build_dataset(False, experiment.redis_cfg)) is tv_datasets.ImageFolder

    experiment.redis_cfg.redis_cache_enabled = False
    assert type(experiment.dataloader_cfg._build_dataset(True, experiment.redis_cfg)) is tv_datasets.ImageFolder

    experiment.dataloader_cfg.fake_data = True
    assert isinstance(
        experiment.dataloader_cfg._build_dataset(True, experiment.redis_cfg), torch.utils.data.TensorDataset
    )
