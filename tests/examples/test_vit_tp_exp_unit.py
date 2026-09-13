"""Unit tests for the ported DeiT model (docs/vit_tp.md).

The ported split-qkv ``Attention`` is checked against a verbatim copy of the official
timm 0.3.2 fused ``Attention`` (the oracle below), and the full model is cross-checked
against ``timm.create_model("deit_small_patch16_224")`` plus the official checkpoint.
"""

from __future__ import annotations

import copy
from collections import OrderedDict

import pytest

pytest.importorskip("timm", reason="tinyexp vit extra (timm) is required for the vit_tp_exp example")

import torch
import torch.nn as nn

from tinyexp.examples.vit_tp_exp import (
    DEIT_SMALL_PATCH16_224_CKPT_URL,
    Attention,
    VisionTransformer,
    convert_fused_qkv_to_split,
    convert_split_qkv_to_fused,
    deit_small_patch16_224,
)

# Official DeiT-S size (paper reports 22.1M).
DEIT_S_NUM_PARAMS = 22050664


class FusedAttention(nn.Module):
    """Oracle: timm 0.3.2 ``timm/models/vision_transformer.py::Attention``, verbatim."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def _to_fused_reference(model: VisionTransformer) -> VisionTransformer:
    """Return a deep copy of ``model`` whose split-qkv attentions are fused oracles."""
    fused = copy.deepcopy(model)
    for name, module in list(fused.named_modules()):
        if not isinstance(module, Attention):
            continue
        new = FusedAttention(module.q.in_features, num_heads=module.num_heads, qkv_bias=module.q.bias is not None)
        with torch.no_grad():
            new.qkv.weight.copy_(torch.cat([module.q.weight, module.k.weight, module.v.weight], dim=0))
            if module.q.bias is not None:
                new.qkv.bias.copy_(torch.cat([module.q.bias, module.k.bias, module.v.bias], dim=0))
            new.proj.weight.copy_(module.proj.weight)
            new.proj.bias.copy_(module.proj.bias)
        parent = fused.get_submodule(".".join(name.split(".")[:-1]))
        setattr(parent, name.split(".")[-1], new)
    return fused


def _tiny_vit() -> VisionTransformer:
    torch.manual_seed(0)
    model = VisionTransformer(
        img_size=32, patch_size=16, embed_dim=64, depth=2, num_heads=4, num_classes=10, qkv_bias=True
    )
    return model.eval()


def test_deit_s_param_count_matches_official() -> None:
    model = deit_small_patch16_224()
    assert sum(p.numel() for p in model.parameters()) == DEIT_S_NUM_PARAMS


def test_split_qkv_forward_and_grad_match_fused_reference() -> None:
    model = _tiny_vit()
    reference = _to_fused_reference(model)
    x = torch.randn(4, 3, 32, 32)

    loss_ours = model(x).square().mean()
    loss_ref = reference(x).square().mean()
    torch.testing.assert_close(loss_ours, loss_ref, atol=1e-6, rtol=0)

    loss_ours.backward()
    loss_ref.backward()
    torch.testing.assert_close(
        model.patch_embed.proj.weight.grad, reference.patch_embed.proj.weight.grad, atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        model.blocks[1].mlp.fc2.weight.grad, reference.blocks[1].mlp.fc2.weight.grad, atol=1e-6, rtol=0
    )


def test_converter_round_trip_is_lossless() -> None:
    state_dict = _tiny_vit().state_dict()
    round_trip = convert_fused_qkv_to_split(convert_split_qkv_to_fused(state_dict))
    assert set(round_trip) == set(state_dict)
    for key, value in state_dict.items():
        assert torch.equal(value, round_trip[key])


def test_timm_reference_cross_check() -> None:
    timm = pytest.importorskip("timm")
    reference = timm.create_model("deit_small_patch16_224", pretrained=False).eval()
    ours = deit_small_patch16_224().eval()

    assert sum(p.numel() for p in ours.parameters()) == sum(p.numel() for p in reference.parameters())

    # Cross-load in both directions: key/shape layouts must agree exactly.
    ours.load_state_dict(convert_fused_qkv_to_split(reference.state_dict()), strict=True)
    reference.load_state_dict(convert_split_qkv_to_fused(ours.state_dict()), strict=True)

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        torch.testing.assert_close(ours(x), reference(x), atol=1e-5, rtol=0)


def test_official_checkpoint_cross_check() -> None:
    """Load the official DeiT-S checkpoint through the converter and compare forwards."""
    try:
        checkpoint = torch.hub.load_state_dict_from_url(
            url=DEIT_SMALL_PATCH16_224_CKPT_URL, map_location="cpu", check_hash=True
        )
    except Exception as error:  # offline environments skip this test
        pytest.skip(f"could not download official checkpoint: {error}")

    official_state: OrderedDict = checkpoint["model"]

    reference = deit_small_patch16_224().eval()
    fused_reference = _to_fused_reference(reference)
    # The fused oracle must strict-load the official checkpoint: proves the ported
    # structure matches the official one key-for-key.
    fused_reference.load_state_dict(official_state, strict=True)
    reference.load_state_dict(convert_fused_qkv_to_split(official_state), strict=True)

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        torch.testing.assert_close(reference(x), fused_reference(x), atol=1e-5, rtol=0)


def test_patch_embed_rejects_wrong_input_size() -> None:
    model = _tiny_vit()
    with pytest.raises(AssertionError):
        model(torch.randn(1, 3, 64, 64))


def test_train_dataset_redis_gating(tmp_path, monkeypatch) -> None:
    """docs/vit_tp.md D7: the Redis byte cache is opt-in, train-only, and never applies to fake data."""
    from PIL import Image
    from torchvision import datasets as tv_datasets

    from tinyexp.examples.vit_tp_exp import RedisCachedImageFolder, VitTpExp

    for split in ("train", "val"):
        for cls in ("a", "b"):
            cls_dir = tmp_path / split / cls
            cls_dir.mkdir(parents=True)
            Image.new("RGB", (8, 8)).save(cls_dir / "x.jpg")

    exp = VitTpExp()
    exp.dataloader_cfg.data_root = str(tmp_path)

    # Default: redis disabled -> the verbatim official ImageFolder pipeline.
    assert not exp.redis_cfg.redis_cache_enabled
    ds = exp.dataloader_cfg._build_dataset(is_train=True, redis_cfg=exp.redis_cfg)
    assert type(ds) is tv_datasets.ImageFolder

    class _FakeRedisManager:
        def __init__(self, *args, **kwargs) -> None:
            pass

    monkeypatch.setattr("tinyexp.utils.redis_utils.RedisClientManager", _FakeRedisManager)
    exp.redis_cfg.redis_cache_enabled = True

    ds = exp.dataloader_cfg._build_dataset(is_train=True, redis_cfg=exp.redis_cfg)
    assert type(ds) is RedisCachedImageFolder

    # The val set is never redis-cached (byte-identical semantics matter most there).
    ds = exp.dataloader_cfg._build_dataset(is_train=False, redis_cfg=exp.redis_cfg)
    assert type(ds) is tv_datasets.ImageFolder

    # Synthetic smoke data wins over the cache.
    exp.dataloader_cfg.fake_data = True
    ds = exp.dataloader_cfg._build_dataset(is_train=True, redis_cfg=exp.redis_cfg)
    assert isinstance(ds, torch.utils.data.TensorDataset)
