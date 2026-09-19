"""Focused correctness tests for the DeiT tensor-parallel example."""

from __future__ import annotations

import copy
import os
import socket

import pytest

pytest.importorskip("timm", reason="tinyexp vit extra (timm) is required for the vit_tp_exp example")

import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.tensor import DTensor

from tinyexp.examples.vit_tp_exp import (
    Attention,
    RedisCachedImageFolder,
    VisionTransformer,
    VitTpExp,
    build_tp_parallelize_plan,
    convert_fused_qkv_to_split,
    convert_split_qkv_to_fused,
    deit_small_patch16_224,
)
from tinyexp.tiny_engine.accelerator import TPAccelerator


class FusedAttention(nn.Module):
    """The fused timm 0.3.2 attention used as the split-QKV oracle."""

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

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, channels // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = (q @ k.transpose(-2, -1)) * self.scale
        attention = self.attn_drop(attention.softmax(dim=-1))
        output = (attention @ v).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(output))


def _fused_reference(model: VisionTransformer) -> VisionTransformer:
    reference = copy.deepcopy(model)
    for name, module in list(reference.named_modules()):
        if not isinstance(module, Attention):
            continue
        fused = FusedAttention(module.q.in_features, num_heads=module.num_heads, qkv_bias=module.q.bias is not None)
        with torch.no_grad():
            fused.qkv.weight.copy_(torch.cat((module.q.weight, module.k.weight, module.v.weight)))
            if module.q.bias is not None:
                fused.qkv.bias.copy_(torch.cat((module.q.bias, module.k.bias, module.v.bias)))
            fused.proj.weight.copy_(module.proj.weight)
            fused.proj.bias.copy_(module.proj.bias)
        parent = reference.get_submodule(".".join(name.split(".")[:-1]))
        setattr(parent, name.rsplit(".", 1)[-1], fused)
    return reference


def _tiny_vit() -> VisionTransformer:
    torch.manual_seed(0)
    return VisionTransformer(
        img_size=32, patch_size=16, embed_dim=64, depth=2, num_heads=4, num_classes=10, qkv_bias=True
    ).eval()


def test_split_qkv_matches_fused_attention() -> None:
    model = _tiny_vit()
    reference = _fused_reference(model)
    x = torch.randn(4, 3, 32, 32)

    model_loss = model(x).square().mean()
    reference_loss = reference(x).square().mean()
    torch.testing.assert_close(model_loss, reference_loss, atol=1e-6, rtol=0)
    model_loss.backward()
    reference_loss.backward()
    torch.testing.assert_close(
        model.patch_embed.proj.weight.grad, reference.patch_embed.proj.weight.grad, atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        model.blocks[1].mlp.fc2.weight.grad, reference.blocks[1].mlp.fc2.weight.grad, atol=1e-6, rtol=0
    )


def test_deit_model_matches_timm() -> None:
    import timm

    model = deit_small_patch16_224().eval()
    reference = timm.create_model("deit_small_patch16_224", pretrained=False).eval()
    assert sum(parameter.numel() for parameter in model.parameters()) == 22050664
    assert sum(parameter.numel() for parameter in model.parameters()) == sum(
        parameter.numel() for parameter in reference.parameters()
    )

    model.load_state_dict(convert_fused_qkv_to_split(reference.state_dict()), strict=True)
    reference.load_state_dict(convert_split_qkv_to_fused(model.state_dict()), strict=True)
    inputs = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        torch.testing.assert_close(model(inputs), reference(inputs), atol=1e-5, rtol=0)


def test_qkv_conversion_round_trip() -> None:
    state_dict = _tiny_vit().state_dict()
    round_trip = convert_fused_qkv_to_split(convert_split_qkv_to_fused(state_dict))
    assert set(round_trip) == set(state_dict)
    for key, value in state_dict.items():
        torch.testing.assert_close(round_trip[key], value, rtol=0, atol=0)


def test_redis_cache_is_train_only(tmp_path, monkeypatch) -> None:
    from PIL import Image
    from torchvision import datasets as tv_datasets

    for split in ("train", "val"):
        for class_name in ("a", "b"):
            class_dir = tmp_path / split / class_name
            class_dir.mkdir(parents=True)
            Image.new("RGB", (8, 8)).save(class_dir / "x.jpg")

    experiment = VitTpExp()
    experiment.dataloader_cfg.data_root = str(tmp_path)
    assert not experiment.redis_cfg.redis_cache_enabled
    assert type(experiment.dataloader_cfg._build_dataset(True, experiment.redis_cfg)) is tv_datasets.ImageFolder

    class FakeRedisManager:
        def __init__(self, *args, **kwargs) -> None:
            pass

    monkeypatch.setattr("tinyexp.utils.redis_utils.RedisClientManager", FakeRedisManager)
    experiment.redis_cfg.redis_cache_enabled = True
    assert isinstance(experiment.dataloader_cfg._build_dataset(True, experiment.redis_cfg), RedisCachedImageFolder)
    assert type(experiment.dataloader_cfg._build_dataset(False, experiment.redis_cfg)) is tv_datasets.ImageFolder

    experiment.dataloader_cfg.fake_data = True
    assert isinstance(
        experiment.dataloader_cfg._build_dataset(True, experiment.redis_cfg), torch.utils.data.TensorDataset
    )


def _tp_model() -> VisionTransformer:
    torch.manual_seed(42)
    return VisionTransformer(
        img_size=32, patch_size=16, embed_dim=64, depth=2, num_heads=4, num_classes=10, qkv_bias=True
    ).eval()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _tp_reference(state_dict: dict, inputs: torch.Tensor) -> dict:
    model = _tp_model()
    model.load_state_dict(state_dict)
    with torch.no_grad():
        logits = model(inputs)
    model(inputs).square().mean().backward()
    return {"logits": logits, "grads": {name: parameter.grad.detach() for name, parameter in model.named_parameters()}}


def _tp_worker(rank: int, world_size: int, port: int, state_dict: dict, inputs: torch.Tensor, output_path: str) -> None:
    os.environ.update(
        CUDA_VISIBLE_DEVICES="",
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
    )
    accelerator = TPAccelerator()
    try:
        model = _tp_model()
        model.load_state_dict(state_dict)
        model = accelerator.prepare_model(model, parallelize_plan=build_tp_parallelize_plan(model, world_size))
        logits = model(inputs).detach()
        model.train()
        accelerator.backward(model(inputs).square().mean())
        grads = {}
        for name, parameter in model.named_parameters():
            grad = parameter.grad
            if isinstance(grad, DTensor):
                grad = grad.full_tensor()
            grads[name] = grad.detach().cpu()
        if accelerator.is_main_process:
            torch.save({"logits": logits.cpu(), "grads": grads}, output_path)
        accelerator.wait_for_everyone()
    finally:
        accelerator.destroy()


def test_tp2_matches_single_process(tmp_path) -> None:
    state_dict = _tp_model().state_dict()
    inputs = torch.randn(4, 3, 32, 32)
    reference = _tp_reference(state_dict, inputs)
    output_path = tmp_path / "tp2_result.pt"
    mp.spawn(_tp_worker, args=(2, _free_port(), state_dict, inputs, str(output_path)), nprocs=2, join=True)
    result = torch.load(output_path)
    torch.testing.assert_close(result["logits"], reference["logits"], atol=1e-5, rtol=0)
    assert set(result["grads"]) == set(reference["grads"])
    for name, reference_grad in reference["grads"].items():
        torch.testing.assert_close(result["grads"][name], reference_grad, atol=1e-5, rtol=0)


def test_tp_plan_requires_divisible_heads() -> None:
    with pytest.raises(ValueError, match="divisible"):
        build_tp_parallelize_plan(_tp_model(), tp_size=3)
