"""L1 numerical-equivalence tests for the TP code path (docs/vit_tp.md L1 / §9 M1).

Two gloo workers on CPU run the tiny ViT with ``TPAccelerator`` (tp=2) and the result
must match the single-process unparallelized forward/backward within fp32 tolerance.
"""

from __future__ import annotations

import os
import socket

import pytest

pytest.importorskip("timm", reason="tinyexp vit extra (timm) is required for the vit_tp_exp example")

import torch
import torch.multiprocessing as mp
from torch.distributed.tensor import DTensor

from tinyexp.examples.vit_tp_exp import VisionTransformer, build_tp_parallelize_plan
from tinyexp.tiny_engine.accelerator import TPAccelerator

ATOL = 1e-5


def _tiny_model() -> VisionTransformer:
    torch.manual_seed(42)
    model = VisionTransformer(
        img_size=32, patch_size=16, embed_dim=64, depth=2, num_heads=4, num_classes=10, qkv_bias=True
    )
    return model.eval()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _reference(state_dict: dict, x: torch.Tensor) -> dict:
    model = _tiny_model()
    model.load_state_dict(state_dict)
    with torch.no_grad():
        logits = model(x).clone()
    loss = model(x).square().mean()
    loss.backward()
    grads = {name: param.grad.detach().clone() for name, param in model.named_parameters()}
    return {"logits": logits, "grads": grads}


def _tp_worker(rank: int, world_size: int, port: int, state_dict: dict, x: torch.Tensor, out_path: str) -> None:
    # Force the CPU/gloo path of TPAccelerator even on machines with GPUs.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)

    accelerator = TPAccelerator()
    try:
        model = _tiny_model()
        model.load_state_dict(state_dict)
        plan = build_tp_parallelize_plan(model, tp_size=world_size)
        model = accelerator.prepare_model(model, parallelize_plan=plan)
        model.eval()

        with torch.no_grad():
            logits = model(x).clone()

        # drop_rate/drop_path_rate are 0 in the tiny model, so train() stays deterministic.
        model.train()
        loss = model(x).square().mean()
        accelerator.backward(loss)
        grads = {}
        for name, param in model.named_parameters():
            grad = param.grad
            if isinstance(grad, DTensor):
                grad = grad.full_tensor()
            grads[name] = grad.detach().cpu()

        if accelerator.is_main_process:
            torch.save({"logits": logits.cpu(), "grads": grads}, out_path)
        accelerator.wait_for_everyone()
    finally:
        accelerator.destroy()


def test_tp2_forward_and_grads_match_single_process(tmp_path) -> None:
    state_dict = _tiny_model().state_dict()
    x = torch.randn(4, 3, 32, 32)
    reference = _reference(state_dict, x)

    out_path = tmp_path / "tp2_result.pt"
    mp.spawn(
        _tp_worker,
        args=(2, _free_port(), state_dict, x, str(out_path)),
        nprocs=2,
        join=True,
    )

    tp_result = torch.load(out_path)
    torch.testing.assert_close(tp_result["logits"], reference["logits"], atol=ATOL, rtol=0)
    assert set(tp_result["grads"]) == set(reference["grads"])
    for name, ref_grad in reference["grads"].items():
        torch.testing.assert_close(tp_result["grads"][name], ref_grad, atol=ATOL, rtol=0)


def test_build_tp_plan_rejects_indivisible_heads() -> None:
    model = _tiny_model()  # num_heads=4
    with pytest.raises(ValueError, match="divisible"):
        build_tp_parallelize_plan(model, tp_size=3)


def test_tp_accelator_world_size_1_is_degenerate(monkeypatch) -> None:
    for var in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(var, raising=False)
    accelerator = TPAccelerator()
    try:
        model = _tiny_model()
        x = torch.randn(2, 3, 32, 32)
        expected = model(x).detach().clone()
        prepared = accelerator.prepare_model(model)  # no plan required at world_size 1
        got = prepared(x.to(accelerator.device)).cpu()
        # fp32 rounding differs between the cpu reference and the accelerator device.
        torch.testing.assert_close(got, expected, atol=1e-4, rtol=0)
    finally:
        accelerator.destroy()
