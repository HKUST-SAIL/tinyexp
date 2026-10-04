"""CPU correctness tests for the official MAE frozen-encoder linear probe."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("timm", reason="tinyexp vit extra (timm) is required for the mae_linprobe_exp example")

import torch
from PIL import Image
from test_mae_exp import _patch_run_dependencies, _tiny_mae, _tiny_model
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import nn
from torchvision import transforms

from tinyexp.examples.mae_exp import MaeExp
from tinyexp.examples.mae_linprobe_exp import (
    LARS,
    MaeLinprobeExp,
    NativeScaler,
    RandomResizedCrop,
    load_encoder_checkpoint,
)


@pytest.fixture(scope="module", autouse=True)
def _single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def tiny_classifier(monkeypatch: pytest.MonkeyPatch) -> None:
    # Patch the inherited class method: every fresh/resume/eval factory stays small.
    monkeypatch.setattr(
        MaeExp.ModuleCfg,
        "build_classifier",
        lambda cfg: _tiny_model(
            img_size=cfg.img_size,
            embed_dim=32,
            depth=1,
            num_heads=4,
            mlp_ratio=2,
            num_classes=cfg.num_classes,
            global_pool=cfg.global_pool,
            drop_path_rate=cfg.drop_path_rate,
        ),
    )


def _classifier(seed: int = 0, img_size: int = 32):
    return _tiny_model(
        seed=seed,
        img_size=img_size,
        embed_dim=32,
        depth=1,
        num_heads=4,
        mlp_ratio=2,
        num_classes=6,
        global_pool=False,
    )


def _mae(seed: int = 0, img_size: int = 32):
    model = _tiny_mae(
        seed=seed,
        img_size=img_size,
        embed_dim=32,
        depth=1,
        num_heads=4,
        mlp_ratio=2,
        decoder_embed_dim=16,
        decoder_depth=1,
        decoder_num_heads=4,
    )
    # Non-default encoder norm values make a silently skipped norm load observable.
    with torch.no_grad():
        model.norm.weight.fill_(1.25)
        model.norm.bias.fill_(0.25)
    return model


def _probe_exp(tmp_path: Path, exp_name: str = "probe") -> MaeLinprobeExp:
    exp = MaeLinprobeExp(
        output_root=str(tmp_path),
        exp_name=exp_name,
        exp_class="tinyexp.examples.mae_linprobe_exp.MaeLinprobeExp",
        epochs=1,
        accum_iter=2,
    )
    exp.module_cfg.img_size = 32
    exp.module_cfg.num_classes = 6  # inherited evaluation requests top-5
    exp.dataloader_cfg.input_size = 32
    exp.dataloader_cfg.fake_data = True
    exp.dataloader_cfg.fake_data_len = 16
    exp.dataloader_cfg.train_batch_size_per_device = 4
    exp.dataloader_cfg.val_batch_size_per_device = 4
    exp.dataloader_cfg.train_num_workers = 0
    exp.dataloader_cfg.val_num_workers = 0
    exp.dataloader_cfg.pin_mem = False
    return exp


def _capture_build(monkeypatch: pytest.MonkeyPatch, exp: MaeLinprobeExp) -> dict:
    initial = {}
    original = exp._build_probe_model

    def build(source=""):
        model = original(source)
        initial.update({key: value.clone() for key, value in model.state_dict().items()})
        return model

    monkeypatch.setattr(exp, "_build_probe_model", build)
    return initial


def test_config_factories_use_probe_defaults_and_independent_instances() -> None:
    first, second = MaeLinprobeExp(), MaeLinprobeExp()
    assert first.epochs == 90 and first.accum_iter == 4
    assert type(first.module_cfg) is MaeLinprobeExp.ModuleCfg
    assert type(first.dataloader_cfg) is MaeLinprobeExp.DataloaderCfg
    assert type(first.optimizer_cfg) is MaeLinprobeExp.OptimizerCfg
    assert type(first.lr_scheduler_cfg) is MaeLinprobeExp.LrSchedulerCfg
    assert type(first.checkpoint_cfg) is MaeLinprobeExp.CheckpointCfg
    assert not first.module_cfg.global_pool and first.module_cfg.drop_path_rate == 0
    assert first.dataloader_cfg.train_batch_size_per_device == 512
    assert first.optimizer_cfg.blr == 0.1 and first.optimizer_cfg.weight_decay == 0
    assert first.lr_scheduler_cfg.warmup_epochs == 10 and first.lr_scheduler_cfg.min_lr == 0
    assert first.optimizer_cfg.scaled_lr(SimpleNamespace(batch_size=512), SimpleNamespace(world_size=8), 4) == 6.4
    for name in ("module_cfg", "dataloader_cfg", "optimizer_cfg", "lr_scheduler_cfg", "checkpoint_cfg"):
        assert getattr(first, name) is not getattr(second, name)
    first.module_cfg.num_classes = 6
    first.checkpoint_cfg.best_ckpt_name = "custom.ckpt"
    assert second.module_cfg.num_classes == 1000 and second.checkpoint_cfg.best_ckpt_name == "best.ckpt"


def test_inherited_classifier_factory_forwards_probe_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = Mock(return_value=object())
    monkeypatch.setattr("tinyexp.examples.mae_exp.vit_base_patch16", factory)
    cfg = MaeLinprobeExp.ModuleCfg(img_size=32, num_classes=6)
    assert cfg.build_classifier() is factory.return_value
    factory.assert_called_once_with(img_size=32, num_classes=6, global_pool=False, drop_path_rate=0.0)


@pytest.mark.parametrize("payload_key", ["model", "model_state_dict"], ids=["official", "tinyexp"])
@pytest.mark.parametrize("full_mae", [True, False], ids=["full-mae", "encoder-only"])
def test_load_encoder_accepts_full_mae_and_encoder_only(
    tmp_path: Path,
    payload_key: str,
    full_mae: bool,
) -> None:
    source = _mae(seed=7)
    target = _classifier(seed=11)
    encoder_keys = set(target.state_dict()) - {"head.weight", "head.bias"}
    state = source.state_dict()
    if not full_mae:
        state = {key: value for key, value in state.items() if key in encoder_keys}
    # The old head must be ignored even if its shape matches the new classifier.
    state.update(
        {"head.weight": torch.full_like(target.head.weight, 99), "head.bias": torch.full_like(target.head.bias, 99)}
    )
    path = tmp_path / "encoder.ckpt"
    torch.save({payload_key: state, "epoch": 3}, path)
    head_before = {key: value.clone() for key, value in target.head.state_dict().items()}

    load_encoder_checkpoint(target, str(path))

    for key in encoder_keys:
        torch.testing.assert_close(target.state_dict()[key], source.state_dict()[key], rtol=0, atol=0)
    for key, value in head_before.items():
        torch.testing.assert_close(target.head.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize("scheme", ["http", "https"])
def test_load_official_encoder_url_without_network(monkeypatch: pytest.MonkeyPatch, scheme: str) -> None:
    source = _mae(seed=7)
    download = Mock(return_value={"model": source.state_dict()})
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", download)
    target = _classifier(seed=11)
    url = f"{scheme}://example.com/mae.pth"
    load_encoder_checkpoint(target, url)
    download.assert_called_once_with(url, map_location="cpu", weights_only=True)
    torch.testing.assert_close(target.norm.weight, source.norm.weight, rtol=0, atol=0)
    torch.testing.assert_close(target.cls_token, source.cls_token, rtol=0, atol=0)


def test_load_encoder_interpolates_patch_positions_and_preserves_cls(tmp_path: Path) -> None:
    source, target = _mae(img_size=64), _classifier(img_size=32)
    path = tmp_path / "larger-grid.pth"
    torch.save({"model": source.state_dict()}, path)
    positions = source.pos_embed.detach()
    expected_patches = (
        nn.functional.interpolate(
            positions[:, 1:].reshape(1, 4, 4, 32).permute(0, 3, 1, 2),
            size=(2, 2),
            mode="bicubic",
            align_corners=False,
        )
        .permute(0, 2, 3, 1)
        .reshape(1, 4, 32)
    )
    load_encoder_checkpoint(target, str(path))
    torch.testing.assert_close(target.pos_embed[:, :1], positions[:, :1], rtol=0, atol=0)
    torch.testing.assert_close(target.pos_embed[:, 1:], expected_patches, rtol=0, atol=0)


@pytest.mark.parametrize("bad_key", ["norm.weight", "norm.bias", "blocks.0.attn.qkv.weight"])
def test_load_encoder_refuses_missing_encoder_keys(tmp_path: Path, bad_key: str) -> None:
    state = _mae().state_dict()
    del state[bad_key]
    path = tmp_path / "incomplete.pth"
    torch.save({"model": state}, path)
    with pytest.raises(ValueError, match=f"missing keys.*{re.escape(bad_key)}"):
        load_encoder_checkpoint(_classifier(), str(path))


@pytest.mark.parametrize("bad_key", ["unrelated.weight", "decoder_unrelated.weight", "head.unrelated"])
def test_load_encoder_refuses_unrelated_keys(tmp_path: Path, bad_key: str) -> None:
    state = _mae().state_dict()
    state[bad_key] = torch.ones(1)
    path = tmp_path / "unrelated.pth"
    torch.save({"model_state_dict": state}, path)
    with pytest.raises(ValueError, match="unexpected keys"):
        load_encoder_checkpoint(_classifier(), str(path))


@pytest.mark.parametrize("bad_key", ["norm.weight", "patch_embed.proj.weight", "blocks.0.attn.qkv.weight"])
def test_load_encoder_refuses_incompatible_tensor_shapes(tmp_path: Path, bad_key: str) -> None:
    state = _mae().state_dict()
    state[bad_key] = state[bad_key][:1]
    path = tmp_path / "wrong-shape.pth"
    torch.save({"model": state}, path)
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_encoder_checkpoint(_classifier(), str(path))


@pytest.mark.parametrize("payload", [None, {}, {"model": []}, {"model_state_dict": None}])
def test_load_encoder_refuses_invalid_payload(tmp_path: Path, payload) -> None:
    path = tmp_path / "invalid.pth"
    torch.save(payload, path)
    with pytest.raises(TypeError, match="model_state_dict.*model"):
        load_encoder_checkpoint(_classifier(), str(path))


def test_probe_has_official_bn_frozen_encoder_and_head_only_lars(
    tmp_path: Path,
    tiny_classifier: None,
) -> None:
    source = _mae(seed=7)
    path = tmp_path / "source.pth"
    torch.save({"model": source.state_dict()}, path)
    exp = _probe_exp(tmp_path)
    torch.manual_seed(19)
    model = exp._build_probe_model(str(path))
    bn, head = model.head
    assert isinstance(bn, nn.BatchNorm1d) and not bn.affine and bn.eps == 1e-6
    assert bn.num_features == 32 and bn.track_running_stats
    assert not list(bn.parameters())
    assert isinstance(head, nn.Linear) and head.out_features == 6
    assert 0.007 < head.weight.std().item() < 0.013
    assert torch.count_nonzero(head.bias) == 0
    torch.testing.assert_close(model.norm.weight, source.norm.weight, rtol=0, atol=0)
    assert {name for name, parameter in model.named_parameters() if parameter.requires_grad} == {
        "head.1.weight",
        "head.1.bias",
    }
    cfg = MaeLinprobeExp.OptimizerCfg(weight_decay=0.2)
    optimizer = cfg.build_optimizer(model, SimpleNamespace(batch_size=4), SimpleNamespace(world_size=2), accum_iter=2)
    assert isinstance(optimizer, LARS)
    assert len(optimizer.param_groups) == 1
    group = optimizer.param_groups[0]
    assert {id(p) for p in group["params"]} == {id(head.weight), id(head.bias)}
    assert group["lr"] == pytest.approx(0.1 * 4 * 2 * 2 / 256)
    assert group["weight_decay"] == 0.2 and group["momentum"] == 0.9 and group["trust_coefficient"] == 0.001


def test_lars_matches_hand_calculated_weight_bias_and_momentum_updates() -> None:
    weight = nn.Parameter(torch.tensor([[3.0, 4.0]]))
    bias = nn.Parameter(torch.tensor([1.0, -2.0]))
    unused = nn.Parameter(torch.ones(1, 2))
    optimizer = LARS([weight, bias, unused], lr=0.2, weight_decay=0.1, momentum=0.9, trust_coefficient=0.01)
    weight.grad = torch.tensor([[0.6, 0.8]])
    bias.grad = torch.tensor([0.5, -0.25])
    # Weight: grad + wd*w = (.9, 1.2); trust=.01*5/1.5; mu=(.03, .04).
    optimizer.step()
    torch.testing.assert_close(weight, torch.tensor([[2.994, 3.992]]))
    torch.testing.assert_close(bias, torch.tensor([0.9, -1.95]))  # no decay or layer adaptation for 1-D
    torch.testing.assert_close(optimizer.state[weight]["mu"], torch.tensor([[0.03, 0.04]]))
    torch.testing.assert_close(weight.grad, torch.tensor([[0.6, 0.8]]))  # decay does not rewrite gradients
    torch.testing.assert_close(unused, torch.ones(1, 2))
    assert unused not in optimizer.state

    bias.grad = torch.tensor([1.0, 2.0])
    # Weight stays collinear: adapted update=(.02994, .03992), plus .9*previous mu.
    optimizer.step()
    torch.testing.assert_close(optimizer.state[weight]["mu"], torch.tensor([[0.05694, 0.07592]]))
    torch.testing.assert_close(weight, torch.tensor([[2.982612, 3.976816]]))
    torch.testing.assert_close(optimizer.state[bias]["mu"], torch.tensor([1.45, 1.775]))
    torch.testing.assert_close(bias, torch.tensor([0.61, -2.305]))


@pytest.mark.parametrize(
    "values,gradient,expected",
    [
        ([[0.0, 0.0]], [[3.0, 4.0]], [[-0.6, -0.8]]),
        ([[3.0, 4.0]], [[0.0, 0.0]], [[3.0, 4.0]]),
        ([[0.0, 0.0]], [[0.0, 0.0]], [[0.0, 0.0]]),
    ],
    ids=["zero-parameter-norm", "zero-update-norm", "both-zero"],
)
def test_lars_zero_norm_fallbacks_are_finite(values, gradient, expected) -> None:
    parameter = nn.Parameter(torch.tensor(values))
    parameter.grad = torch.tensor(gradient)
    optimizer = LARS([parameter], lr=0.2)
    optimizer.step()
    assert torch.isfinite(parameter).all() and torch.isfinite(optimizer.state[parameter]["mu"]).all()
    torch.testing.assert_close(parameter, torch.tensor(expected))


def test_probe_transforms_match_official_recipe() -> None:
    cfg = MaeLinprobeExp.DataloaderCfg(input_size=32)
    crop, flip, to_tensor, normalize = cfg._build_transform(True).transforms
    assert type(crop) is RandomResizedCrop and crop.size == (32, 32)
    assert crop.scale == (0.08, 1.0) and crop.ratio == (3 / 4, 4 / 3)
    assert crop.interpolation == transforms.InterpolationMode.BICUBIC
    assert type(flip) is transforms.RandomHorizontalFlip and type(to_tensor) is transforms.ToTensor
    assert type(normalize) is transforms.Normalize
    assert normalize.mean == IMAGENET_DEFAULT_MEAN and normalize.std == IMAGENET_DEFAULT_STD
    actual = cfg._build_transform(False)
    inherited = MaeExp.DataloaderCfg(input_size=32)._build_transform(False)
    image = Image.new("RGB", (48, 64), color=(42, 81, 153))
    torch.testing.assert_close(actual(image), inherited(image), rtol=0, atol=0)
    assert cfg._build_transform(True)(image).shape == (3, 32, 32)


@pytest.mark.parametrize("ratio,expected", [(0.25, (0, 13, 10, 7)), (4.0, (3, 0, 7, 20)), (2.0, (0, 0, 10, 20))])
@pytest.mark.parametrize("tensor_image", [False, True], ids=["pil", "tensor"])
def test_official_crop_clamps_each_dimension_instead_of_retrying(
    monkeypatch: pytest.MonkeyPatch,
    ratio: float,
    expected: tuple[int, int, int, int],
    tensor_image: bool,
) -> None:
    image = torch.zeros(3, 10, 20) if tensor_image else Image.new("RGB", (20, 10))
    # Select the final legal offset to also check randint's inclusive spatial bounds.
    monkeypatch.setattr(torch, "randint", lambda low, high, size: torch.tensor([high - 1]))
    assert RandomResizedCrop.get_params(image, (1.0, 1.0), (ratio, ratio)) == expected


@pytest.mark.parametrize(
    "case,message",
    [
        ("missing-source", "requires module_cfg.pretrained_from"),
        ("eval-without-resume", "eval requires resume_from"),
        ("same-output", "output directory must differ"),
        ("global-pool", "CLS pooling"),
        ("drop-path", "drop_path_rate=0"),
        ("small-batch", "batch >= 2"),
        ("zero-accum", "accum_iter >= 1"),
        ("too-few-steps", "enough steps"),
    ],
)
def test_run_guards_fire_before_logger_or_output_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
) -> None:
    exp = _probe_exp(tmp_path)
    if case == "missing-source":
        exp.dataloader_cfg.fake_data = False
    elif case == "eval-without-resume":
        exp.mode = "eval"
    elif case == "same-output":
        run_dir = Path(exp.get_run_dir())
        run_dir.mkdir()
        source = run_dir / "encoder.ckpt"
        source.write_bytes(b"pretraining artifact must survive")
        # Resolve .. as well as the simple string spelling of the shared directory.
        exp.module_cfg.pretrained_from = str(run_dir / ".." / run_dir.name / source.name)
    elif case == "global-pool":
        exp.module_cfg.global_pool = True
    elif case == "drop-path":
        exp.module_cfg.drop_path_rate = 0.1
    elif case == "small-batch":
        exp.dataloader_cfg.train_batch_size_per_device = 1
    elif case == "zero-accum":
        exp.accum_iter = 0
    elif case == "too-few-steps":
        exp.max_train_steps = 1
    logger, accelerator = Mock(), Mock()
    monkeypatch.setattr(exp.logger_cfg, "build_logger", logger)
    monkeypatch.setattr(exp.accelerator_cfg, "build_accelerator", accelerator)
    with pytest.raises(ValueError, match=message):
        exp.run()
    logger.assert_not_called()
    accelerator.assert_not_called()
    if case == "same-output":
        assert source.read_bytes() == b"pretraining artifact must survive"
        assert list(run_dir.iterdir()) == [source]
    else:
        assert not Path(exp.get_run_dir()).exists()


@pytest.mark.parametrize("mode", ["train", "eval"])
@pytest.mark.parametrize("task", [None, "mae_pretrain"])
def test_resume_refuses_non_probe_provenance_before_logger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    task: str | None,
) -> None:
    path = tmp_path / "pretrain.ckpt"
    payload = {"model_state_dict": _mae().state_dict(), "epoch": 0, "global_step": 4}
    if task is not None:
        payload["extra_state"] = {"task": task}
    torch.save(payload, path)
    exp = _probe_exp(tmp_path)
    exp.mode, exp.resume_from = mode, str(path)
    logger = Mock()
    monkeypatch.setattr(exp.logger_cfg, "build_logger", logger)
    with pytest.raises(ValueError, match="TinyExp linear-probe checkpoint.*pretrained_from"):
        exp.run()
    logger.assert_not_called()
    assert not Path(exp.get_run_dir()).exists()


def test_fake_train_resume_eval_preserves_encoder_updates_head_bn_and_checkpoint_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tiny_classifier: None,
) -> None:
    source_dir = tmp_path / "pretrain"
    source_dir.mkdir()
    source = source_dir / "last.ckpt"
    torch.save({"model_state_dict": _mae(seed=7).state_dict(), "epoch": 5}, source)
    source_bytes = source.read_bytes()
    exp = _probe_exp(tmp_path)
    exp.module_cfg.pretrained_from = str(source)
    exp.max_train_steps = 4  # four batches / accum=2 => two updates; the first LR is zero
    _patch_run_dependencies(monkeypatch, exp)
    initial = _capture_build(monkeypatch, exp)
    update_lrs = []
    original_step = LARS.step

    def step(optimizer):
        update_lrs.append(optimizer.param_groups[0]["lr"])
        return original_step(optimizer)

    monkeypatch.setattr(LARS, "step", step)
    exp.run()

    run_dir = Path(exp.get_run_dir())
    checkpoint_path = run_dir / "last.ckpt"
    trained = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = trained["model_state_dict"]
    assert update_lrs == pytest.approx([0, 0.1 * 4 * 2 / 256 * 0.5 / 10])
    assert trained["epoch"] == 0 and trained["global_step"] == 4
    assert trained["extra_state"] == {"task": "mae_linprobe"}
    assert trained["meta"]["exp_name"] == exp.exp_name and trained["meta"]["exp_class"] == exp.exp_class
    assert trained["meta"]["saved_at"]
    assert isinstance(trained["scaler_state_dict"], dict)
    assert len(trained["optimizer_state_dict"]["state"]) == 2
    assert all(torch.count_nonzero(value["mu"]) for value in trained["optimizer_state_dict"]["state"].values())
    encoder_keys = [key for key in initial if not key.startswith("head.")]
    for key in encoder_keys:
        torch.testing.assert_close(state[key], initial[key], rtol=0, atol=0)
    assert not torch.equal(state["head.1.weight"], initial["head.1.weight"])
    assert not torch.equal(state["head.1.bias"], initial["head.1.bias"])
    assert state["head.0.num_batches_tracked"].item() == 4
    assert not torch.equal(state["head.0.running_mean"], initial["head.0.running_mean"])
    assert not torch.equal(state["head.0.running_var"], initial["head.0.running_var"])
    assert exp.get_ray_run_result().startswith("linear probe best acc@1=")
    best = torch.load(run_dir / "best.ckpt", map_location="cpu", weights_only=False)
    assert best["epoch"] == 0 and best["best_metric"] == trained["best_metric"]

    resumed = _probe_exp(tmp_path)
    resumed.epochs, resumed.max_train_steps = 2, 8
    resumed.resume_from = str(checkpoint_path)
    resumed.module_cfg.pretrained_from = str(checkpoint_path)  # resume also bypasses the source/output guard
    _patch_run_dependencies(monkeypatch, resumed)
    # Resume must bypass MAE initialization, and restore optimizer momentum before prepare.
    encoder_load = Mock(side_effect=AssertionError("resume/eval must not load the encoder source"))
    monkeypatch.setattr("tinyexp.examples.mae_linprobe_exp.load_encoder_checkpoint", encoder_load)
    restored = {}
    load_scaler_state = NativeScaler.load_state_dict

    def capture_scaler_load(scaler, state_dict):
        load_scaler_state(scaler, state_dict)
        restored["scaler"] = scaler.state_dict().copy()

    monkeypatch.setattr(NativeScaler, "load_state_dict", capture_scaler_load)
    accelerator = resumed.accelerator_cfg.build_accelerator()

    def prepare(model, optimizer):
        restored["model"] = {key: value.clone() for key, value in model.state_dict().items()}
        restored["optimizer"] = {
            key: {name: value.clone() for name, value in values.items()}
            for key, values in optimizer.state_dict()["state"].items()
        }
        return model, optimizer

    accelerator.prepare = prepare
    resumed.run()
    assert restored["scaler"] == trained["scaler_state_dict"]
    for key, value in state.items():
        torch.testing.assert_close(restored["model"][key], value, rtol=0, atol=0)
    for key, value in trained["optimizer_state_dict"]["state"].items():
        torch.testing.assert_close(restored["optimizer"][key]["mu"], value["mu"], rtol=0, atol=0)
    last = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert last["epoch"] == 1 and last["global_step"] == 8
    assert len(update_lrs) == 4 and all(lr > 0 for lr in update_lrs[2:])
    assert last["model_state_dict"]["head.0.num_batches_tracked"].item() == 8
    assert not torch.equal(last["model_state_dict"]["head.1.weight"], state["head.1.weight"])
    assert last["best_metric"] >= trained["best_metric"]
    assert last["extra_state"] == {"task": "mae_linprobe"}
    for key in encoder_keys:
        torch.testing.assert_close(last["model_state_dict"][key], initial[key], rtol=0, atol=0)
    stats = [json.loads(line) for line in (run_dir / "log.txt").read_text().splitlines()]
    assert [entry["epoch"] for entry in stats] == [0, 1]
    assert all(math.isfinite(entry["train_loss"]) and math.isfinite(entry["test_loss"]) for entry in stats)
    assert all(0 <= entry["test_acc1"] <= entry["test_acc5"] <= 100 for entry in stats)
    assert all(entry["n_parameters"] == 32 * 6 + 6 for entry in stats)
    best = torch.load(run_dir / "best.ckpt", map_location="cpu", weights_only=False)
    assert best["best_metric"] == last["best_metric"]
    assert best["extra_state"] == {"task": "mae_linprobe"}

    evaluator = _probe_exp(tmp_path, exp_name="probe_eval")
    evaluator.mode, evaluator.resume_from = "eval", str(checkpoint_path)
    evaluator.module_cfg.pretrained_from = str(tmp_path / "also-nonexistent.pth")
    _patch_run_dependencies(monkeypatch, evaluator)
    evaluate = evaluator._evaluate
    observed = {}

    def capture_eval(accelerator, logger, model, device, val_dataloader):
        observed["before"] = {key: value.clone() for key, value in model.state_dict().items()}
        result = evaluate(accelerator, logger, model, device, val_dataloader)
        observed["after"] = model.state_dict()
        assert not model.training and all(parameter.grad is None for parameter in model.parameters())
        return result

    monkeypatch.setattr(evaluator, "_evaluate", capture_eval)
    checkpoint_bytes = checkpoint_path.read_bytes()
    evaluator.run()
    assert evaluator.get_ray_run_result().startswith("eval acc@1=")
    for key, value in last["model_state_dict"].items():
        torch.testing.assert_close(observed["before"][key], value, rtol=0, atol=0)
        torch.testing.assert_close(observed["after"][key], value, rtol=0, atol=0)
    assert checkpoint_path.read_bytes() == checkpoint_bytes and source.read_bytes() == source_bytes
    encoder_load.assert_not_called()
