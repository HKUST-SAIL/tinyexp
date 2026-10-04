"""MAE ViT-B frozen-encoder linear probing (official main_linprobe.py recipe).

Run with ``python -m tinyexp.examples.mae_linprobe_exp``; see docs/mae.md for
initialization, resume, and eval commands. No pretraining artifacts are overwritten.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import torch
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import trunc_normal_
from timm.utils import NativeScaler
from torch import nn
from torchvision import transforms
from torchvision.transforms import functional as F

from tinyexp import store_and_run_exp
from tinyexp.examples.mae_exp import MaeExp, adjust_learning_rate, interpolate_pos_embed
from tinyexp.examples.vit_tp_exp import MetricLogger, SmoothedValue


# Direct ports of facebookresearch/mae util/lars.py and util/crop.py (CC BY-NC 4.0),
# with the current TorchVision public image-size API. Keep the official optimizer math.
class LARS(torch.optim.Optimizer):
    def __init__(self, params, lr=0.0, weight_decay=0.0, momentum=0.9, trust_coefficient=0.001):
        super().__init__(
            params,
            {"lr": lr, "weight_decay": weight_decay, "momentum": momentum, "trust_coefficient": trust_coefficient},
        )

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for parameter in group["params"]:
                update = parameter.grad
                if update is None:
                    continue
                if parameter.ndim > 1:
                    update = update.add(parameter, alpha=group["weight_decay"])
                    param_norm, update_norm = torch.norm(parameter), torch.norm(update)
                    one = torch.ones_like(param_norm)
                    trust = torch.where(
                        param_norm > 0,
                        torch.where(update_norm > 0, group["trust_coefficient"] * param_norm / update_norm, one),
                        one,
                    )
                    update = update.mul(trust)
                state = self.state[parameter]
                if "mu" not in state:
                    state["mu"] = torch.zeros_like(parameter)
                momentum = state["mu"]
                momentum.mul_(group["momentum"]).add_(update)
                parameter.add_(momentum, alpha=-group["lr"])


class RandomResizedCrop(transforms.RandomResizedCrop):
    @staticmethod
    def get_params(img, scale, ratio):
        width, height = F.get_image_size(img)
        target_area = height * width * torch.empty(1).uniform_(*scale).item()
        log_ratio = torch.log(torch.tensor(ratio))
        aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()
        w = min(round(math.sqrt(target_area * aspect_ratio)), width)
        h = min(round(math.sqrt(target_area / aspect_ratio)), height)
        i = torch.randint(0, height - h + 1, size=(1,)).item()
        j = torch.randint(0, width - w + 1, size=(1,)).item()
        return i, j, h, w


def load_encoder_checkpoint(model: nn.Module, source: str) -> None:
    """Load all encoder tensors, ignoring only the MAE decoder and old classifier head."""
    if source.startswith("https://") or source.startswith("http://"):
        checkpoint = torch.hub.load_state_dict_from_url(source, map_location="cpu", weights_only=True)
    else:
        checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("Encoder checkpoint must contain model_state_dict (TinyExp) or model (official MAE)")
    state = checkpoint.get("model_state_dict", checkpoint.get("model"))
    if not isinstance(state, dict):
        raise TypeError("Encoder checkpoint must contain model_state_dict (TinyExp) or model (official MAE)")
    state = {
        key: value
        for key, value in state.items()
        if key not in {"mask_token", "decoder_pos_embed", "head.weight", "head.bias"}
        and not key.startswith(("decoder_embed.", "decoder_blocks.", "decoder_norm.", "decoder_pred."))
    }
    expected = set(model.state_dict()) - {"head.weight", "head.bias"}
    missing, unexpected = expected - set(state), set(state) - expected
    if missing or unexpected:
        raise ValueError(f"Invalid MAE encoder: missing keys {sorted(missing)}, unexpected keys {sorted(unexpected)}")
    interpolate_pos_embed(model, state)
    model.load_state_dict(state, strict=False)  # only the fresh linear head is absent; shapes are still checked


@dataclass(repr=False)
class MaeLinprobeExp(MaeExp):
    epochs: int = 90
    accum_iter: int = 4  # 512 * 4 * 8 GPUs = official effective batch 16384
    eval_every_n_epochs: int = 1  # probe always validates each epoch, independent of pretraining monitors

    @dataclass
    class ModuleCfg(MaeExp.ModuleCfg):
        global_pool: bool = False
        drop_path_rate: float = 0.0

    module_cfg: ModuleCfg = field(default_factory=ModuleCfg)

    @dataclass
    class DataloaderCfg(MaeExp.DataloaderCfg):
        train_batch_size_per_device: int = 512

        def _build_transform(self, is_train: bool):
            if not is_train:
                return super()._build_transform(False)
            return transforms.Compose(
                [
                    RandomResizedCrop(
                        self.input_size, scale=(0.08, 1.0), interpolation=transforms.InterpolationMode.BICUBIC
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
                ]
            )

    dataloader_cfg: DataloaderCfg = field(default_factory=DataloaderCfg)

    @dataclass
    class OptimizerCfg(MaeExp.OptimizerCfg):
        blr: float = 0.1
        weight_decay: float = 0.0

        def build_optimizer(self, module, dataloader, accelerator, accum_iter: int = 1):
            return LARS(
                [p for p in module.parameters() if p.requires_grad],
                lr=self.scaled_lr(dataloader, accelerator, accum_iter),
                weight_decay=self.weight_decay,
            )

    optimizer_cfg: OptimizerCfg = field(default_factory=OptimizerCfg)

    @dataclass
    class LrSchedulerCfg(MaeExp.LrSchedulerCfg):
        warmup_epochs: int = 10

    lr_scheduler_cfg: LrSchedulerCfg = field(default_factory=LrSchedulerCfg)

    @dataclass
    class CheckpointCfg(MaeExp.CheckpointCfg):
        def _validate_checkpoint_payload(self, path, checkpoint):
            checkpoint = super()._validate_checkpoint_payload(path, checkpoint)
            if checkpoint.get("extra_state", {}).get("task") != "mae_linprobe":
                raise ValueError(
                    "resume_from requires a TinyExp linear-probe checkpoint; "
                    "use module_cfg.pretrained_from to initialize from an MAE encoder"
                )
            return checkpoint

    checkpoint_cfg: CheckpointCfg = field(default_factory=CheckpointCfg)

    def run(self) -> None:
        # Guard before super().run() creates the logger or writes artifacts. Resume takes precedence.
        if self.resume_from:
            self.checkpoint_cfg.load_checkpoint(self.resume_from, map_location="cpu")
        elif self.mode == "train":
            source = self.module_cfg.pretrained_from
            if not source and not self.dataloader_cfg.fake_data:
                raise ValueError("Linear probing requires module_cfg.pretrained_from (MAE encoder) or resume_from")
            if (
                source
                and not source.startswith(("https://", "http://"))
                and Path(source).resolve().parent == Path(self.get_run_dir()).resolve()
            ):
                raise ValueError("Probe output directory must differ from the pretrained encoder directory")
        elif self.mode == "eval":
            raise ValueError("Linear-probe eval requires resume_from=<probe best.ckpt or last.ckpt>")
        if self.module_cfg.global_pool or self.module_cfg.drop_path_rate != 0:
            raise ValueError("Official linear probing requires CLS pooling and drop_path_rate=0")
        if self.mode == "train" and (
            self.accum_iter < 1
            or self.dataloader_cfg.train_batch_size_per_device < 2
            or 0 < self.max_train_steps < self.accum_iter
        ):
            raise ValueError("Probe training requires batch >= 2, accum_iter >= 1, and enough steps for an update")
        super().run()

    def _build_probe_model(self, source: str = "") -> nn.Module:
        model = self.module_cfg.build_classifier()
        if source:
            load_encoder_checkpoint(model, source)
        trunc_normal_(model.head.weight, std=0.01)
        nn.init.zeros_(model.head.bias)
        model.head = nn.Sequential(nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
        model.requires_grad_(False)
        model.head.requires_grad_(True)  # also retained for DDP eval; _evaluate itself is no-grad
        return model

    def _train(self, accelerator, logger, cfg_dict, run_dir: str) -> None:
        train_loader = self.dataloader_cfg.build_train_dataloader(accelerator, self.redis_cfg)
        val_loader = self.dataloader_cfg.build_val_dataloader(accelerator)
        if len(train_loader) < self.accum_iter or len(val_loader) == 0:
            raise ValueError("Probe requires a nonempty val loader and at least accum_iter training batches")
        model = self._build_probe_model("" if self.resume_from else self.module_cfg.pretrained_from)
        model.to(accelerator.device)
        lr = self.optimizer_cfg.scaled_lr(train_loader, accelerator, self.accum_iter)
        optimizer = self.optimizer_cfg.build_optimizer(model, train_loader, accelerator, self.accum_iter)
        loss_scaler = NativeScaler(device=accelerator.device.type)
        start_epoch, global_step, best = 0, 0, float("-inf")
        if self.resume_from:
            checkpoint = self.checkpoint_cfg.load_checkpoint(
                self.resume_from, model=model, optimizer=optimizer, scaler=loss_scaler, map_location=accelerator.device
            )
            start_epoch = int(checkpoint["epoch"]) + 1
            global_step = int(checkpoint["global_step"])
            best = float(checkpoint["best_metric"])
        model, optimizer = accelerator.prepare(model, optimizer)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Trainable head parameters: {n_parameters}")
        logger.info(f"effective batch size: {train_loader.batch_size * self.accum_iter * accelerator.world_size}")
        logger.info(f"base lr: {self.optimizer_cfg.blr:.2e}; actual lr: {lr:.2e}")
        if self.wandb_cfg.enable_wandb and accelerator.is_main_process:
            self.wandb_cfg.build_wandb(accelerator=accelerator, project="TinyExp", config=cfg_dict, name=self.exp_name)
        logger.info(f"Start linear probing for {self.epochs} epochs")
        for epoch in range(start_epoch, self.epochs):
            if hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)
            train_stats, global_step = self._train_one_epoch(
                accelerator,
                logger,
                model,
                train_loader,
                optimizer,
                accelerator.device,
                epoch,
                loss_scaler,
                lr,
                global_step,
            )
            test_stats = self._evaluate(accelerator, logger, model, accelerator.device, val_loader)
            improved = test_stats["acc1"] > best
            best = max(best, test_stats["acc1"])
            logger.info(f"Best linear-probe top-1: {best:.2f}%")
            if accelerator.is_main_process:
                for name in (
                    [self.checkpoint_cfg.last_ckpt_name, self.checkpoint_cfg.best_ckpt_name]
                    if improved
                    else [self.checkpoint_cfg.last_ckpt_name]
                ):
                    self.checkpoint_cfg.save_checkpoint(
                        run_dir=run_dir,
                        name=name,
                        model=accelerator.unwrap_model(model),
                        optimizer=optimizer,
                        scaler=loss_scaler,
                        epoch=epoch,
                        global_step=global_step,
                        best_metric=best,
                        exp_name=self.exp_name,
                        exp_class=self.exp_class,
                        extra_state={"task": "mae_linprobe"},
                    )
                log_stats = {
                    **{f"train_{k}": v for k, v in train_stats.items()},
                    **{f"test_{k}": v for k, v in test_stats.items()},
                    "best_acc1": best,
                    "epoch": epoch,
                    "n_parameters": n_parameters,
                }
                with open(os.path.join(run_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")
                if self.wandb_cfg.enable_wandb:
                    import wandb

                    wandb.log(log_stats)
            if 0 < self.max_train_steps <= global_step:
                break
        self._run_result = f"linear probe best acc@1={best:.2f}%"

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
        model.train(True)
        metric_logger = MetricLogger(log_fn=logger.info)
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        optimizer.zero_grad()
        adjust = partial(
            adjust_learning_rate,
            optimizer,
            lr=lr,
            min_lr=self.lr_scheduler_cfg.min_lr,
            warmup_epochs=self.lr_scheduler_cfg.warmup_epochs,
            epochs=self.epochs,
        )
        for step, (images, target) in enumerate(metric_logger.log_every(data_loader, 20, f"Epoch: [{epoch}]")):
            if step % self.accum_iter == 0:
                adjust(step / len(data_loader) + epoch)
            images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)
            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                loss = nn.functional.cross_entropy(model(images), target)
            loss_value = loss.item()
            if not math.isfinite(loss_value):
                raise FloatingPointError(f"Nonfinite probe loss: {loss_value}")
            loss_scaler(
                loss / self.accum_iter,
                optimizer,
                parameters=[p for p in model.parameters() if p.requires_grad],
                need_update=(step + 1) % self.accum_iter == 0,
            )
            if (step + 1) % self.accum_iter == 0:
                optimizer.zero_grad()
            if device.type == "cuda":
                torch.cuda.synchronize()
            metric_logger.update(loss=loss_value, lr=optimizer.param_groups[0]["lr"])
            global_step += 1
            if 0 < self.max_train_steps <= global_step:
                break
        # Like upstream, an incomplete accumulation window is discarded at the next epoch.
        metric_logger.synchronize_between_processes(accelerator)
        logger.info(f"Averaged stats: {metric_logger}")
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, global_step

    def _prepare_for_eval(self, accelerator, logger):
        model = self._build_probe_model()
        self.checkpoint_cfg.load_checkpoint(self.resume_from, model=model, map_location="cpu")
        val_loader = self.dataloader_cfg.build_val_dataloader(accelerator)
        if len(val_loader) == 0:
            raise ValueError("Probe evaluation requires a nonempty validation loader")
        model.to(accelerator.device)
        return val_loader, model


if __name__ == "__main__":
    store_and_run_exp(MaeLinprobeExp)
