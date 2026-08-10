import datetime
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

from tinyexp import TinyExp, store_and_run_exp
from tinyexp.exceptions import UnknownAcceleratorTypeError
from tinyexp.exp_mixins import CheckpointCfgMixin, LoggerCfgMixin, RayCfgMixin, WandbCfgMixin
from tinyexp.tiny_engine.accelerator import AcceleratorProtocol


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.loss = F.nll_loss

    def forward(self, x, target=None, onnx_exporting=False) -> torch.Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        if onnx_exporting:
            return x
        output = F.log_softmax(x, dim=1)

        if self.training and target is not None:
            return self.loss(output, target)
        else:
            return output


@dataclass(repr=False)
class Exp(TinyExp, RayCfgMixin, CheckpointCfgMixin, WandbCfgMixin, LoggerCfgMixin):
    @dataclass
    class RayCfg(RayCfgMixin.RayCfg):
        ray_num_worker: int = 2
        ray_num_cpus_per_worker: int = 4  # Main process plus train/validation dataloader workers.
        ray_num_gpus_per_worker: float = 0.0  # 0.0 means do not use GPU

    ray_cfg: RayCfg = field(default_factory=RayCfg)
    mode: str = "train"
    launcher: str = "ray"

    @dataclass
    class AcceleratorCfg:
        accelerator: str = "cpu"

        def build_accelerator(self) -> AcceleratorProtocol:
            from tinyexp.tiny_engine.accelerator import CPUAccelerator, DDPAccelerator

            if self.accelerator == "cpu":
                accelerator = CPUAccelerator()
            elif self.accelerator == "ddp":
                accelerator = DDPAccelerator()
            else:
                raise UnknownAcceleratorTypeError(self.accelerator)
            return accelerator

    accelerator_cfg: AcceleratorCfg = field(default_factory=AcceleratorCfg)

    @dataclass
    class DataloaderCfg:
        data_root: str = "./data/"
        train_batch_size_per_device: int = 32
        train_data_worker_per_gpu: int = 2
        val_data_worker_per_gpu: int = 1

        def build_train_dataloader(self, accelerator):
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            ds_train = datasets.MNIST(self.data_root, train=True, download=True, transform=transform)
            sampler = torch.utils.data.DistributedSampler(
                ds_train, num_replicas=accelerator.world_size, rank=accelerator.rank
            )
            dl_train = torch.utils.data.DataLoader(
                ds_train,
                batch_size=self.train_batch_size_per_device,
                shuffle=False,
                num_workers=self.train_data_worker_per_gpu,
                drop_last=True,
                sampler=sampler,
            )
            return dl_train

        def build_val_dataloader(self, accelerator):
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            ds_val = datasets.MNIST(self.data_root, train=False, download=True, transform=transform)
            sampler = torch.utils.data.DistributedSampler(
                ds_val, num_replicas=accelerator.world_size, rank=accelerator.rank
            )
            dl_val = torch.utils.data.DataLoader(
                ds_val,
                batch_size=self.train_batch_size_per_device,
                shuffle=False,
                num_workers=self.val_data_worker_per_gpu,
                drop_last=True,
                sampler=sampler,
            )
            return dl_val

    dataloader_cfg: DataloaderCfg = field(default_factory=DataloaderCfg)

    @dataclass
    class OptimizerCfg:
        lr_per_img: float = 1.0 / 64.0  # single image learning rate

        def build_optimizer(self, module, dataloader, accelerator):
            optimizer = optim.Adadelta(
                module.parameters(),
                lr=self.lr_per_img * dataloader.batch_size * accelerator.world_size,
            )
            return optimizer

    optimizer_cfg: OptimizerCfg = field(default_factory=OptimizerCfg)

    @dataclass
    class ModuleCfg:
        def build_module(self):
            return Net()

    module_cfg: ModuleCfg = field(default_factory=ModuleCfg)

    @dataclass
    class LrSchedulerCfg:
        def build_lr_scheduler(self, optimizer):
            return StepLR(optimizer, step_size=1, gamma=0.7)

    lr_scheduler_cfg: LrSchedulerCfg = field(default_factory=LrSchedulerCfg)

    # ------------------------------ bellowing is the execution part --------------------- #
    def run(self) -> None:
        accelerator = self.accelerator_cfg.build_accelerator()
        run_dir = self.get_run_dir()
        logger = self.logger_cfg.build_logger(save_dir=run_dir, distributed_rank=accelerator.rank)
        cfg_dict = self.print_cfg(logger)

        if self.mode == "train":
            self._train(
                accelerator=accelerator,
                logger=logger,
                cfg_dict=cfg_dict,
                run_dir=run_dir,
            )
        elif self.mode == "val":
            if not self.resume_from:
                raise ValueError("resume_from is required when mode='val'")  # noqa: TRY003
            self._evaluate(
                accelerator=accelerator,
                logger=logger,
                module_or_module_path=self.resume_from,
            )
        else:
            raise NotImplementedError(f"Mode {self.mode} is not implemented")

    def _evaluate(self, accelerator, logger, module_or_module_path, val_dataloader=None) -> float:
        if isinstance(module_or_module_path, str):
            module = Net()
            self.checkpoint_cfg.load_checkpoint(
                module_or_module_path,
                model=module,
                map_location=accelerator.device,
            )
            module = accelerator.prepare(module)
        else:
            module = module_or_module_path

        if val_dataloader is None:
            val_dataloader = self.dataloader_cfg.build_val_dataloader(accelerator)

        module.eval()
        accurate = torch.tensor(0.0, device=accelerator.device)

        for batch in val_dataloader:
            features, labels = (item.to(accelerator.device) for item in batch)
            with torch.no_grad():
                preds = module(features)
            predictions = preds.argmax(dim=-1)
            accurate_preds = predictions == labels
            accurate_preds_sum = accelerator.reduce_sum(accurate_preds.sum())
            accurate += accurate_preds_sum
        eval_metric = accurate.item() / len(val_dataloader.dataset)

        accelerator.wait_for_everyone()
        nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"{nowtime} --> eval_metric= {100 * eval_metric:.2f}%")

        if self.wandb_cfg.enable_wandb and accelerator.is_main_process:
            wandb.log({"val_metric": eval_metric})

        return eval_metric

    def _train(self, accelerator, logger, cfg_dict, run_dir: str) -> None:  # noqa: C901
        train_dataloader = self.dataloader_cfg.build_train_dataloader(accelerator)
        val_dataloader = self.dataloader_cfg.build_val_dataloader(accelerator)
        ori_module = self.module_cfg.build_module()
        ori_optimizer = self.optimizer_cfg.build_optimizer(ori_module, train_dataloader, accelerator)
        lr_scheduler = self.lr_scheduler_cfg.build_lr_scheduler(ori_optimizer)

        module, optimizer = accelerator.prepare(ori_module, ori_optimizer)
        start_epoch = 0
        global_step = 0
        best_metric = None
        if self.resume_from:
            checkpoint = self.checkpoint_cfg.load_checkpoint(
                self.resume_from,
                model=accelerator.unwrap_model(module),
                optimizer=optimizer,
                scheduler=lr_scheduler,
                map_location=accelerator.device,
            )
            start_epoch = int(checkpoint.get("epoch", -1)) + 1
            global_step = int(checkpoint.get("global_step", 0))
            best_metric = checkpoint.get("best_metric")

        train_iter = iter(train_dataloader)
        if self.wandb_cfg.enable_wandb and accelerator.rank == 0:
            self.wandb_cfg.build_wandb(
                accelerator=accelerator,
                project="Baselines",
                config=cfg_dict,
            )

        for epoch in range(start_epoch, 3):
            module.train()

            for step in range(len(train_dataloader)):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_dataloader)
                    batch = next(train_iter)

                features, labels = (item.to(accelerator.device) for item in batch)
                preds = module(features)
                loss = nn.CrossEntropyLoss()(preds, labels)

                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()
                global_step += 1
                if (step + 1) % 20 == 0:
                    logger.info(f"epoch {epoch} loss: {loss.item(): .4f} lr: {optimizer.param_groups[0]['lr']: .4f}")
                    if self.wandb_cfg.enable_wandb and accelerator.rank == 0:
                        wandb.log(
                            {
                                "epoch": epoch,
                                "loss": loss.item(),
                                "lr": optimizer.param_groups[0]["lr"],
                            }
                        )
            eval_metric = self._evaluate(
                accelerator=accelerator,
                logger=logger,
                module_or_module_path=module,
                val_dataloader=val_dataloader,
            )
            lr_scheduler.step()
            is_best = best_metric is None or eval_metric > best_metric
            if is_best:
                best_metric = eval_metric

            if accelerator.is_main_process:
                self.checkpoint_cfg.save_checkpoint(
                    run_dir=run_dir,
                    name=self.checkpoint_cfg.last_ckpt_name,
                    model=accelerator.unwrap_model(module),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    epoch=epoch,
                    global_step=global_step,
                    best_metric=best_metric,
                    exp_name=self.exp_name,
                    exp_class=self.exp_class,
                )
                if is_best:
                    self.checkpoint_cfg.save_checkpoint(
                        run_dir=run_dir,
                        name=self.checkpoint_cfg.best_ckpt_name,
                        model=accelerator.unwrap_model(module),
                        optimizer=optimizer,
                        scheduler=lr_scheduler,
                        epoch=epoch,
                        global_step=global_step,
                        best_metric=best_metric,
                        exp_name=self.exp_name,
                        exp_class=self.exp_class,
                    )


# import hydra
# @hydra.main(version_base=None, config_name="cfg")
# def simple_launch_exp(cfg: DictConfig) -> None:
#     from omegaconf import DictConfig, OmegaConf
#     print(OmegaConf.to_yaml(cfg))
#     exp_class = hydra.utils.get_class(cfg.exp_class)
#     exp_class().set_cfg(cfg).run()

if __name__ == "__main__":
    store_and_run_exp(Exp)
