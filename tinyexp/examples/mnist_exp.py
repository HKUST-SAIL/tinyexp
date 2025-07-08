import datetime
from dataclasses import dataclass

import hydra
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from tqdm import tqdm

from tinyexp import TinyExp
from tinyexp.utils.ray_utils import get_num_gpus_worker_options


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


class MnistExp(TinyExp):
    @dataclass
    class Config:
        data_root: str = "./data/"
        accelerator: str = "ddp"  # "cpu", "ddp"
        train_lr_per_img: float = 1.0 / 64.0  # single image learning rate
        train_batch_size_per_device: int = 32
        train_max_epoch: int = 3
        train_enable_wandb: bool = False
        launch: str = "ray"  # "ray", "local"

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.train_dataloader = self._configure_train_dataloader()
        self.val_dataloader = self._configure_val_dataloader()

    def _configure_accelerator(self):
        from tinyexp.tiny_engine.accelerator import CPUAccelerator, DDPAccelerator

        if self.cfg.accelerator == "cpu":
            accelerator = CPUAccelerator()
        elif self.cfg.accelerator == "ddp":
            accelerator = DDPAccelerator()
        else:
            raise ValueError(f"Unknown accelerator type: {self.cfg.accelerator}")
        return accelerator

    def _configure_module(self):
        return Net()

    def _configure_train_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        ds_train = datasets.MNIST(self.cfg.data_root, train=True, download=True, transform=transform)
        sampler = torch.utils.data.distributed.DistributedSampler(
            ds_train, num_replicas=self.accelerator.world_size, rank=self.accelerator.rank
        )
        dl_train = torch.utils.data.DataLoader(
            ds_train,
            batch_size=self.cfg.train_batch_size_per_device,
            shuffle=False,
            num_workers=2,
            drop_last=True,
            sampler=sampler,
        )
        return dl_train

    def _configure_val_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        ds_val = datasets.MNIST(self.cfg.data_root, train=False, download=True, transform=transform)
        sampler = torch.utils.data.distributed.DistributedSampler(
            ds_val, num_replicas=self.accelerator.world_size, rank=self.accelerator.rank
        )
        dl_val = torch.utils.data.DataLoader(
            ds_val,
            batch_size=self.cfg.train_batch_size_per_device,
            shuffle=False,
            num_workers=2,
            drop_last=True,
            sampler=sampler,
        )
        return dl_val

    def _configure_optimizer(self):
        cfg = self.cfg
        optimizer = optim.Adadelta(
            self.module.parameters(),
            lr=cfg.train_lr_per_img * cfg.train_batch_size_per_device * self.accelerator.world_size,
        )
        return optimizer

    def _configure_lr_scheduler(self):
        return StepLR(self.optimizer, step_size=1, gamma=0.7)

    def run(self) -> None:
        accelerator = self.accelerator
        module, optimizer = accelerator.prepare(self.module, self.optimizer)

        lr_scheduler, train_dataloader, val_dataloader = self.lr_scheduler, self.train_dataloader, self.val_dataloader
        accelerator.print(f"device {accelerator.device!s} is used!")

        train_iter = iter(train_dataloader)
        if self.cfg.train_enable_wandb and accelerator.rank == 0:
            wandb.init(config=self.cfg, project="Baselines", name=self.exp_name)

        for epoch in range(self.cfg.train_max_epoch):
            module.train()

            for step in tqdm(
                range(len(train_dataloader)),
                ncols=100,
                desc="Train",
                bar_format="{n_fmt}/{total_fmt} [{elapsed}<{remaining}] {l_bar}{bar:50}|",
                colour="blue",
                ascii=" ·─",
                unit="batch",
                disable=accelerator.rank != 0,
            ):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_dataloader)
                    batch = next(train_iter)

                features, labels = (_.to(accelerator.device) for _ in batch)
                preds = module(features)
                loss = nn.CrossEntropyLoss()(preds, labels)

                optimizer.zero_grad()
                # ======================================================================
                accelerator.backward(loss)  # loss.backward()
                # ======================================================================

                optimizer.step()
                if step % 20 == 0:
                    self.logger.info(
                        f"epoch {epoch} loss: {loss.item(): .4f} lr: {optimizer.param_groups[0]['lr']: .4f}"
                    )
                    if self.cfg.train_enable_wandb and accelerator.rank == 0:
                        wandb.log({"epoch": epoch, "loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]})
            self.eval(module, val_dataloader)
            lr_scheduler.step()

        # dump model
        # state_dict = accelerator.dump_model_to_state_dict()
        # if accelerator.rank == 0:
        #     self.logger.info(f"Dumping model to {self.exp_name}.pth")
        #     torch.save(state_dict, f"{self.exp_name}.pth")

    def eval(self, module_or_module_path, val_dataloader=None):
        if isinstance(module_or_module_path, str):
            module = Net()
            module.load_state_dict(torch.load(module_or_module_path))
            module = self.accelerator.prepare(module)
            val_dataloader = self.val_dataloader
        else:
            module = module_or_module_path

        module.eval()
        accurate = 0.0

        for _, batch in enumerate(val_dataloader):
            features, labels = (_.to(self.accelerator.device) for _ in batch)
            with torch.no_grad():
                preds = module(features)
            predictions = preds.argmax(dim=-1)
            accurate_preds = predictions == labels
            accurate_preds_sum = self.accelerator.reduce_sum(accurate_preds.sum())
            accurate += accurate_preds_sum
        eval_metric = accurate.item() / len(val_dataloader.dataset)

        print(f"======> data_shard_count: {len(val_dataloader.dataset)}")
        # ======================================================================
        self.accelerator.wait_for_everyone()
        nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logger.info(f"{nowtime} --> eval_metric= {100 * eval_metric:.2f}%")

        if self.cfg.train_enable_wandb and self.accelerator.is_main_process:
            wandb.log({"val_metric": eval_metric})


@hydra.main(version_base=None, config_name="cfg")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    if cfg.launch == "ray":
        ray.init()
        remote_exp = ray.remote(MnistExp)
        options_list = get_num_gpus_worker_options(torch.cuda.device_count())
        worker_group = [remote_exp.options(**options).remote(cfg) for options in options_list]
        run_futures = [worker.run.remote() for worker in worker_group]
        ray.get(run_futures)
        ray.shutdown()
    else:
        # MnistExp(cfg).run()
        MnistExp(cfg).eval("xxx.pth")


if __name__ == "__main__":
    ConfigStore.instance().store(name="cfg", node=MnistExp.Config)
    main()
