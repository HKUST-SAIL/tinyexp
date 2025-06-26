import hydra
import ray
import datetime
from dataclasses import dataclass
from functools import cached_property

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

# from tinyexp.tiny_engine import CPU
from tinyexp import TinyExp


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


@ray.remote
class MnistExp(TinyExp):
    @dataclass
    class Config:
        data_root: str = "./data/"
        accelerator: str = "cpu"  # "cpu", "gpu", "ddp"
        train_lr_per_img: float = 1.0 / 64.0  # single image learning rate
        train_batch_size_per_device: int = 32
        train_max_epoch: int = 3
        train_enable_wandb: bool = False

    @cached_property
    def module(self):
        return Net()

    @cached_property
    def train_dataloader(self):
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

    @cached_property
    def val_dataloader(self):
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

    @cached_property
    def optimizer(self):
        cfg = self.cfg
        optimizer = optim.Adadelta(
            self.module.parameters(),
            lr=cfg.train_lr_per_img * cfg.train_batch_size_per_device * self.accelerator.world_size,
        )
        return optimizer

    @cached_property
    def lr_scheduler(self):
        return StepLR(self.optimizer, step_size=1 * len(self.train_dataloader), gamma=0.7)

    @cached_property
    def accelerator(self):
        from tinyexp.tiny_engine.accelerator import CPUAccelerator, DDPAccelerator
        if self.cfg.accelerator == "cpu":
            accelerator = CPUAccelerator()
        elif self.cfg.accelerator == "ddp":
            accelerator = DDPAccelerator()
            accelerator._init_process_group()
        else:
            raise ValueError(f"Unknown accelerator type: {self.cfg.accelerator}")
        return accelerator

    def run(self) -> None:
        accelerator = self.accelerator
        module, optimizer = accelerator.prepare(self.module, self.optimizer)

        lr_scheduler, train_dataloader, val_dataloader = self.lr_scheduler, self.train_dataloader, self.val_dataloader
        accelerator.print(f"device {accelerator.device!s} is used!")

        train_iter = iter(train_dataloader)
        if self.cfg.train_enable_wandb and accelerator.is_main_process:
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
                disable=not accelerator.is_local_main_process,
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
                        f'epoch {epoch} loss: {loss.item(): .4f} lr: {optimizer.param_groups[0]["lr"]: .4f}'
                    )
                    if self.cfg.train_enable_wandb and accelerator.is_main_process:
                        wandb.log({"epoch": epoch, "loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]})
                lr_scheduler.step()

            module.eval()
            accurate = 0.0

            for _, batch in enumerate(val_dataloader):
                features, labels = (_.to(accelerator.device) for _ in batch)
                with torch.no_grad():
                    preds = module(features)
                predictions = preds.argmax(dim=-1)
                accurate_preds = predictions == labels
                accurate_preds_sum = accelerator.reduce_sum(accurate_preds.sum())
                accurate += accurate_preds_sum

            eval_metric = accurate.item() / len(val_dataloader.dataset)

            # ======================================================================
            # print logs and save ckpt
            accelerator.wait_for_everyone()
            nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.logger.info(f"epoch【{epoch}】@{nowtime} --> eval_metric= {100 * eval_metric:.2f}%")

            if self.cfg.train_enable_wandb and accelerator.is_main_process:
                wandb.log({"eval_metric": eval_metric})

from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import socket

def create_placement_group(num_gpus):
    """Create and return a placement group for GPU allocation."""
    bundles = [{"CPU": 10, "GPU": 1} for _ in range(num_gpus)]
    pg = placement_group(bundles=bundles, strategy="STRICT_PACK")
    ray.get(pg.ready())
    return pg

def create_worker_options(pg, rank, local_rank, num_gpus, master_addr, master_port):
    """Create options for Ray workers."""
    return {
        'runtime_env': {
            'env_vars': {
                'WORLD_SIZE': str(num_gpus),
                'RANK': str(rank),
                'MASTER_ADDR': master_addr,
                'MASTER_PORT': str(master_port),
                "LOCAL_RANK": str(local_rank),
            }
        },
        'scheduling_strategy': PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_bundle_index=rank
        ),
        'num_gpus': 1.0
    }

def get_network_config():
    """Get network configuration for distributed setup."""
    master_addr = ray._private.services.get_node_ip_address()
    with socket.socket() as sock:
        sock.bind(('', 0))
        master_port = sock.getsockname()[1]
    return master_addr, master_port

@hydra.main(version_base=None, config_name="cfg")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    ray.init()

    num_gpus = 2
    pg = create_placement_group(num_gpus=num_gpus)
    master_addr, master_port = get_network_config()
    worker_group = []
    for i in range(num_gpus):
        options = create_worker_options(pg, i, i, num_gpus, master_addr, master_port)
        worker_group.append(MnistExp.options(**options).remote(cfg))

    run_futures = []
    for i in range(num_gpus):
        run_futures.append(worker_group[i].run.remote())
    
    # Wait for all run tasks to complete
    ray.get(run_futures)


if __name__ == "__main__":
    ConfigStore.instance().store(name="cfg", node=MnistExp.Config)
    main()