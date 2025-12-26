import datetime
import io
import os
import time
from contextlib import suppress
from dataclasses import dataclass, field

import redis
import torch
import torch.nn as nn
import torchvision.models as models
import wandb
from omegaconf import OmegaConf
from PIL import Image
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

from tinyexp import RedisCfgMixin, TinyExp, store_and_run_exp
from tinyexp.dataset.sampler import InfiniteSampler
from tinyexp.exceptions import UnknownAcceleratorTypeError


def transform_template_imagenet(
    is_train=True,
    resize_size=256,
    target_size=224,
    target_mean=None,
    target_std=None,
    interpolation=2,
):
    if target_mean is None:
        target_mean = (0.485, 0.456, 0.406)
    if target_std is None:
        target_std = (0.229, 0.224, 0.225)
    if is_train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(target_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=target_mean, std=target_std),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize(resize_size, interpolation=interpolation),
                transforms.CenterCrop(target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=target_mean, std=target_std),
            ]
        )


class LocalCachedImageFolder:
    def __init__(self, root: str, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.dataset = datasets.ImageFolder(root)

        # Local memory cache
        self.local_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def __getitem__(self, index):
        path, target = self.dataset.samples[index]

        # Use path as cache key
        if path in self.local_cache:
            self.cache_hits += 1
            file_data = self.local_cache[path]
        else:
            self.cache_misses += 1
            try:
                with open(path, "rb") as f:
                    file_data = f.read()
                self.local_cache[path] = file_data
            except Exception as e:
                print(f"Error reading file {path}: {e}")
                raise

        try:
            image = Image.open(io.BytesIO(file_data)).convert("RGB")
        except Exception as e:
            print(f"Error decoding image: {e}")
            raise

        # if (self.cache_hits + self.cache_misses) % 1000 == 0:
        #     print(f"Local Cache stats - hits: {self.cache_hits}, misses: {self.cache_misses}")

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target

    def __len__(self):
        return len(self.dataset)


class RedisCachedImageFolder:
    def __init__(self, redis_ports: list, root: str, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.dataset = datasets.ImageFolder(root)

        # Simplified redis connection
        self.redis_ports = redis_ports
        self.redis_clients = []
        self.num_shards = len(redis_ports)

        self._init_redis_connection()
        self.cache_misses = 0
        self.cache_hits = 0
        self.dataset_prefix = os.path.basename(root)[0]

    def _init_redis_connection(self):
        try:
            for redis_client in self.redis_clients:
                redis_client.close()

            # Simple Redis connection
            for port in self.redis_ports:
                redis_client = redis.StrictRedis(
                    host="localhost", port=port, decode_responses=False, socket_connect_timeout=5, socket_timeout=5
                )
                redis_client.ping()
                self.redis_clients.append(redis_client)

        except Exception as e:
            print(f"Redis connection failed: {e}")
            self.redis_clients = []

    def _safe_redis_get(self, key):
        if not self.redis_clients:
            return None
        redis_client = self.redis_clients[key % self.num_shards]
        try:
            return redis_client.get(key)
        except redis.exceptions.RedisError:
            return None

    def _safe_redis_set(self, key, value):
        if not self.redis_clients:
            return False
        redis_client = self.redis_clients[key % self.num_shards]
        try:
            return redis_client.set(key, value)
        except redis.exceptions.RedisError:
            return False

    def __getitem__(self, index):
        path, target = self.dataset.samples[index]
        # cache_key = f"{self.dataset_prefix}{index}"
        cache_key = index

        file_data = self._safe_redis_get(cache_key)
        if file_data is None:
            self.cache_misses += 1
            try:
                with open(path, "rb") as f:
                    file_data = f.read()
                self._safe_redis_set(cache_key, file_data)
            except Exception as e:
                print(f"Error reading file {path}: {e}")
                raise
        else:
            self.cache_hits += 1

        try:
            image = Image.open(io.BytesIO(file_data)).convert("RGB")
        except Exception as e:
            print(f"Error decoding image data for index {index}: {e}")
            with open(path, "rb") as f:
                file_data = f.read()
            image = Image.open(io.BytesIO(file_data)).convert("RGB")

        # if (self.cache_hits + self.cache_misses) % 1000 == 0:
        #     print(f"Redis Cache stats - hits: {self.cache_hits}, misses: {self.cache_misses}")

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target

    def __len__(self):
        return len(self.dataset)

    def __del__(self):
        """Destructor to ensure connections are properly closed"""
        for redis_client in self.redis_clients:
            with suppress(Exception):
                redis_client.close()


@dataclass(repr=False)
class ResNetExp(TinyExp, RedisCfgMixin):
    mode: str = "train"
    exp_class: str = f"{__name__}.ResNetExp"
    num_worker: int = torch.cuda.device_count()

    @dataclass
    class AcceleratorCfg:
        accelerator: str = "ddp"

        def build_accelerator(self):
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
    class ModuleCfg:
        def build_module(self):
            return models.__dict__["resnet50"](weights=None)

    module_cfg: ModuleCfg = field(default_factory=ModuleCfg)

    @dataclass
    class OptimizerCfg:
        lr_per_img: float = 0.1 / 256.0  # single image learning rate

        def build_optimizer(self, module, dataloader, accelerator):
            lr = self.lr_per_img * dataloader.batch_size * accelerator.world_size
            optimizer = SGD(
                module.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=1e-4,
                nesterov=False,
            )
            return optimizer

    optimizer_cfg: OptimizerCfg = field(default_factory=OptimizerCfg)

    @dataclass
    class LrSchedulerCfg:
        warmup_epoch: int = 0

        def build_lr_scheduler(self, optimizer):
            from torch.optim.lr_scheduler import LinearLR, SequentialLR

            main_scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

            if self.warmup_epoch > 0:
                warmup_factor: float = 0.001
                train_warmup_epoch: int = 3

                warmup_scheduler = LinearLR(
                    optimizer,
                    start_factor=warmup_factor,
                    end_factor=1.0,
                    total_iters=self.warmup_epoch,
                )
                scheduler = SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, main_scheduler],
                    milestones=[train_warmup_epoch],
                )
            else:
                scheduler = main_scheduler
            return scheduler

    lr_scheduler_cfg: LrSchedulerCfg = field(default_factory=LrSchedulerCfg)

    @dataclass
    class DataloaderCfg:
        data_root: str = os.environ.get("IMAGENET_HOME", "./data/imagenet/")
        train_batch_size_per_device: int = 32
        train_data_worker_per_gpu: int = 10
        val_batch_size_per_device: int = 50
        val_data_worker_per_gpu: int = 1
        seed: int = 42

        def build_train_dataloader(self, accelerator, redis_cache_cfg):
            transform = transform_template_imagenet(is_train=True)
            if redis_cache_cfg.redis_cache_enabled:
                ds_train = RedisCachedImageFolder(
                    redis_ports=redis_cache_cfg.redis_cache_shard_ports,
                    root=os.path.join(self.data_root, "train"),
                    transform=transform,
                )
            else:
                ds_train = datasets.ImageFolder(root=os.path.join(self.data_root, "train"), transform=transform)
            sampler = InfiniteSampler(len(ds_train), shuffle=True, seed=self.seed, accelerator=accelerator)
            train_kwargs = {
                "batch_size": self.train_batch_size_per_device,
                "num_workers": self.train_data_worker_per_gpu,
                "pin_memory": True,
                "sampler": sampler,
                "persistent_workers": True,  # Keep workers alive for multiple epochs
            }
            train_dataloader = torch.utils.data.DataLoader(ds_train, **train_kwargs)
            # from tinyexp.dataset.fake_dataloader import HoldOnesampleDataLoader
            # train_dataloader = HoldOnesampleDataLoader(train_dataloader)
            return train_dataloader

        def build_val_dataloader(self, accelerator):
            transform = transform_template_imagenet(is_train=False, interpolation=2)
            ds_val = LocalCachedImageFolder(root=os.path.join(self.data_root, "val"), transform=transform)
            # ds_val = datasets.ImageFolder(root=os.path.join(self.data_root, "val"), transform=transform)
            sampler = torch.utils.data.distributed.DistributedSampler(
                ds_val, num_replicas=accelerator.world_size, rank=accelerator.rank, shuffle=False
            )
            val_kwargs = {
                "batch_size": self.val_batch_size_per_device,
                "num_workers": self.val_data_worker_per_gpu,
                "pin_memory": True,
                "sampler": sampler,
                "persistent_workers": True,  # Keep workers alive for multiple epochs
            }
            val_dataloader = torch.utils.data.DataLoader(ds_val, **val_kwargs)
            return val_dataloader

    dataloader_cfg: DataloaderCfg = field(default_factory=DataloaderCfg)

    def run(self) -> None:
        accelerator = self.accelerator_cfg.build_accelerator()
        logger = self.logger_cfg.build_logger(
            save_dir=os.path.join(self.output_root, self.__class__.__name__), distributed_rank=accelerator.rank
        )
        cfg_dict = OmegaConf.to_container(OmegaConf.structured(self), resolve=True)
        del cfg_dict["hydra"]
        logger.info(f"-------- Configurations --------\n{OmegaConf.to_yaml(cfg_dict)}")

        self._train(accelerator=accelerator, logger=logger, cfg_dict=cfg_dict)

    def _evaluate(self, accelerator, logger, module_or_module_path, val_dataloader=None) -> None:
        if isinstance(module_or_module_path, str):
            module: nn.Module = self.module_cfg.build_module()
            module.load_state_dict(torch.load(module_or_module_path))
            module = accelerator.prepare_model(module)
        else:
            module = module_or_module_path

        if val_dataloader is None:
            val_dataloader = self.dataloader_cfg.build_val_dataloader(accelerator)

        module.eval()
        accurate = torch.tensor(0.0, device=accelerator.device)

        for step, batch in enumerate(val_dataloader):
            images, labels = (item.to(accelerator.device) for item in batch)
            with torch.no_grad():
                preds = module(images)
            predictions = preds.argmax(dim=-1)
            accurate_preds = predictions == labels
            accurate_preds_sum = accelerator.reduce_sum(accurate_preds.sum())
            accurate += accurate_preds_sum
            if step % 20 == 0:
                logger.info(f"Eval step {step}, accurate: {accurate.item()}")

        eval_metric = accurate.item() / len(val_dataloader.dataset)

        nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"{nowtime} --> eval_metric= {100 * eval_metric:.2f}%")

        if self.wandb_cfg.enable_wandb and accelerator.is_main_process:
            wandb.log({"val_metric": eval_metric})

    def _train(self, accelerator, logger, cfg_dict) -> None:
        train_dataloader = self.dataloader_cfg.build_train_dataloader(accelerator, self.redis_cache_cfg)
        val_dataloader = self.dataloader_cfg.build_val_dataloader(accelerator)
        ori_module = self.module_cfg.build_module()
        ori_optimizer = self.optimizer_cfg.build_optimizer(ori_module, train_dataloader, accelerator)
        module, optimizer = accelerator.prepare(ori_module, ori_optimizer)
        lr_scheduler = self.lr_scheduler_cfg.build_lr_scheduler(optimizer)

        if self.wandb_cfg.enable_wandb and accelerator.rank == 0:
            self.wandb_cfg.build_wandb(
                accelerator=accelerator, config=cfg_dict, project="Baselines", name=self.__class__.__name__
            )

        train_iter = iter(train_dataloader)
        global_step = 0

        for global_epoch in range(90):
            module.train()

            epoch_start_time = time.time()
            steps_per_epoch = len(train_dataloader)

            for step_in_epoch in range(len(train_dataloader)):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_dataloader)
                    batch = next(train_iter)

                images, labels = (item.to(accelerator.device) for item in batch)
                preds = module(images)
                loss = nn.CrossEntropyLoss()(preds, labels)

                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()
                global_step += 1

                if global_step % 20 == 0:
                    epoch_elapsed_time = time.time() - epoch_start_time
                    epoch_elapsed_str = f"{int(epoch_elapsed_time / 60):02d}:{int(epoch_elapsed_time % 60):02d}"

                    epoch_total_seconds = epoch_elapsed_time / ((step_in_epoch + 1) / steps_per_epoch)
                    epoch_total_str = f"{int(epoch_total_seconds / 60):02d}:{int(epoch_total_seconds % 60):02d}"

                    logger.info(
                        f"e:{global_epoch},{step_in_epoch + 1}/{steps_per_epoch}, "
                        f"loss:{loss.item():.4f}, lr:{optimizer.param_groups[0]['lr']:.4f}, "
                        f"elapsed:{epoch_elapsed_str}, total:{epoch_total_str}"
                    )

            lr_scheduler.step()
            self._evaluate(
                accelerator=accelerator, logger=logger, module_or_module_path=module, val_dataloader=val_dataloader
            )


if __name__ == "__main__":
    store_and_run_exp(ResNetExp)
