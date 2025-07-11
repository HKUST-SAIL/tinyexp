import datetime
import io
import os
import time
from dataclasses import dataclass

import redis
import torch
import torch.nn as nn
import torchvision.models as models
import wandb
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from omegaconf.listconfig import ListConfig
from PIL import Image
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

from tinyexp import TinyCfg, TinyExp, simple_ray_launch_exp
from tinyexp.dataset.sampler import InfiniteSampler
from tinyexp.tiny_engine.accelerator import CPUAccelerator, DDPAccelerator


def transform_template_imagenet(
    is_train=True,
    resize_size=256,
    target_size=224,
    target_mean=[0.485, 0.456, 0.406],
    target_std=[0.229, 0.224, 0.225],
    interpolation=2,
):
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
        redis_client = self.redis_clients[key % self.num_shards]
        try:
            return redis_client.get(key)
        except:
            return None

    def _safe_redis_set(self, key, value):
        redis_client = self.redis_clients[key % self.num_shards]
        try:
            return redis_client.set(key, value)
        except:
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
            try:
                redis_client.close()
            except:
                pass


class ResNetExp(TinyExp):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.module = self._configure_module()
        self.optimizer = self._configure_optimizer()
        self.lr_scheduler = self._configure_lr_scheduler()
        self.train_dataloader = self._configure_train_dataloader()
        self.val_dataloader = self._configure_val_dataloader()

    def _configure_accelerator(self):
        if self.cfg.accelerator == "cpu":
            accelerator = CPUAccelerator()
        elif self.cfg.accelerator == "ddp":
            accelerator = DDPAccelerator()
        else:
            raise ValueError(f"Unknown accelerator type: {self.cfg.accelerator}")
        return accelerator

    def _configure_module(self):
        return models.__dict__["resnet50"](weights=None)

    def _configure_optimizer(self):
        cfg = self.cfg
        lr = cfg.train_lr_per_img * cfg.train_batch_size_per_device * self.accelerator.world_size
        optimizer = SGD(
            self.module.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=1e-4,
            nesterov=False,
        )
        return optimizer

    def _configure_lr_scheduler(self):
        from torch.optim.lr_scheduler import LinearLR, SequentialLR

        main_scheduler = StepLR(self.optimizer, step_size=30, gamma=0.1)

        if self.cfg.train_warmup_epoch > 0:
            warmup_factor: float = 0.001

            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=warmup_factor,
                end_factor=1.0,
                total_iters=self.cfg.train_warmup_epoch,
            )
            scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[self.cfg.train_warmup_epoch],
            )
        else:
            scheduler = main_scheduler
        return scheduler

    def _configure_train_dataloader(self):
        cfg = self.cfg
        transform = transform_template_imagenet(is_train=True)
        if cfg.redis_cache_enabled:
            ds_train = RedisCachedImageFolder(
                redis_ports=cfg.redis_cache_shard_ports, root=os.path.join(cfg.data_root, "train"), transform=transform
            )
        else:
            ds_train = datasets.ImageFolder(root=os.path.join(cfg.data_root, "train"), transform=transform)
        sampler = InfiniteSampler(len(ds_train), shuffle=True, seed=cfg.seed, accelerator=self.accelerator)
        train_kwargs = {
            "batch_size": cfg.train_batch_size_per_device,
            "num_workers": cfg.train_data_worker_per_gpu,
            "pin_memory": True,
            "sampler": sampler,
            "persistent_workers": True,  # Keep workers alive for multiple epochs
        }
        train_dataloader = torch.utils.data.DataLoader(ds_train, **train_kwargs)
        # from tinyexp.dataset.fake_dataloader import HoldOnesampleDataLoader
        # train_dataloader = HoldOnesampleDataLoader(train_dataloader)
        return train_dataloader

    def _configure_val_dataloader(self):
        transform = transform_template_imagenet(is_train=False, interpolation=2)
        if self.cfg.redis_cache_enabled:
            ds_val = LocalCachedImageFolder(root=os.path.join(self.cfg.data_root, "val"), transform=transform)
        else:
            ds_val = datasets.ImageFolder(root=os.path.join(self.cfg.data_root, "val"), transform=transform)
        sampler = torch.utils.data.distributed.DistributedSampler(
            ds_val, num_replicas=self.accelerator.world_size, rank=self.accelerator.rank, shuffle=False
        )
        val_kwargs = {
            "batch_size": self.cfg.val_batch_size_per_device,
            "num_workers": self.cfg.val_data_worker_per_gpu,
            "pin_memory": True,
            "sampler": sampler,
            "persistent_workers": True,  # Keep workers alive for multiple epochs
        }
        val_dataloader = torch.utils.data.DataLoader(ds_val, **val_kwargs)
        return val_dataloader

    def run(self) -> None:
        accelerator = self.accelerator
        module, optimizer = accelerator.prepare(self.module, self.optimizer)
        if self.cfg.train_enable_wandb and accelerator.rank == 0:
            wandb.init(config=self.cfg, project="Baselines", name=self.exp_name)

        train_dataloader = self.train_dataloader
        train_iter = iter(train_dataloader)

        for _ in range(self.cfg.train_max_epoch):
            module.train()

            epoch_start_time = time.time()
            steps_per_epoch = len(train_dataloader)

            for step_in_epoch in range(len(train_dataloader)):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_dataloader)
                    batch = next(train_iter)

                images, labels = (_.to(accelerator.device) for _ in batch)
                preds = module(images)
                loss = nn.CrossEntropyLoss()(preds, labels)

                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()
                self.global_step += 1

                if self.global_step % 20 == 0:
                    epoch_elapsed_time = time.time() - epoch_start_time
                    epoch_elapsed_str = f"{int(epoch_elapsed_time / 60):02d}:{int(epoch_elapsed_time % 60):02d}"

                    epoch_total_seconds = epoch_elapsed_time / ((step_in_epoch + 1) / steps_per_epoch)
                    epoch_total_str = f"{int(epoch_total_seconds / 60):02d}:{int(epoch_total_seconds % 60):02d}"

                    self.logger.info(
                        f"e:{self.global_epoch},{step_in_epoch + 1}/{steps_per_epoch}, "
                        f"loss:{loss.item():.4f}, lr:{optimizer.param_groups[0]['lr']:.4f}, "
                        f"elapsed:{epoch_elapsed_str}, total:{epoch_total_str}"
                    )

            self.lr_scheduler.step()
            self.global_epoch += 1
            self.eval(module, self.val_dataloader)

    def eval(self, module_or_module_path, val_dataloader) -> None:
        if isinstance(module_or_module_path, str):
            module = models.__dict__["resnet50"](weights=None)
            module.load_state_dict(torch.load(module_or_module_path))
            module = self.accelerator.prepare(module)
        else:
            module = module_or_module_path

        module.eval()
        accurate = 0.0

        for _, batch in enumerate(val_dataloader):
            images, labels = (_.to(self.accelerator.device) for _ in batch)
            with torch.no_grad():
                preds = module(images)
            predictions = preds.argmax(dim=-1)
            accurate_preds = predictions == labels
            accurate_preds_sum = self.accelerator.reduce_sum(accurate_preds.sum())
            accurate += accurate_preds_sum

        eval_metric = accurate.item() / len(val_dataloader.dataset)

        nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logger.info(f"{nowtime} --> eval_metric= {100 * eval_metric:.2f}%")

        if self.cfg.train_enable_wandb and self.accelerator.is_main_process:
            wandb.log({"val_metric": eval_metric})


@dataclass
class Config(TinyCfg):
    # for exp actor
    exp_class: str = f"{ResNetExp.__module__}.{ResNetExp.__name__}"
    data_root: str = os.environ.get("IMAGENET_HOME", "./data/imagenet/")

    accelerator: str = "ddp"  # "cpu", "ddp"

    # bellowing config specific the cpu and gpu resources for the experiment
    # total_cpu = num_gpus * (train_data_worker_per_gpu + val_data_worker_per_gpu + 1) + redis_cluster_manager_cpus
    num_gpus: int = torch.cuda.device_count()
    train_data_worker_per_gpu: int = 6
    val_data_worker_per_gpu: int = 1
    redis_cluster_manager_cpus: int = 10  # Number of CPUs allocated for Redis cluster manager

    train_lr_per_img: float = 0.1 / 256.0  # single image learning rate
    train_batch_size_per_device: int = 32
    val_batch_size_per_device: int = 50
    train_max_epoch: int = 90
    train_warmup_epoch: int = 0
    train_enable_wandb: bool = False
    launch: str = "ray"  # "ray", "local"
    seed: int = 42

    # for redis actor
    redis_cache_enabled: bool = True  # Whether to use Redis cache for images
    redis_cache_max_memory: int = 160  # Maximum memory is 160GB, according to the ImageNet dataset size
    redis_cache_shard_ports: ListConfig = ListConfig(
        [
            7000,
            7001,
            7002,
            7003,
            7004,
        ]
    )  # List of Redis cache shard used ports


if __name__ == "__main__":
    ConfigStore.instance().store(name="cfg", node=Config)
    simple_ray_launch_exp()
