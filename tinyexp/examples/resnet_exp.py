import datetime
import io
import os
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torchvision.models as models
import wandb
from PIL import Image
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

from tinyexp import TinyExp, store_and_run_exp
from tinyexp.dataset.sampler import InfiniteSampler
from tinyexp.exceptions import UnknownAcceleratorTypeError
from tinyexp.exp_mixins import CheckpointCfgMixin, LoggerCfgMixin, RayCfgMixin, RedisCfgMixin, WandbCfgMixin
from tinyexp.tiny_engine.accelerator import AcceleratorProtocol


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
    def __init__(
        self,
        redis_host: str,
        redis_ports: list[int],
        root: str,
        transform=None,
        target_transform=None,
        redis_world_size: int = 1,
    ):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.dataset = datasets.ImageFolder(root)

        self.cache_misses = 0
        self.cache_hits = 0
        self.dataset_prefix = os.path.basename(root)[0]
        from tinyexp.utils.redis_utils import RedisClientManager

        self.redis_client_manager = RedisClientManager(redis_host, redis_ports, redis_world_size)

    def __getitem__(self, index):
        path, target = self.dataset.samples[index]
        # cache_key = f"{self.dataset_prefix}{index}"
        cache_key = index

        file_data = self.redis_client_manager.safe_get(cache_key)
        if file_data is None:
            self.cache_misses += 1
            try:
                with open(path, "rb") as f:
                    file_data = f.read()
                self.redis_client_manager.safe_set(cache_key, file_data)
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


@dataclass(repr=False)
class ResNetExp(TinyExp, RayCfgMixin, RedisCfgMixin, CheckpointCfgMixin, WandbCfgMixin, LoggerCfgMixin):
    mode: str = "train"
    launcher: str = "ray"
    max_train_epochs: int = 90
    max_train_steps: int = -1

    @dataclass
    class RayCfg(RayCfgMixin.RayCfg):
        ray_num_cpus_per_worker: int = 12  # Main process plus train/validation dataloader workers.

    ray_cfg: RayCfg = field(default_factory=RayCfg)

    @dataclass
    class AcceleratorCfg:
        accelerator: str = "ddp"

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

                warmup_scheduler = LinearLR(
                    optimizer,
                    start_factor=warmup_factor,
                    end_factor=1.0,
                    total_iters=self.warmup_epoch,
                )
                scheduler = SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, main_scheduler],
                    milestones=[self.warmup_epoch],
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

        def build_train_dataloader(self, accelerator, redis_cfg):
            transform = transform_template_imagenet(is_train=True)
            if redis_cfg.redis_cache_enabled:
                ds_train = RedisCachedImageFolder(
                    redis_host=redis_cfg.redis_cluster_host,
                    redis_ports=list(redis_cfg.redis_cluster_ports),
                    root=os.path.join(self.data_root, "train"),
                    transform=transform,
                    redis_world_size=int(redis_cfg.redis_rendezvous_world_size),
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
            ds_val = torch.utils.data.Subset(
                ds_val,
                range(accelerator.rank, len(ds_val), accelerator.world_size),
            )
            val_kwargs = {
                "batch_size": self.val_batch_size_per_device,
                "num_workers": self.val_data_worker_per_gpu,
                "pin_memory": True,
                "persistent_workers": True,
            }
            val_dataloader = torch.utils.data.DataLoader(ds_val, **val_kwargs)
            return val_dataloader

    dataloader_cfg: DataloaderCfg = field(default_factory=DataloaderCfg)

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

        accelerator.destroy()

    def _evaluate(self, accelerator, logger, module_or_module_path, val_dataloader=None) -> None:
        if isinstance(module_or_module_path, str):
            module: nn.Module = self.module_cfg.build_module()
            self.checkpoint_cfg.load_checkpoint(
                module_or_module_path,
                model=module,
                map_location=accelerator.device,
            )
            module = accelerator.prepare_model(module)
        else:
            module = module_or_module_path

        if val_dataloader is None:
            val_dataloader = self.dataloader_cfg.build_val_dataloader(accelerator)

        # Ranks may have different eval batch counts when the dataset is not divisible.
        module = accelerator.unwrap_model(module)
        module.eval()
        accurate = torch.tensor(0, dtype=torch.long, device=accelerator.device)
        seen = torch.tensor(0, dtype=torch.long, device=accelerator.device)

        for step, batch in enumerate(val_dataloader):
            images, labels = (item.to(accelerator.device) for item in batch)
            with torch.no_grad():
                preds = module(images)
            predictions = preds.argmax(dim=-1)
            accurate += (predictions == labels).sum()
            seen += labels.numel()
            if step % 20 == 0:
                logger.info(f"Eval step {step}, accurate: {accurate.item()}")

        global_accurate = accelerator.reduce_sum(accurate)
        global_seen = accelerator.reduce_sum(seen)
        eval_metric = global_accurate.item() / global_seen.item() if global_seen.item() else 0.0

        nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"{nowtime} --> eval_metric= {100 * eval_metric:.2f}%")

        if self.wandb_cfg.enable_wandb and accelerator.is_main_process:
            wandb.log({"val_metric": eval_metric})

        return eval_metric

    def _train(self, accelerator, logger, cfg_dict, run_dir: str) -> None:  # noqa: C901
        train_dataloader = self.dataloader_cfg.build_train_dataloader(accelerator, self.redis_cfg)
        val_dataloader = self.dataloader_cfg.build_val_dataloader(accelerator)
        ori_module = self.module_cfg.build_module()
        # Keep the optimizer attached to the final device-side parameters.
        ori_module.to(accelerator.device)
        ori_optimizer = self.optimizer_cfg.build_optimizer(ori_module, train_dataloader, accelerator)
        module, optimizer = accelerator.prepare(ori_module, ori_optimizer)
        lr_scheduler = self.lr_scheduler_cfg.build_lr_scheduler(optimizer)
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
            rng_state = (checkpoint.get("extra_state") or {}).get("rng_state")
            if rng_state is not None and getattr(accelerator, "world_size", 1) == 1:
                self.checkpoint_cfg.restore_rng_state(rng_state)

        if self.wandb_cfg.enable_wandb and accelerator.rank == 0:
            self.wandb_cfg.build_wandb(
                accelerator=accelerator,
                config=cfg_dict,
                project="Baselines",
                name=self.__class__.__name__,
            )

        train_sampler = getattr(train_dataloader, "sampler", None)
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(start_epoch)
        train_iter = iter(train_dataloader)

        for global_epoch in range(start_epoch, self.max_train_epochs):
            if global_epoch != start_epoch and train_sampler is not None and hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(global_epoch)
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

                if 0 < self.max_train_steps <= global_step:
                    return

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
            eval_metric = self._evaluate(
                accelerator=accelerator,
                logger=logger,
                module_or_module_path=module,
                val_dataloader=val_dataloader,
            )
            is_best = best_metric is None or eval_metric > best_metric
            if is_best:
                best_metric = eval_metric
            if accelerator.is_main_process:
                checkpoint_extra_state = None
                if getattr(accelerator, "world_size", 1) == 1:
                    checkpoint_extra_state = {"rng_state": self.checkpoint_cfg.capture_rng_state()}
                self.checkpoint_cfg.save_checkpoint(
                    run_dir=run_dir,
                    name=self.checkpoint_cfg.last_ckpt_name,
                    model=accelerator.unwrap_model(module),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    epoch=global_epoch,
                    global_step=global_step,
                    best_metric=best_metric,
                    exp_name=self.exp_name,
                    exp_class=self.exp_class,
                    extra_state=checkpoint_extra_state,
                )
                if is_best:
                    self.checkpoint_cfg.save_checkpoint(
                        run_dir=run_dir,
                        name=self.checkpoint_cfg.best_ckpt_name,
                        model=accelerator.unwrap_model(module),
                        optimizer=optimizer,
                        scheduler=lr_scheduler,
                        epoch=global_epoch,
                        global_step=global_step,
                        best_metric=best_metric,
                        exp_name=self.exp_name,
                        exp_class=self.exp_class,
                        extra_state=checkpoint_extra_state,
                    )


if __name__ == "__main__":
    store_and_run_exp(ResNetExp)
