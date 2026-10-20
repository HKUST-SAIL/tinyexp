from __future__ import annotations

import contextlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from omegaconf import ListConfig

from tinyexp.examples.resnet_exp import RedisCachedImageFolder, ResNetExp


class _TrainAccelerator:
    rank = 0
    device = "cpu"
    is_main_process = True
    world_size = 1

    def prepare(self, module, optimizer):
        return module, optimizer

    def unwrap_model(self, module):
        return module

    def backward(self, loss):
        loss.backward()

    def autocast(self):
        return contextlib.nullcontext()

    def optimizer_step(self, optimizer):
        return optimizer.step()


def _batches() -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [(torch.randn(2, 2), torch.tensor([0, 1]))]


def _train(monkeypatch: pytest.MonkeyPatch, exp: ResNetExp, train_dataloader, val_dataloader, run_dir: str) -> None:  # type: ignore[no-untyped-def]
    """Run ``exp._train`` on a 2-class linear model with the given dataloaders."""
    monkeypatch.setattr(exp.dataloader_cfg, "build_train_dataloader", lambda accelerator, redis_cfg: train_dataloader)
    monkeypatch.setattr(exp.dataloader_cfg, "build_val_dataloader", lambda accelerator: val_dataloader)
    monkeypatch.setattr(exp.module_cfg, "build_module", lambda: nn.Linear(2, 2))
    monkeypatch.setattr(
        exp.optimizer_cfg,
        "build_optimizer",
        lambda module, dataloader, accelerator: torch.optim.SGD(module.parameters(), lr=0.1),
    )
    exp._train(
        accelerator=_TrainAccelerator(),
        logger=SimpleNamespace(info=lambda *args, **kwargs: None),
        cfg_dict={},
        run_dir=run_dir,
    )


def test_accelerator_cfg_builds_hf_and_maps_none_to_accelerates_no(monkeypatch) -> None:
    pytest.importorskip("accelerate")
    # accelerate keeps process-global state; swapping mixed_precision across
    # constructions raises, so the mapping is asserted through a fake class.
    import tinyexp.tiny_engine.accelerator as accelerator_module

    created: list[dict] = []

    class FakeHFAccelerator:
        def __init__(self, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr(accelerator_module, "HFAccelerator", FakeHFAccelerator)

    ResNetExp.AcceleratorCfg(accelerator="hf").build_accelerator()
    assert created == [{"mixed_precision": "no"}]

    ResNetExp.AcceleratorCfg(accelerator="hf", mixed_precision="bf16").build_accelerator()
    assert created == [{"mixed_precision": "no"}, {"mixed_precision": "bf16"}]


def test_resnet_warmup_milestone_uses_configured_epoch() -> None:
    module = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    scheduler = ResNetExp.LrSchedulerCfg(warmup_epoch=5).build_lr_scheduler(optimizer)

    assert scheduler._milestones == [5]


def test_redis_cached_image_folder_passes_world_size_to_the_client(tmp_path: Path, monkeypatch) -> None:
    """A multi-machine cache (world size > 1) gets one cluster client; see tests/utils for the standalone mode."""
    train_root = tmp_path / "train" / "class0"
    train_root.mkdir(parents=True)
    (train_root / "sample.jpg").write_bytes(b"not-an-image")
    connections: list[tuple[str, int]] = []

    class FakeRedisCluster:
        def __init__(self, *, startup_nodes, **kwargs):
            connections.extend((node.host, node.port) for node in startup_nodes)

        def ping(self):
            return True

    monkeypatch.setattr("tinyexp.utils.redis_utils.redis.RedisCluster", FakeRedisCluster)

    RedisCachedImageFolder(
        redis_host="10.0.0.1",
        redis_ports=[7000, 7001],
        root=str(tmp_path / "train"),
        redis_world_size=2,
    )

    assert connections == [("10.0.0.1", 7000), ("10.0.0.1", 7001)]


def test_resnet_dataloader_passes_complete_redis_cfg_to_cached_folder(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeRedisCachedImageFolder:
        def __init__(self, *, redis_host, redis_ports, root, transform, redis_world_size):
            captured["redis_host"] = redis_host
            captured["redis_ports"] = redis_ports
            captured["root"] = root
            captured["transform"] = transform
            captured["redis_world_size"] = redis_world_size

        def __len__(self):
            return 1

    monkeypatch.setattr("tinyexp.examples.resnet_exp.RedisCachedImageFolder", FakeRedisCachedImageFolder)
    monkeypatch.setattr(
        "tinyexp.examples.resnet_exp.transform_template_imagenet",
        lambda is_train: "transform",
    )
    monkeypatch.setattr("tinyexp.examples.resnet_exp.InfiniteSampler", lambda *args, **kwargs: "sampler")
    monkeypatch.setattr(
        "tinyexp.examples.resnet_exp.torch.utils.data.DataLoader",
        lambda dataset, **kwargs: dataset,
    )

    dataloader_cfg = ResNetExp.DataloaderCfg(data_root="/imagenet")
    redis_cfg = SimpleNamespace(
        redis_cache_enabled=True,
        redis_cluster_host="10.0.0.1",
        redis_cluster_ports=ListConfig([7300, 7301, 7302]),
        redis_rendezvous_world_size=2,
    )

    dataloader_cfg.build_train_dataloader(accelerator=SimpleNamespace(), redis_cfg=redis_cfg)

    assert captured["redis_host"] == "10.0.0.1"
    assert captured["redis_ports"] == [7300, 7301, 7302]
    assert captured["redis_world_size"] == 2
    assert captured["root"] == "/imagenet/train"


@pytest.mark.parametrize("num_workers", [0, 2])
def test_resnet_dataloaders_match_persistent_workers_to_num_workers(
    monkeypatch,
    num_workers: int,
) -> None:
    class FakeDataset:
        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: int) -> tuple[int, int]:
            return index, index

    monkeypatch.setattr(
        "tinyexp.examples.resnet_exp.datasets.ImageFolder",
        lambda root, transform=None: FakeDataset(),
    )
    monkeypatch.setattr(
        "tinyexp.examples.resnet_exp.LocalCachedImageFolder",
        lambda root, transform=None: FakeDataset(),
    )
    monkeypatch.setattr(
        "tinyexp.examples.resnet_exp.transform_template_imagenet",
        lambda **kwargs: None,
    )

    dataloader_cfg = ResNetExp.DataloaderCfg(
        data_root="/imagenet",
        train_data_worker_per_gpu=num_workers,
        val_data_worker_per_gpu=num_workers,
    )
    accelerator = SimpleNamespace(rank=0, world_size=1)
    redis_cfg = SimpleNamespace(redis_cache_enabled=False)

    train_dataloader = dataloader_cfg.build_train_dataloader(accelerator, redis_cfg)
    val_dataloader = dataloader_cfg.build_val_dataloader(accelerator)

    expected = num_workers > 0
    assert train_dataloader.persistent_workers is expected
    assert val_dataloader.persistent_workers is expected


def test_resnet_train_saves_last_and_best_checkpoints_until_max_train_epochs(tmp_path: Path, monkeypatch) -> None:
    exp = ResNetExp(output_root=str(tmp_path), exp_name="resnet_train", max_train_epochs=2)
    monkeypatch.setattr(exp, "_evaluate", lambda **kwargs: 0.5)
    saved: list[dict[str, object]] = []
    monkeypatch.setattr(exp.checkpoint_cfg, "save_checkpoint", lambda **kwargs: saved.append(kwargs))
    run_dir = str(tmp_path / "resnet_train")

    _train(monkeypatch, exp, _batches(), _batches(), run_dir)

    last, best = exp.checkpoint_cfg.last_ckpt_name, exp.checkpoint_cfg.best_ckpt_name
    # Epoch 0 sets the best metric (last + best); an equal metric in epoch 1 only saves last.
    assert [item["name"] for item in saved] == [last, best, last]
    assert [item["epoch"] for item in saved] == [0, 0, 1]
    assert [item["global_step"] for item in saved] == [1, 1, 2]
    assert {item["run_dir"] for item in saved} == {run_dir}
    assert {item["best_metric"] for item in saved} == {0.5}


def test_resnet_train_positions_sampler_once_before_creating_iterator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exp = ResNetExp(output_root=str(tmp_path), exp_name="resnet_train", max_train_epochs=2)
    events: list[object] = []

    class DummySampler:
        def set_epoch(self, epoch: int) -> None:
            events.append(("set_epoch", epoch))

    class DummyDataLoader:
        batch_size = 2
        sampler = DummySampler()

        def __len__(self) -> int:
            return 1

        def __iter__(self):
            events.append("iter")
            while True:
                yield torch.randn(2, 2), torch.tensor([0, 1])

    monkeypatch.setattr(exp, "_evaluate", lambda **kwargs: 0.5)
    monkeypatch.setattr(exp.checkpoint_cfg, "save_checkpoint", lambda **kwargs: None)

    _train(monkeypatch, exp, DummyDataLoader(), [], str(tmp_path / "resnet_train"))

    assert events == [("set_epoch", 0), "iter"]


def test_resnet_train_stops_at_max_train_steps(tmp_path: Path, monkeypatch) -> None:
    exp = ResNetExp(output_root=str(tmp_path), exp_name="resnet_train", max_train_steps=1)
    monkeypatch.setattr(exp, "_evaluate", lambda **kwargs: pytest.fail("eval should not run"))
    monkeypatch.setattr(exp.checkpoint_cfg, "save_checkpoint", lambda **kwargs: pytest.fail("should not save"))

    _train(monkeypatch, exp, _batches(), [], str(tmp_path / "resnet_train"))


def test_resnet_train_resume_loads_checkpoint_state(tmp_path: Path, monkeypatch) -> None:
    exp = ResNetExp(output_root=str(tmp_path), exp_name="resnet_train", resume_from="resume.ckpt")
    load_calls: list[dict[str, object]] = []
    saved: list[dict[str, object]] = []

    def fake_load_checkpoint(path, **kwargs):
        load_calls.append({"path": path, **kwargs})
        return {"epoch": 4, "global_step": 17, "best_metric": 0.7}

    def fake_save_checkpoint(**kwargs):
        saved.append(kwargs)
        raise StopIteration

    monkeypatch.setattr(exp.checkpoint_cfg, "load_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(exp.checkpoint_cfg, "save_checkpoint", fake_save_checkpoint)
    monkeypatch.setattr(exp, "_evaluate", lambda **kwargs: 0.5)

    with pytest.raises(StopIteration):
        _train(monkeypatch, exp, _batches(), _batches(), str(tmp_path / "resnet_train"))

    assert load_calls[0]["path"] == "resume.ckpt"
    assert load_calls[0]["map_location"] == "cpu"
    assert saved[0]["epoch"] == 5
    assert saved[0]["global_step"] == 18
    assert saved[0]["best_metric"] == 0.7


def test_resnet_val_dataloader_partitions_without_padding(tmp_path: Path, monkeypatch) -> None:
    dataset = torch.utils.data.TensorDataset(torch.arange(10), torch.arange(10))
    monkeypatch.setattr(
        "tinyexp.examples.resnet_exp.LocalCachedImageFolder",
        lambda **kwargs: dataset,
    )

    cfg = ResNetExp.DataloaderCfg(
        data_root=str(tmp_path),
        val_batch_size_per_device=4,
        val_data_worker_per_gpu=1,
    )
    partitions = []
    for rank in range(3):
        dataloader = cfg.build_val_dataloader(SimpleNamespace(rank=rank, world_size=3))
        assert dataloader.drop_last is False
        partitions.append(list(dataloader.dataset.indices))

    assert partitions == [[0, 3, 6, 9], [1, 4, 7], [2, 5, 8]]
