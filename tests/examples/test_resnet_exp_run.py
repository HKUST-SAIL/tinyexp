from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from omegaconf import ListConfig

from tinyexp.examples.resnet_exp import RedisCachedImageFolder, ResNetExp


def test_redis_cached_image_folder_uses_standalone_clients_for_localhost(tmp_path: Path, monkeypatch) -> None:
    train_root = tmp_path / "train" / "class0"
    train_root.mkdir(parents=True)
    (train_root / "sample.jpg").write_bytes(b"not-an-image")
    connections: list[tuple[str, int]] = []

    class FakeRedis:
        def __init__(self, *, host, port, **kwargs):
            connections.append((host, port))

        def ping(self):
            return True

    monkeypatch.setattr("tinyexp.examples.resnet_exp.redis.Redis", FakeRedis)

    RedisCachedImageFolder(redis_host="127.0.0.1", redis_ports=[7000, 7001], root=str(tmp_path / "train"))

    assert connections == [("127.0.0.1", 7000), ("127.0.0.1", 7001)]


def test_redis_cached_image_folder_uses_cluster_client_for_multi_machine_cache(tmp_path: Path, monkeypatch) -> None:
    train_root = tmp_path / "train" / "class0"
    train_root.mkdir(parents=True)
    (train_root / "sample.jpg").write_bytes(b"not-an-image")
    connections: list[tuple[str, int]] = []

    class FakeRedisCluster:
        def __init__(self, *, startup_nodes, **kwargs):
            connections.extend((node.host, node.port) for node in startup_nodes)

        def ping(self):
            return True

    monkeypatch.setattr("tinyexp.examples.resnet_exp.redis.RedisCluster", FakeRedisCluster)

    RedisCachedImageFolder(
        redis_host="10.0.0.1",
        redis_ports=[7000, 7001],
        root=str(tmp_path / "train"),
        redis_world_size=2,
    )

    assert connections == [("10.0.0.1", 7000), ("10.0.0.1", 7001)]


def test_redis_cached_image_folder_uses_standalone_clients_when_world_size_is_one(tmp_path: Path, monkeypatch) -> None:
    train_root = tmp_path / "train" / "class0"
    train_root.mkdir(parents=True)
    (train_root / "sample.jpg").write_bytes(b"not-an-image")
    connections: list[tuple[str, int]] = []

    class FakeRedis:
        def __init__(self, *, host, port, **kwargs):
            connections.append((host, port))

        def ping(self):
            return True

    monkeypatch.setattr("tinyexp.examples.resnet_exp.redis.Redis", FakeRedis)

    RedisCachedImageFolder(redis_host="10.0.0.1", redis_ports=[7000, 7001], root=str(tmp_path / "train"))

    assert connections == [("10.0.0.1", 7000), ("10.0.0.1", 7001)]


def test_redis_cached_image_folder_shards_standalone_keys_across_ports(tmp_path: Path, monkeypatch) -> None:
    train_root = tmp_path / "train" / "class0"
    train_root.mkdir(parents=True)
    (train_root / "sample0.jpg").write_bytes(b"not-an-image")
    (train_root / "sample1.jpg").write_bytes(b"not-an-image")
    calls: list[tuple[int, str, int]] = []

    class FakeRedis:
        def __init__(self, *, host, port, **kwargs):
            self.port = port

        def ping(self):
            return True

        def get(self, key):
            calls.append((self.port, "get", key))
            return None

        def set(self, key, value):
            calls.append((self.port, "set", key))
            return True

    monkeypatch.setattr("tinyexp.examples.resnet_exp.redis.Redis", FakeRedis)

    dataset = RedisCachedImageFolder(redis_host="127.0.0.1", redis_ports=[7000, 7001], root=str(tmp_path / "train"))
    dataset._safe_redis_get(0)
    dataset._safe_redis_get(1)
    dataset._safe_redis_set(0, b"0")
    dataset._safe_redis_set(1, b"1")

    assert calls == [
        (7000, "get", 0),
        (7001, "get", 1),
        (7000, "set", 0),
        (7001, "set", 1),
    ]


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
    redis_cache_cfg = SimpleNamespace(
        redis_cache_enabled=True,
        redis_cluster_host="10.0.0.1",
        redis_cluster_ports=ListConfig([7300, 7301, 7302]),
        redis_rendezvous_world_size=2,
    )

    dataloader_cfg.build_train_dataloader(accelerator=SimpleNamespace(), redis_cache_cfg=redis_cache_cfg)

    assert captured["redis_host"] == "10.0.0.1"
    assert captured["redis_ports"] == [7300, 7301, 7302]
    assert captured["redis_world_size"] == 2
    assert captured["root"] == "/imagenet/train"


def test_resnet_run_val_mode_requires_resume_from(tmp_path: Path, monkeypatch) -> None:
    exp = ResNetExp(output_root=str(tmp_path), exp_name="resnet_val", mode="val", resume_from="")

    dummy_accelerator = SimpleNamespace(rank=0, device="cpu", is_main_process=True)
    dummy_logger = SimpleNamespace(info=lambda *args, **kwargs: None)

    monkeypatch.setattr(exp.accelerator_cfg, "build_accelerator", lambda: dummy_accelerator)
    monkeypatch.setattr(exp.logger_cfg, "build_logger", lambda **kwargs: dummy_logger)

    with pytest.raises(ValueError, match="resume_from"):
        exp.run()


def test_resnet_run_val_mode_uses_checkpoint(tmp_path: Path, monkeypatch) -> None:
    exp_for_ckpt = ResNetExp(output_root=str(tmp_path), exp_name="resnet_val")
    checkpoint_path = exp_for_ckpt.checkpoint_cfg.save_checkpoint(
        run_dir=str(tmp_path / "resnet_val"),
        name="demo.ckpt",
        model=nn.Linear(2, 2),
        exp_name=exp_for_ckpt.exp_name,
        exp_class=exp_for_ckpt.exp_class,
    )

    exp = ResNetExp(
        output_root=str(tmp_path),
        exp_name="resnet_val",
        mode="val",
        resume_from=checkpoint_path,
    )

    dummy_accelerator = SimpleNamespace(rank=0, device="cpu", is_main_process=True)
    dummy_logger = SimpleNamespace(info=lambda *args, **kwargs: None)

    monkeypatch.setattr(exp.accelerator_cfg, "build_accelerator", lambda: dummy_accelerator)
    monkeypatch.setattr(exp.logger_cfg, "build_logger", lambda **kwargs: dummy_logger)

    called: dict[str, object] = {}

    def fake_evaluate(*, accelerator, logger, module_or_module_path, val_dataloader=None):
        called["accelerator"] = accelerator
        called["logger"] = logger
        called["module_or_module_path"] = module_or_module_path
        called["val_dataloader"] = val_dataloader
        return 0.5

    monkeypatch.setattr(exp, "_evaluate", fake_evaluate)

    exp.run()

    assert called["accelerator"] is dummy_accelerator
    assert called["logger"] is dummy_logger
    assert called["module_or_module_path"] == checkpoint_path
    assert called["val_dataloader"] is None


def test_resnet_train_saves_last_and_best_checkpoints(tmp_path: Path, monkeypatch) -> None:
    exp = ResNetExp(output_root=str(tmp_path), exp_name="resnet_train")

    class DummyAccelerator:
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

    train_batch = [(torch.randn(2, 2), torch.tensor([0, 1]))]
    val_batch = [(torch.randn(2, 2), torch.tensor([0, 1]))]

    monkeypatch.setattr(
        exp.dataloader_cfg,
        "build_train_dataloader",
        lambda accelerator, redis_cache_cfg: train_batch,
    )
    monkeypatch.setattr(exp.dataloader_cfg, "build_val_dataloader", lambda accelerator: val_batch)
    monkeypatch.setattr(exp.module_cfg, "build_module", lambda: nn.Linear(2, 2))
    monkeypatch.setattr(
        exp.optimizer_cfg,
        "build_optimizer",
        lambda module, dataloader, accelerator: torch.optim.SGD(module.parameters(), lr=0.1),
    )

    saved: list[dict[str, object]] = []

    def fake_save_checkpoint(**kwargs):
        saved.append(kwargs)
        if len(saved) >= 2:
            raise StopIteration
        return str(tmp_path / "resnet_train" / "last.ckpt")

    monkeypatch.setattr(exp.checkpoint_cfg, "save_checkpoint", fake_save_checkpoint)
    monkeypatch.setattr(exp, "_evaluate", lambda **kwargs: 0.5)

    with pytest.raises(StopIteration):
        exp._train(
            accelerator=DummyAccelerator(),
            logger=SimpleNamespace(info=lambda *args, **kwargs: None),
            cfg_dict={},
            run_dir=str(tmp_path / "resnet_train"),
        )

    assert saved[0]["run_dir"] == str(tmp_path / "resnet_train")
    assert saved[0]["name"] == exp.checkpoint_cfg.last_ckpt_name
    assert saved[0]["epoch"] == 0
    assert saved[0]["global_step"] == 1
    assert saved[0]["best_metric"] is None
    assert saved[1]["name"] == exp.checkpoint_cfg.best_ckpt_name
    assert saved[1]["best_metric"] == 0.5


def test_resnet_train_stops_at_max_train_epochs(tmp_path: Path, monkeypatch) -> None:
    exp = ResNetExp(output_root=str(tmp_path), exp_name="resnet_train", max_train_epochs=2)

    class DummyAccelerator:
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

    train_batch = [(torch.randn(2, 2), torch.tensor([0, 1]))]
    val_batch = [(torch.randn(2, 2), torch.tensor([0, 1]))]

    monkeypatch.setattr(
        exp.dataloader_cfg,
        "build_train_dataloader",
        lambda accelerator, redis_cache_cfg: train_batch,
    )
    monkeypatch.setattr(exp.dataloader_cfg, "build_val_dataloader", lambda accelerator: val_batch)
    monkeypatch.setattr(exp.module_cfg, "build_module", lambda: nn.Linear(2, 2))
    monkeypatch.setattr(
        exp.optimizer_cfg,
        "build_optimizer",
        lambda module, dataloader, accelerator: torch.optim.SGD(module.parameters(), lr=0.1),
    )
    monkeypatch.setattr(exp, "_evaluate", lambda **kwargs: 0.5)
    saved: list[dict[str, object]] = []
    monkeypatch.setattr(exp.checkpoint_cfg, "save_checkpoint", lambda **kwargs: saved.append(kwargs))

    exp._train(
        accelerator=DummyAccelerator(),
        logger=SimpleNamespace(info=lambda *args, **kwargs: None),
        cfg_dict={},
        run_dir=str(tmp_path / "resnet_train"),
    )

    assert [item["epoch"] for item in saved] == [0, 0, 1]
    assert [item["global_step"] for item in saved] == [1, 1, 2]


def test_resnet_train_stops_at_max_train_steps(tmp_path: Path, monkeypatch) -> None:
    exp = ResNetExp(output_root=str(tmp_path), exp_name="resnet_train", max_train_steps=1)

    class DummyAccelerator:
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

    train_batch = [(torch.randn(2, 2), torch.tensor([0, 1]))]

    monkeypatch.setattr(
        exp.dataloader_cfg,
        "build_train_dataloader",
        lambda accelerator, redis_cache_cfg: train_batch,
    )
    monkeypatch.setattr(exp.dataloader_cfg, "build_val_dataloader", lambda accelerator: [])
    monkeypatch.setattr(exp.module_cfg, "build_module", lambda: nn.Linear(2, 2))
    monkeypatch.setattr(
        exp.optimizer_cfg,
        "build_optimizer",
        lambda module, dataloader, accelerator: torch.optim.SGD(module.parameters(), lr=0.1),
    )
    monkeypatch.setattr(exp, "_evaluate", lambda **kwargs: pytest.fail("eval should not run"))
    monkeypatch.setattr(
        exp.checkpoint_cfg, "save_checkpoint", lambda **kwargs: pytest.fail("checkpoint should not save")
    )

    exp._train(
        accelerator=DummyAccelerator(),
        logger=SimpleNamespace(info=lambda *args, **kwargs: None),
        cfg_dict={},
        run_dir=str(tmp_path / "resnet_train"),
    )


def test_resnet_train_resume_loads_checkpoint_state(tmp_path: Path, monkeypatch) -> None:
    exp = ResNetExp(output_root=str(tmp_path), exp_name="resnet_train", resume_from="resume.ckpt")

    class DummyAccelerator:
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

    train_batch = [(torch.randn(2, 2), torch.tensor([0, 1]))]
    val_batch = [(torch.randn(2, 2), torch.tensor([0, 1]))]

    monkeypatch.setattr(
        exp.dataloader_cfg,
        "build_train_dataloader",
        lambda accelerator, redis_cache_cfg: train_batch,
    )
    monkeypatch.setattr(exp.dataloader_cfg, "build_val_dataloader", lambda accelerator: val_batch)
    monkeypatch.setattr(exp.module_cfg, "build_module", lambda: nn.Linear(2, 2))
    monkeypatch.setattr(
        exp.optimizer_cfg,
        "build_optimizer",
        lambda module, dataloader, accelerator: torch.optim.SGD(module.parameters(), lr=0.1),
    )

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
        exp._train(
            accelerator=DummyAccelerator(),
            logger=SimpleNamespace(info=lambda *args, **kwargs: None),
            cfg_dict={},
            run_dir=str(tmp_path / "resnet_train"),
        )

    assert load_calls[0]["path"] == "resume.ckpt"
    assert load_calls[0]["map_location"] == "cpu"
    assert saved[0]["epoch"] == 5
    assert saved[0]["global_step"] == 18
    assert saved[0]["best_metric"] == 0.7
