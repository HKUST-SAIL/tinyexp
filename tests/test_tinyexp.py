from dataclasses import dataclass, field

import pytest
from omegaconf import OmegaConf

from tinyexp import TinyExp, store_and_run_exp
from tinyexp.exceptions import UnknownConfigurationKeyError


@dataclass
class _CfgExp(TinyExp):
    @dataclass
    class SubCfg:
        a: int = 1

    sub_cfg: SubCfg = field(default_factory=SubCfg)
    b: int = 2


@dataclass
class _StoreAndRunExpCfg(TinyExp):
    check_exp_class: str = f"{__name__}._StoreAndRunExpCfg"


def test_tiny_exp_instantiation():
    class MyExperiment(TinyExp):
        pass

    _ = MyExperiment()


def test_set_cfg_overrides_nested(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RANK", "1")  # avoid noisy stdout prints during tests
    exp = _CfgExp()

    cfg = OmegaConf.create({"sub_cfg": {"a": 3}, "b": 4})
    exp.set_cfg(cfg)

    assert exp.sub_cfg.a == 3
    assert exp.b == 4


def test_set_cfg_unknown_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RANK", "1")
    exp = _CfgExp()

    cfg = OmegaConf.create({"no_such_key": 1})
    with pytest.raises(UnknownConfigurationKeyError):
        exp.set_cfg(cfg)


def test_store_and_run_exp_derives_exp_class_from_class(monkeypatch: pytest.MonkeyPatch) -> None:
    import tinyexp

    recorded: dict[str, object] = {}

    class _DummyConfigStore:
        def store(self, name: str, node: object) -> None:
            recorded["name"] = name
            recorded["node"] = node

    dummy_store = _DummyConfigStore()

    def _instance(cls):
        return dummy_store

    monkeypatch.setattr(tinyexp.ConfigStore, "instance", classmethod(_instance))
    monkeypatch.setattr(tinyexp, "simple_launch_exp", lambda: None)

    store_and_run_exp(_StoreAndRunExpCfg)

    expected_path = f"{_StoreAndRunExpCfg.__module__}.{_StoreAndRunExpCfg.__qualname__}"
    assert recorded["name"] == "cfg"
    assert isinstance(recorded["node"], _StoreAndRunExpCfg)
    assert recorded["node"].exp_class == expected_path


def test_store_and_run_exp_overrides_exp_class_field(monkeypatch: pytest.MonkeyPatch) -> None:
    import tinyexp

    @dataclass
    class _BadExpClass(TinyExp):
        exp_class: str = "some.wrong.Path"

    recorded: dict[str, object] = {}

    class _DummyConfigStore:
        def store(self, name: str, node: object) -> None:
            recorded["name"] = name
            recorded["node"] = node

    dummy_store = _DummyConfigStore()

    def _instance(cls):
        return dummy_store

    monkeypatch.setattr(tinyexp.ConfigStore, "instance", classmethod(_instance))
    monkeypatch.setattr(tinyexp, "simple_launch_exp", lambda: None)

    store_and_run_exp(_BadExpClass)

    expected_path = f"{_BadExpClass.__module__}.{_BadExpClass.__qualname__}"
    assert recorded["name"] == "cfg"
    assert isinstance(recorded["node"], _BadExpClass)
    assert recorded["node"].exp_class == expected_path
