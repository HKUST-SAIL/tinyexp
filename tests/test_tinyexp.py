import sys
import types
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
class _MappingCfgExp(TinyExp):
    options: dict[str, object] = field(default_factory=lambda: {"base": 1})


@dataclass
class _StoreAndRunExpCfg(TinyExp):
    check_exp_class: str = f"{__name__}._StoreAndRunExpCfg"


def test_tiny_exp_instantiation():
    class MyExperiment(TinyExp):
        pass

    _ = MyExperiment()


def test_exp_name_defaults_from_main_module_file(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    dummy_main = types.ModuleType("__main__")
    dummy_main.__file__ = str(tmp_path / "resnet_exp.py")
    monkeypatch.setitem(sys.modules, "__main__", dummy_main)

    exp = TinyExp()
    assert exp.exp_name == "resnet_exp"


def test_exp_name_falls_back_to_argv(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    dummy_main = types.ModuleType("__main__")
    monkeypatch.setitem(sys.modules, "__main__", dummy_main)
    monkeypatch.setattr(sys, "argv", [str(tmp_path / "mnist_exp.py")])

    exp = TinyExp()
    assert exp.exp_name == "mnist_exp"


def test_exp_name_defaults_to_exp_for_dash_c(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_main = types.ModuleType("__main__")
    monkeypatch.setitem(sys.modules, "__main__", dummy_main)
    monkeypatch.setattr(sys, "argv", ["-c"])

    exp = TinyExp()
    assert exp.exp_name == "exp"


def test_set_cfg_overrides_exp_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RANK", "1")  # avoid noisy stdout prints during tests
    exp = TinyExp()
    original_exp_name = exp.exp_name

    cfg = OmegaConf.create({"exp_name": "my_exp"})
    exp.set_cfg(cfg)

    assert exp.exp_name == "my_exp"
    assert exp.overrided_cfg["exp_name"] == {"value": "my_exp", "original": original_exp_name}


def test_print_cfg_can_show_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RANK", "0")
    exp = TinyExp()
    original_exp_name = exp.exp_name
    exp.set_cfg(OmegaConf.create({"exp_name": "my_exp"}))

    assert exp.exp_name == "my_exp"
    assert exp.overrided_cfg["exp_name"] == {"value": "my_exp", "original": original_exp_name}

    messages = []
    cfg_dict = exp.print_cfg(types.SimpleNamespace(info=lambda message: messages.append(message)))
    assert messages[0].startswith("-------- Overridden Configurations --------\n    exp_name: my_exp <-- ")
    assert messages[1].startswith("-------- Configurations --------\n")
    assert "overrided_cfg:" not in messages[1]
    assert cfg_dict["exp_name"] == "my_exp"
    assert "overrided_cfg" not in cfg_dict

    messages.clear()
    exp.print_cfg(types.SimpleNamespace(info=lambda message: messages.append(message)), show_overrided=False)
    assert len(messages) == 1
    assert messages[0].startswith("-------- Configurations --------\n")


def test_set_cfg_overrides_nested(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RANK", "1")  # avoid noisy stdout prints during tests
    exp = _CfgExp()

    cfg = OmegaConf.create({"sub_cfg": {"a": 3}, "b": 4})
    exp.set_cfg(cfg)

    assert exp.sub_cfg.a == 3
    assert exp.b == 4


def test_set_cfg_overrides_mapping_field(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RANK", "1")
    exp = _MappingCfgExp()

    cfg = OmegaConf.create({"options": {"base": 3, "extra": {"enabled": True}}})
    exp.set_cfg(cfg)

    assert exp.options == {"base": 3, "extra": {"enabled": True}}
    assert isinstance(exp.options, dict)
    assert exp.overrided_cfg["options"] == {
        "value": {"base": 3, "extra": {"enabled": True}},
        "original": {"base": 1},
    }


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
