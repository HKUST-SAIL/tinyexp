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
    pass


@dataclass
class _WrongExpClassCfg(TinyExp):
    exp_class: str = "some.wrong.Path"


@pytest.fixture
def quiet_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RANK", "1")  # avoid noisy stdout prints during tests


@pytest.mark.parametrize(
    ("main_file", "argv", "expected"),
    [("resnet_exp.py", None, "resnet_exp"), (None, "mnist_exp.py", "mnist_exp"), (None, "-c", "exp")],
)
def test_exp_name_defaults_from_main_module_then_argv(
    monkeypatch: pytest.MonkeyPatch, tmp_path, main_file, argv, expected
) -> None:
    dummy_main = types.ModuleType("__main__")
    if main_file:
        dummy_main.__file__ = str(tmp_path / main_file)
    if argv:
        monkeypatch.setattr(sys, "argv", [argv if argv == "-c" else str(tmp_path / argv)])
    monkeypatch.setitem(sys.modules, "__main__", dummy_main)

    assert TinyExp().exp_name == expected


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


def test_set_cfg_overrides_nested_fields_and_records_them(quiet_rank: None) -> None:
    """A full Hydra config carries ``overrided_cfg: {}``; it must not wipe records of earlier fields.

    ``exp_class`` is always filled in by ``store_and_run_exp``; listing it under overrides is noise.
    """
    exp = _CfgExp()

    # Field order matches the dataclass: base TinyExp fields (incl. overrided_cfg) come before subclass fields.
    exp.set_cfg(
        OmegaConf.create({"mode": "help", "overrided_cfg": {}, "exp_class": "pkg.mod.Exp", "sub_cfg": {"a": 3}, "b": 4})
    )

    assert exp.sub_cfg.a == 3
    assert exp.b == 4
    assert exp.exp_class == "pkg.mod.Exp"
    assert exp.overrided_cfg == {
        "mode": {"value": "help", "original": "train"},
        "a": {"value": 3, "original": 1},
        "b": {"value": 4, "original": 2},
    }


def test_set_cfg_overrides_mapping_field(quiet_rank: None) -> None:
    exp = _MappingCfgExp()

    exp.set_cfg(OmegaConf.create({"options": {"base": 3, "extra": {"enabled": True}}}))

    assert exp.options == {"base": 3, "extra": {"enabled": True}}
    assert isinstance(exp.options, dict)
    assert exp.overrided_cfg["options"] == {
        "value": {"base": 3, "extra": {"enabled": True}},
        "original": {"base": 1},
    }


def test_set_cfg_unknown_key_raises(quiet_rank: None) -> None:
    with pytest.raises(UnknownConfigurationKeyError):
        _CfgExp().set_cfg(OmegaConf.create({"no_such_key": 1}))


@pytest.mark.parametrize("exp_cls", [_StoreAndRunExpCfg, _WrongExpClassCfg])
def test_store_and_run_exp_derives_exp_class_from_class(monkeypatch: pytest.MonkeyPatch, exp_cls: type) -> None:
    import tinyexp

    recorded: dict[str, object] = {}
    store = types.SimpleNamespace(store=lambda name, node: recorded.update(name=name, node=node))
    monkeypatch.setattr(tinyexp.ConfigStore, "instance", classmethod(lambda cls: store))
    monkeypatch.setattr(tinyexp, "simple_launch_exp", lambda: None)

    store_and_run_exp(exp_cls)

    assert recorded["name"] == "cfg"
    assert isinstance(recorded["node"], exp_cls)
    # A wrong exp_class default is overridden too.
    assert recorded["node"].exp_class == f"{exp_cls.__module__}.{exp_cls.__qualname__}"


def test_canonical_exp_class_prefers_importable_spec_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """``python -m pkg.mod`` runs the module as __main__; the recorded exp must use
    the canonical importable class so ray ships it by reference (regression:
    by-value export crashed with "Can't pickle <built-in function print>" when a
    platform agent patched builtins.print)."""
    from tinyexp import _resolve_canonical_exp_class
    from tinyexp.examples import vit_tp_exp

    # A class from a normally imported module has no canonical twin to swap in.
    assert _resolve_canonical_exp_class(_CfgExp) is None

    # A __main__ class whose running module has a spec name resolves canonically.
    dummy_main = types.ModuleType("__main__")
    dummy_main.__spec__ = types.SimpleNamespace(name="tinyexp.examples.vit_tp_exp")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "__main__", dummy_main)
    monkeypatch.setattr(vit_tp_exp.VitTpExp, "__module__", "__main__")
    resolved = _resolve_canonical_exp_class(vit_tp_exp.VitTpExp)
    # In this single-import simulation the spec-name lookup returns the very
    # class object we aliased to __main__; under ``python -m`` it is a
    # distinct twin of the same file. Either way it must be a TinyExp subclass
    # with the right qualname.
    assert resolved is not None
    assert issubclass(resolved, TinyExp)
    assert resolved.__qualname__ == "VitTpExp"

    # A plain script (no spec) has no canonical twin.
    dummy_main.__spec__ = None  # type: ignore[attr-defined]
    assert _resolve_canonical_exp_class(vit_tp_exp.VitTpExp) is None
