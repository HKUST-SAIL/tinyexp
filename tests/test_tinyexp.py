from dataclasses import dataclass, field

import pytest
from omegaconf import OmegaConf

from tinyexp import TinyExp
from tinyexp.exceptions import UnknownConfigurationKeyError


@dataclass
class _CfgExp(TinyExp):
    @dataclass
    class SubCfg:
        a: int = 1

    sub_cfg: SubCfg = field(default_factory=SubCfg)
    b: int = 2


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
