from __future__ import annotations

import pytest

from tinyexp.utils.ray_utils import _build_worker_env_vars, _should_print_launcher, get_launcher, get_placement_group


def test_get_launcher_defaults_to_python() -> None:
    assert get_launcher() == "python"


def test_should_print_launcher_based_on_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RANK", raising=False)
    assert _should_print_launcher() is True

    monkeypatch.setenv("RANK", "1")
    assert _should_print_launcher() is False


def test_build_worker_env_vars_prefers_user_defined_ifname(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "eth9")
    env_vars = _build_worker_env_vars(num_worker=2, rank=1, local_rank=1, master_addr="127.0.0.1", master_port=12345)
    assert env_vars["GLOO_SOCKET_IFNAME"] == "eth9"


def test_build_worker_env_vars_omits_ifname_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GLOO_SOCKET_IFNAME", raising=False)
    env_vars = _build_worker_env_vars(num_worker=2, rank=1, local_rank=1, master_addr="127.0.0.1", master_port=12345)
    assert "GLOO_SOCKET_IFNAME" not in env_vars


def test_get_placement_group_defaults_to_pack(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    class FakePlacementGroup:
        def ready(self):
            return "ready"

    def fake_placement_group(bundles, strategy):
        captured["bundles"] = bundles
        captured["strategy"] = strategy
        return FakePlacementGroup()

    monkeypatch.setattr("tinyexp.utils.ray_utils.placement_group", fake_placement_group)
    monkeypatch.setattr("tinyexp.utils.ray_utils.ray.get", lambda ref: ref)

    get_placement_group(num_worker=2, num_gpus_per_worker=0, num_cpus_per_worker=3)

    assert captured == {
        "bundles": [{"CPU": 3, "GPU": 0}, {"CPU": 3, "GPU": 0}],
        "strategy": "PACK",
    }


def test_get_placement_group_accepts_explicit_strategy(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    class FakePlacementGroup:
        def ready(self):
            return "ready"

    def fake_placement_group(bundles, strategy):
        captured["strategy"] = strategy
        return FakePlacementGroup()

    monkeypatch.setattr("tinyexp.utils.ray_utils.placement_group", fake_placement_group)
    monkeypatch.setattr("tinyexp.utils.ray_utils.ray.get", lambda ref: ref)

    get_placement_group(num_worker=2, strategy="SPREAD")

    assert captured["strategy"] == "SPREAD"
