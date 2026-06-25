from __future__ import annotations

import pytest

from tinyexp.utils.ray_utils import _build_worker_env_vars, _should_print_launcher, get_launcher


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


def test_build_worker_env_vars_uses_default_ifname(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GLOO_SOCKET_IFNAME", raising=False)
    env_vars = _build_worker_env_vars(num_worker=2, rank=1, local_rank=1, master_addr="127.0.0.1", master_port=12345)
    assert env_vars["GLOO_SOCKET_IFNAME"] in {"lo0", "lo"}
