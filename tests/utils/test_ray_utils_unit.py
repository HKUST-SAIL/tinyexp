from __future__ import annotations

import pytest

from tinyexp.utils.ray_utils import _should_print_launcher, get_launcher


def test_get_launcher_defaults_to_python() -> None:
    assert get_launcher() == "python"


def test_should_print_launcher_based_on_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RANK", raising=False)
    assert _should_print_launcher() is True

    monkeypatch.setenv("RANK", "1")
    assert _should_print_launcher() is False
