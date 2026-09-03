from __future__ import annotations

import sys
from pathlib import Path

from tinyexp.utils.log_utils import tiny_logger_setup


def test_tiny_logger_setup_writes_log_file(tmp_path: Path) -> None:
    logger = tiny_logger_setup(str(tmp_path), distributed_rank=0, filename="test.log", mode="o")
    logger.info("hello tinyexp")
    logger.complete()

    log_path = tmp_path / "test.log"
    assert log_path.is_file()
    assert "hello tinyexp" in log_path.read_text(encoding="utf-8")


def test_tiny_logger_setup_avoids_python314_multiprocessing_queue(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls = []

    def fake_add(*args, **kwargs):
        calls.append(kwargs)
        return 0

    monkeypatch.setattr("tinyexp.utils.log_utils.logger.add", fake_add)

    tiny_logger_setup(str(tmp_path), distributed_rank=0, filename="test.log", mode="o")

    assert calls[0]["enqueue"] is (sys.version_info < (3, 14))
