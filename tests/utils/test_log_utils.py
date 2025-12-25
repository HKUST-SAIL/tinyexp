from __future__ import annotations

from pathlib import Path

from tinyexp.utils.log_utils import tiny_logger_setup


def test_tiny_logger_setup_writes_log_file(tmp_path: Path) -> None:
    logger = tiny_logger_setup(str(tmp_path), distributed_rank=0, filename="test.log", mode="o")
    logger.info("hello tinyexp")
    logger.complete()

    log_path = tmp_path / "test.log"
    assert log_path.is_file()
    assert "hello tinyexp" in log_path.read_text(encoding="utf-8")
