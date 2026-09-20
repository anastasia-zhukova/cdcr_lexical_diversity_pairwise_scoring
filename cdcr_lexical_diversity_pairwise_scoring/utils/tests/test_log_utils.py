import logging
from pathlib import Path

from cdcr_lexical_diversity_pairwise_scoring import logger
from cdcr_lexical_diversity_pairwise_scoring.utils.log_utils import LogFile


def test_attached_file_receives_both_logging_systems(tmp_path: Path) -> None:
    log_file = LogFile(tmp_path / "train.log")

    with log_file.attached():
        logger.info("a loguru line")
        logging.getLogger("dataobjs.test").warning("a stdlib line")

    contents = log_file.path.read_text(encoding="utf-8")
    assert "a loguru line" in contents
    assert "a stdlib line" in contents
    assert "WARNING" in contents
    assert "\x1b[" not in contents


def test_detached_file_receives_nothing_more(tmp_path: Path) -> None:
    log_file = LogFile(tmp_path / "train.log")
    log_file.attach()
    logger.info("before detach")
    log_file.detach()

    logger.info("after detach")
    logging.getLogger("dataobjs.test").warning("stdlib after detach")

    contents = log_file.path.read_text(encoding="utf-8")
    assert "before detach" in contents
    assert "after detach" not in contents
    assert "stdlib after detach" not in contents


def test_detach_without_attach_is_harmless(tmp_path: Path) -> None:
    LogFile(tmp_path / "train.log").detach()

    assert not (tmp_path / "train.log").exists()
