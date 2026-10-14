import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Self

from cdcr_lexical_diversity_pairwise_scoring import logger
from cdcr_lexical_diversity_pairwise_scoring.constants import LOG_FILE_FORMAT, LOGURU_TO_STDLIB_FORMAT


class LogFile:
    """A file that receives everything logged while it is attached.

    The project logs through two systems: `loguru` in the scripts and stdlib `logging` in `dataobjs`
    (which Hydra wires to its own handlers). Attaching adds one stdlib handler to the root logger and
    routes loguru into the same handler, so both end up in the file with a single format.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._handler: logging.FileHandler | None = None
        self._loguru_sink_id: int | None = None

    @property
    def path(self) -> Path:
        return self._path

    def attach(self) -> Self:
        self._handler = logging.FileHandler(self._path, encoding="utf-8")
        self._handler.setFormatter(logging.Formatter(LOG_FILE_FORMAT))
        logging.getLogger().addHandler(self._handler)
        self._loguru_sink_id = logger.add(self._handler, format=LOGURU_TO_STDLIB_FORMAT, level="DEBUG")
        return self

    def detach(self) -> None:
        if self._handler is None or self._loguru_sink_id is None:
            return

        logger.remove(self._loguru_sink_id)
        logging.getLogger().removeHandler(self._handler)
        self._handler.close()
        self._handler = None
        self._loguru_sink_id = None

    @contextmanager
    def attached(self) -> Iterator[Self]:
        self.attach()
        try:
            yield self
        finally:
            self.detach()
