"""Persistent cache of mention vectors used by the contrastive pair sampling."""

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from cdcr_lexical_diversity_pairwise_scoring import logger

HDF_KEY = "df"


class MentionVectorCache:
    """Vectors of mentions keyed by mention id, persisted in one HDF5 file per vector model.

    The file is read once, on first use, and written back by `save()` / `save_if_due()`: the pair sampling
    saves every few thousand new vectors and at the end of a split instead of rewriting the whole file after
    every topic. The file is replaced atomically so an interrupted save leaves the previous version intact.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._vectors: pd.DataFrame | None = None
        self._unsaved = 0

    @property
    def path(self) -> Path:
        return self._path

    @property
    def vectors(self) -> pd.DataFrame:
        if self._vectors is None:
            if self._path.exists():
                self._vectors = pd.read_hdf(self._path, key=HDF_KEY)
                logger.info(f"Loaded {len(self._vectors)} cached mention vectors from {self._path}.")
            else:
                self._vectors = pd.DataFrame()
        return self._vectors

    def missing(self, mention_ids: Iterable[str]) -> list[str]:
        """Return the given ids that have no vector yet.

        The order is that of a set difference, as in the per-topic caching this class replaced: the
        sentence-transformer pads per batch, so the order in which mentions are encoded changes their
        vectors in the last bits, and keeping it reproduces the previous vectors bit for bit (under the same
        PYTHONHASHSEED).
        """
        return list(set(mention_ids) - set(self.vectors.index))

    def add(self, vectors: pd.DataFrame) -> None:
        if vectors.empty:
            return
        self._vectors = pd.concat([self.vectors, vectors])
        self._unsaved += len(vectors)

    def get(self, mention_ids: list[str]) -> pd.DataFrame:
        return self.vectors.loc[mention_ids]

    def save_if_due(self, every: int) -> None:
        """Save once at least `every` vectors were added since the last save: bounds what a crash loses."""
        if self._unsaved >= every:
            self.save()

    def save(self) -> None:
        if not self._unsaved:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_name(self._path.name + ".tmp")
        self.vectors.to_hdf(tmp_path, key=HDF_KEY, mode="w")
        tmp_path.replace(self._path)
        self._unsaved = 0
        logger.info(f"Saved {len(self.vectors)} mention vectors to {self._path}.")
