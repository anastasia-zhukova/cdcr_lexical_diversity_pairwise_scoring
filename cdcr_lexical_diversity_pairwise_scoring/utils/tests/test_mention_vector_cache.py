from pathlib import Path

import numpy as np
import pandas as pd

from cdcr_lexical_diversity_pairwise_scoring.utils.mention_vector_cache import MentionVectorCache


def _vectors(ids: list[str], seed: int) -> pd.DataFrame:
    return pd.DataFrame(np.random.default_rng(seed).random((len(ids), 3)), index=ids)


def test_empty_when_no_file(tmp_path: Path) -> None:
    cache = MentionVectorCache(tmp_path / "model.h5")

    assert cache.vectors.empty
    assert sorted(cache.missing(["a", "b"])) == ["a", "b"]


def test_add_then_get_and_missing(tmp_path: Path) -> None:
    cache = MentionVectorCache(tmp_path / "model.h5")
    cache.add(_vectors(["a", "b"], seed=1))

    assert sorted(cache.missing(["c", "a", "d"])) == ["c", "d"]
    pd.testing.assert_frame_equal(cache.get(["b", "a"]), _vectors(["a", "b"], seed=1).loc[["b", "a"]])


def test_save_persists_and_reloads(tmp_path: Path) -> None:
    path = tmp_path / "model.h5"
    cache = MentionVectorCache(path)
    cache.add(_vectors(["a", "b"], seed=1))
    cache.save()
    cache.add(_vectors(["c"], seed=2))
    cache.save()

    reloaded = MentionVectorCache(path)
    assert list(reloaded.vectors.index) == ["a", "b", "c"]
    pd.testing.assert_frame_equal(reloaded.get(["c"]), _vectors(["c"], seed=2))
    assert not list(tmp_path.glob("*.tmp"))


def test_save_without_changes_does_not_touch_the_file(tmp_path: Path) -> None:
    path = tmp_path / "model.h5"
    cache = MentionVectorCache(path)
    cache.add(_vectors(["a"], seed=1))
    cache.save()
    mtime = path.stat().st_mtime_ns

    MentionVectorCache(path).save()
    cache.add(pd.DataFrame())
    cache.save()

    assert path.stat().st_mtime_ns == mtime


def test_save_if_due_saves_once_enough_vectors_are_pending(tmp_path: Path) -> None:
    path = tmp_path / "model.h5"
    cache = MentionVectorCache(path)

    cache.add(_vectors(["a", "b"], seed=1))
    cache.save_if_due(every=3)
    assert not path.exists()

    cache.add(_vectors(["c"], seed=2))
    cache.save_if_due(every=3)
    assert list(MentionVectorCache(path).vectors.index) == ["a", "b", "c"]
