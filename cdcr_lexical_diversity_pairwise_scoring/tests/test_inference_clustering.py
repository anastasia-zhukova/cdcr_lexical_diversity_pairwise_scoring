from types import SimpleNamespace

import numpy as np
import pandas as pd

from cdcr_lexical_diversity_pairwise_scoring.inference_clustering import build_similarity_matrix


def _mention(mention_id: str) -> SimpleNamespace:
    return SimpleNamespace(mention_id=mention_id)


def _reference_matrix(all_pairs: list, all_scores: list[float], mention_ids: list[str]) -> np.ndarray:
    # the per-pair .loc fill this function replaced, verbatim
    sim_df = pd.DataFrame(np.zeros((len(mention_ids), len(mention_ids))), index=mention_ids, columns=mention_ids)
    for i, pair in enumerate(all_pairs):
        sim_df.loc[pair[0].mention_id, pair[1].mention_id] = all_scores[i]
        sim_df.loc[pair[1].mention_id, pair[0].mention_id] = all_scores[i]
    return sim_df.values


def test_matches_the_per_pair_fill_on_unique_pairs() -> None:
    rng = np.random.default_rng(0)
    mention_ids = [f"m{i}" for i in range(40)]
    mentions = {m: _mention(m) for m in mention_ids}
    pairs = [(mentions[a], mentions[b]) for a in mention_ids for b in mention_ids if a < b]
    pairs = [pairs[i] for i in rng.permutation(len(pairs))[:500]]
    scores = rng.random(len(pairs)).tolist()

    matrix = build_similarity_matrix(pairs, scores, mention_ids)

    np.testing.assert_array_equal(matrix, _reference_matrix(pairs, scores, mention_ids))
    np.testing.assert_array_equal(matrix, matrix.T)


def test_matches_the_per_pair_fill_when_pairs_repeat_or_are_reversed() -> None:
    a, b, c = _mention("a"), _mention("b"), _mention("c")
    pairs = [(a, b), (b, c), (b, a), (a, b), (c, b)]
    scores = [0.1, 0.2, 0.3, 0.4, 0.5]

    matrix = build_similarity_matrix(pairs, scores, ["c", "a", "b"])

    np.testing.assert_array_equal(matrix, _reference_matrix(pairs, scores, ["c", "a", "b"]))
    assert matrix[1, 2] == matrix[2, 1] == 0.4
    assert matrix[0, 2] == matrix[2, 0] == 0.5


def test_unscored_cells_stay_zero() -> None:
    matrix = build_similarity_matrix([(_mention("a"), _mention("b"))], [0.7], ["a", "b", "c"])

    assert matrix[0, 2] == matrix[2, 0] == matrix[2, 2] == 0.0
    assert matrix[0, 1] == matrix[1, 0] == 0.7
