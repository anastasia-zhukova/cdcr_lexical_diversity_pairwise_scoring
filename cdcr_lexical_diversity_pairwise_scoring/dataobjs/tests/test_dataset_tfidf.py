from types import SimpleNamespace

import numpy as np

from cdcr_lexical_diversity_pairwise_scoring.dataobjs.dataset import uCDCRDataSet


def _mention(mention_id: str, tokens: list[str], head_lemma: str) -> SimpleNamespace:
    return SimpleNamespace(mention_id=mention_id, tokens_text=tokens, mention_head_lemma=head_lemma)


def _dataset_with_topic(mentions: list[SimpleNamespace]) -> uCDCRDataSet:
    # bypass __init__: _encode_tfidf only needs the topic's mentions
    dataset = uCDCRDataSet.__new__(uCDCRDataSet)
    dataset.topics = SimpleNamespace(topics_dict={"t": SimpleNamespace(mentions=mentions)})
    return dataset


def test_tfidf_vectors_are_indexed_by_unique_mention_id() -> None:
    dataset = _dataset_with_topic(
        [
            _mention("m1", ["plane", "crash"], "crash"),
            _mention("m2", ["the", "rescue"], "rescue"),
            _mention("m1", ["plane", "crashed"], "crash"),  # duplicate id, as in CD2CR / MEANTIME
        ]
    )

    embed_df, sim_df = dataset._encode_tfidf("t")

    assert list(embed_df.index) == ["m1", "m2"]
    assert embed_df.index.is_unique and sim_df.index.is_unique
    assert sim_df.shape == (2, 2)
    assert embed_df.loc["m1"].ndim == 1
    # the last occurrence wins, consistent with the mention dictionaries of the pair sampling
    assert "crashed" in embed_df.columns
    assert np.isclose(sim_df.loc["m1", "m1"], 1.0)


def test_tfidf_topic_without_vocabulary_yields_no_vectors() -> None:
    dataset = _dataset_with_topic([_mention("m1", ["May"], "May")])  # "may" is an sklearn stop word

    embed_df, sim_df = dataset._encode_tfidf("t")

    assert embed_df.empty and sim_df.empty
