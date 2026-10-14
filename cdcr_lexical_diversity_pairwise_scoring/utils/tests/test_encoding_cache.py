import pickle
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import torch

from cdcr_lexical_diversity_pairwise_scoring.utils import encoding_cache
from cdcr_lexical_diversity_pairwise_scoring.utils.encoding_cache import DatasetMentionEncoder, EncodingCache

CONTEXT_TOKENS = 64
HIDDEN_SIZE = 8
LANGUAGE_MODEL = "fake-model"
SPLIT = "test"


@dataclass
class FakeMention:
    mention_id: str


@dataclass
class FakeTopic:
    topic_id: str
    mentions: list[FakeMention]


@dataclass
class FakeTopics:
    topics_dict: dict[str, FakeTopic]
    topics_to_datasets: dict[str, str]


@dataclass
class FakeDataset:
    topics: FakeTopics
    positive_pairs: list = field(default_factory=list)
    negative_pairs: list = field(default_factory=list)
    positive_pairs_eval_format: dict = field(default_factory=dict)
    negative_pairs_eval_format: dict = field(default_factory=dict)


class FakeEmbedder:
    """Mimics `EmbedTransformersGenerics.get_mention_full_rep`: returns views into a whole-context tensor."""

    def __init__(self) -> None:
        self.encoded_ids: list[str] = []

    def get_mention_full_rep(self, mention: FakeMention) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        self.encoded_ids.append(mention.mention_id)
        generator = torch.Generator().manual_seed(hash(mention.mention_id) % (2**31))
        context = torch.randn(CONTEXT_TOKENS, HIDDEN_SIZE, generator=generator)
        mention_size = 1 + len(mention.mention_id) % 4
        span = context[10 : 10 + mention_size]
        return span, span[0], span[-1], mention_size


def _dataset(topics_to_datasets: dict[str, list[str]], pair_ids: list[tuple[str, str]]) -> FakeDataset:
    """Build a dataset whose topic `t` of dataset `d` holds mentions `<t>_<i>`; only `pair_ids` form pairs."""
    topics_dict = {}
    topic_datasets = {}
    for dataset_name, topic_ids in topics_to_datasets.items():
        for topic_id in topic_ids:
            topics_dict[topic_id] = FakeTopic(topic_id, [FakeMention(f"{topic_id}_{i}") for i in range(3)])
            topic_datasets[topic_id] = dataset_name

    mentions = {m.mention_id: m for topic in topics_dict.values() for m in topic.mentions}
    pairs = [(mentions[a], mentions[b]) for a, b in pair_ids]
    return FakeDataset(topics=FakeTopics(topics_dict, topic_datasets), positive_pairs=pairs)


def _cache_path(root: Path, dataset_name: str) -> Path:
    return root / f"cached_{SPLIT}_{dataset_name}_{LANGUAGE_MODEL}.pickle"


@pytest.fixture
def cache_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(
        encoding_cache,
        "get_encoding_cache_file",
        lambda split, dataset, language_model: tmp_path / f"cached_{split}_{dataset}_{language_model}.pickle",
    )
    return tmp_path


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def encoder(embedder: FakeEmbedder) -> DatasetMentionEncoder:
    return DatasetMentionEncoder(embed_model=embedder, language_model=LANGUAGE_MODEL, split=SPLIT)


def _storage_bytes(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage().nbytes()


def test_cache_entries_are_compact_cpu_copies_with_equal_values(embedder: FakeEmbedder) -> None:
    hidden, first, last, size = embedder.get_mention_full_rep(FakeMention("A_1"))

    cached_hidden, cached_first, cached_last, cached_size = EncodingCache.to_cache_entry((hidden, first, last, size))

    assert torch.equal(cached_hidden, hidden)
    assert torch.equal(cached_first, first)
    assert torch.equal(cached_last, last)
    assert cached_size == size
    assert cached_hidden.device.type == "cpu"
    # the view carried the whole context; the cached copy holds exactly the mention's elements
    assert _storage_bytes(hidden) == CONTEXT_TOKENS * HIDDEN_SIZE * hidden.element_size()
    assert _storage_bytes(cached_hidden) == cached_hidden.numel() * cached_hidden.element_size()
    assert _storage_bytes(cached_first) == cached_first.numel() * cached_first.element_size()
    assert _storage_bytes(cached_last) == cached_last.numel() * cached_last.element_size()


def test_only_mentions_taking_part_in_pairs_are_encoded(
    cache_root: Path,
    embedder: FakeEmbedder,
    encoder: DatasetMentionEncoder,
) -> None:
    dataset = _dataset({"A": ["t1"]}, [("t1_0", "t1_2")])

    encoder.encode(dataset)

    assert sorted(embedder.encoded_ids) == ["t1_0", "t1_2"]
    with _cache_path(cache_root, "A").open("rb") as file:
        assert sorted(pickle.load(file)) == ["t1_0", "t1_2"]


def test_evaluation_mix_pairs_are_encoded_too(cache_root: Path, encoder: DatasetMentionEncoder) -> None:
    dataset = _dataset({"A": ["t1"]}, [])
    mentions = dataset.topics.topics_dict["t1"].mentions
    dataset.positive_pairs_eval_format = {"A": {"t1": {"mix": [(mentions[0], mentions[1])], "events": []}}}
    dataset.negative_pairs_eval_format = {"A": {"t1": {"mix": [(mentions[1], mentions[2])], "events": []}}}

    encoder.encode(dataset)

    with _cache_path(cache_root, "A").open("rb") as file:
        assert sorted(pickle.load(file)) == ["t1_0", "t1_1", "t1_2"]


def test_each_dataset_gets_its_own_cache_without_leaking_other_datasets(
    cache_root: Path,
    encoder: DatasetMentionEncoder,
) -> None:
    # topics of the two datasets interleave in the topic order
    dataset = _dataset({"A": ["a1", "a2"], "B": ["b1"]}, [("a1_0", "a1_1"), ("b1_0", "b1_1"), ("a2_0", "a2_1")])
    dataset.topics.topics_to_datasets = {"a1": "A", "b1": "B", "a2": "A"}

    encoder.encode(dataset)

    with _cache_path(cache_root, "A").open("rb") as file:
        assert sorted(pickle.load(file)) == ["a1_0", "a1_1", "a2_0", "a2_1"]
    with _cache_path(cache_root, "B").open("rb") as file:
        assert sorted(pickle.load(file)) == ["b1_0", "b1_1"]


def test_existing_cache_is_reused_and_extended(
    cache_root: Path,
    embedder: FakeEmbedder,
    encoder: DatasetMentionEncoder,
) -> None:
    dataset = _dataset({"A": ["t1"]}, [("t1_0", "t1_1")])
    already_cached = (torch.ones(2, HIDDEN_SIZE), torch.ones(HIDDEN_SIZE), torch.ones(HIDDEN_SIZE), 2)
    with _cache_path(cache_root, "A").open("wb") as file:
        pickle.dump({"t1_0": already_cached}, file)

    encoder.encode(dataset)

    assert embedder.encoded_ids == ["t1_1"]
    with _cache_path(cache_root, "A").open("rb") as file:
        cached = pickle.load(file)
    assert sorted(cached) == ["t1_0", "t1_1"]
    assert torch.equal(cached["t1_0"][0], already_cached[0])


def test_cache_is_not_rewritten_when_nothing_new_was_encoded(
    cache_root: Path,
    encoder: DatasetMentionEncoder,
) -> None:
    dataset = _dataset({"A": ["t1"]}, [("t1_0", "t1_1")])
    encoder.encode(dataset)
    first_write = _cache_path(cache_root, "A").stat().st_mtime_ns

    encoder.encode(dataset)

    assert _cache_path(cache_root, "A").stat().st_mtime_ns == first_write


def test_timings_report_every_topic_with_its_encoded_count(
    cache_root: Path,  # noqa: ARG001
    encoder: DatasetMentionEncoder,
) -> None:
    dataset = _dataset({"A": ["t1", "t2"]}, [("t1_0", "t1_1")])

    timings = encoder.encode(dataset)

    assert [(t.dataset, t.topic_id, t.mentions_in_topic, t.encoded_mentions) for t in timings] == [
        ("A", "t1", 3, 2),
        ("A", "t2", 3, 0),
    ]
    assert all(timing.seconds >= 0 for timing in timings)


def _reference_encode_dataset_mentions(  # noqa: C901
    dataset: FakeDataset,
    embed_model: FakeEmbedder,
    cache_root: Path,
) -> None:
    """Run the cache logic of `preprocess_embed.encode_dataset_mentions` as it was before the fix.

    Verbatim except for the removed device placement and timing bookkeeping; kept as the behavioural
    reference for the regression test.
    """
    mentions_to_encode = set()
    for mention_pair in dataset.positive_pairs + dataset.negative_pairs:
        mentions_to_encode.add(mention_pair[0].mention_id)
        mentions_to_encode.add(mention_pair[1].mention_id)

    if len(dataset.positive_pairs_eval_format):
        for dataset_name, topic_dict in dataset.positive_pairs_eval_format.items():
            for topic_id, mention_types_dict in topic_dict.items():
                for mention_pair in (
                    mention_types_dict["mix"] + dataset.negative_pairs_eval_format[dataset_name][topic_id]["mix"]
                ):
                    mentions_to_encode.add(mention_pair[0].mention_id)
                    mentions_to_encode.add(mention_pair[1].mention_id)

    encoded_mentions = {}
    cached_vector_path = None

    for topic_id, dataset_name in dataset.topics.topics_to_datasets.items():
        cached_vector_path_next = _cache_path(cache_root, dataset_name)

        if cached_vector_path_next != cached_vector_path:  # noqa: SIM102
            if cached_vector_path_next.exists():
                with open(cached_vector_path_next, "rb") as file:  # noqa: PTH123
                    encoded_mentions = pickle.load(file)

        cached_vector_path = cached_vector_path_next
        topic = dataset.topics.topics_dict[topic_id]

        added_mentions = False
        for mention in topic.mentions:
            if mention.mention_id in encoded_mentions:
                continue

            if mention.mention_id not in mentions_to_encode:
                continue

            added_mentions = True
            hidden, first_token, last_token, mention_size = embed_model.get_mention_full_rep(mention)
            encoded_mentions[mention.mention_id] = (hidden, first_token, last_token, mention_size)

        if added_mentions:
            with cached_vector_path.open("wb") as file:
                pickle.dump(encoded_mentions, file)


def test_cached_representations_match_the_reference_implementation(
    cache_root: Path,
    embedder: FakeEmbedder,  # noqa: ARG001
    encoder: DatasetMentionEncoder,
    tmp_path: Path,
) -> None:
    topics = {"A": ["a1", "a2"], "B": ["b1", "b2"]}
    pairs = [("a1_0", "a1_1"), ("a2_2", "a1_2"), ("b1_0", "b2_1"), ("b2_0", "b2_2")]
    dataset = _dataset(topics, pairs)
    mentions = dataset.topics.topics_dict
    dataset.positive_pairs_eval_format = {
        "B": {"b1": {"mix": [(mentions["b1"].mentions[1], mentions["b1"].mentions[2])]}},
    }
    dataset.negative_pairs_eval_format = {"B": {"b1": {"mix": []}}}

    reference_root = tmp_path / "reference"
    reference_root.mkdir()
    _reference_encode_dataset_mentions(dataset, FakeEmbedder(), reference_root)
    encoder.encode(dataset)

    for dataset_name, topic_ids in topics.items():
        with _cache_path(reference_root, dataset_name).open("rb") as file:
            reference = pickle.load(file)
        with _cache_path(cache_root, dataset_name).open("rb") as file:
            fixed = pickle.load(file)

        # the reference cache of a dataset also carried the mentions of the datasets encoded before it
        own_mention_ids = {m for m in reference if m.split("_")[0] in topic_ids}
        assert set(fixed) == own_mention_ids

        for mention_id, (hidden, first, last, size) in fixed.items():
            reference_hidden, reference_first, reference_last, reference_size = reference[mention_id]
            assert torch.equal(hidden, reference_hidden)
            assert torch.equal(first, reference_first)
            assert torch.equal(last, reference_last)
            assert size == reference_size

    # the reference file of dataset B did contain leaked mentions of dataset A
    with _cache_path(reference_root, "B").open("rb") as file:
        assert any(mention_id.startswith("a") for mention_id in pickle.load(file))
