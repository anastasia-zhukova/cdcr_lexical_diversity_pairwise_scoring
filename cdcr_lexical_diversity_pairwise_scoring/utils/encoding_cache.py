import pickle
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

import torch
from tqdm import tqdm

from cdcr_lexical_diversity_pairwise_scoring import logger
from cdcr_lexical_diversity_pairwise_scoring.dataobjs.dataset import EvalPairsType
from cdcr_lexical_diversity_pairwise_scoring.dataobjs.mention_data import MentionuCDCR
from cdcr_lexical_diversity_pairwise_scoring.utils.io_utils import get_encoding_cache_file

# hidden states of the mention tokens, the first token, the last token, the number of tokens
CacheEntry = tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]
ENCODING_PAIR_TYPE = "encoding"


class MentionEmbedder(Protocol):
    def get_mention_full_rep(self, mention: MentionuCDCR) -> CacheEntry: ...


class EncodableDataset(Protocol):
    """The parts of `uCDCRDataSet` the encoder relies on."""

    positive_pairs: list
    negative_pairs: list
    positive_pairs_eval_format: dict
    negative_pairs_eval_format: dict
    topics: object


@dataclass
class TopicEncodingTiming:
    dataset: str
    topic_id: str
    mentions_in_topic: int
    encoded_mentions: int
    seconds: float


class EncodingCache:
    """One pickle per (split, dataset, language model) mapping mention ids to their cached representation.

    The file is shared across experiments: mentions already present are kept and never re-encoded,
    new ones are added to it.
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> dict[str, CacheEntry]:
        if not self._path.exists():
            return {}

        with self._path.open("rb") as file:
            return pickle.load(file)

    def save(self, encoded_mentions: dict[str, CacheEntry]) -> None:
        with self._path.open("wb") as file:
            pickle.dump(encoded_mentions, file)

    @staticmethod
    def to_cache_entry(representation: CacheEntry) -> CacheEntry:
        """Detach the representation from the encoder's buffers before it is cached.

        The embedder returns views into the hidden states of the whole context; pickling a view
        serialises the entire backing storage, so each tensor is copied into a compact CPU tensor
        of exactly its own elements.
        """
        hidden, first_token, last_token, mention_size = representation
        return (
            EncodingCache._compact_copy(hidden),
            EncodingCache._compact_copy(first_token),
            EncodingCache._compact_copy(last_token),
            mention_size,
        )

    @staticmethod
    def _compact_copy(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().cpu().clone()


class DatasetMentionEncoder:
    """Encodes the mentions of a dataset split that take part in its pairs and caches them per dataset."""

    def __init__(
        self,
        embed_model: MentionEmbedder,
        language_model: str,
        split: str,
    ) -> None:
        self._embed_model = embed_model
        self._language_model = language_model
        self._split = split

    def encode(self, dataset: EncodableDataset) -> list[TopicEncodingTiming]:
        mentions_to_encode = self.collect_mentions_to_encode(dataset)
        topics_by_dataset = self.group_topics_by_dataset(dataset.topics.topics_to_datasets)
        topic_num = len(dataset.topics.topics_to_datasets)
        timings = []
        topic_position = 0

        for dataset_name, topic_ids in topics_by_dataset.items():
            cache = EncodingCache(get_encoding_cache_file(self._split, dataset_name, self._language_model))
            encoded_mentions = cache.load()
            added_mentions = False

            for topic_id in topic_ids:
                topic = dataset.topics.topics_dict[topic_id]
                encoded_in_topic = 0
                start = datetime.now()

                for mention in tqdm(
                    topic.mentions,
                    desc=f"Encoding mentions of topic {topic_id} ({topic_position}/{topic_num - 1})",
                ):
                    if mention.mention_id in encoded_mentions or mention.mention_id not in mentions_to_encode:
                        continue

                    representation = self._embed_model.get_mention_full_rep(mention)
                    encoded_mentions[mention.mention_id] = EncodingCache.to_cache_entry(representation)
                    encoded_in_topic += 1

                seconds = (datetime.now() - start).total_seconds()
                timings.append(
                    TopicEncodingTiming(dataset_name, topic_id, len(topic.mentions), encoded_in_topic, seconds),
                )
                added_mentions = added_mentions or encoded_in_topic > 0
                topic_position += 1

            # one write per dataset: the cache holds only this dataset's mentions
            if added_mentions:
                cache.save(encoded_mentions)
                logger.info(f"Cached {len(encoded_mentions)} encoded mentions of {dataset_name} in {cache.path}.")

        return timings

    @staticmethod
    def collect_mentions_to_encode(dataset: EncodableDataset) -> set[str]:
        """Only mentions that take part in a pair (train/dev pairs or the evaluation 'mix' pairs) are encoded."""
        mentions_to_encode = set()

        for first_mention, second_mention in dataset.positive_pairs + dataset.negative_pairs:
            mentions_to_encode.add(first_mention.mention_id)
            mentions_to_encode.add(second_mention.mention_id)

        for dataset_name, topic_dict in dataset.positive_pairs_eval_format.items():
            for topic_id, mention_types_dict in topic_dict.items():
                negative_pairs = dataset.negative_pairs_eval_format[dataset_name][topic_id][EvalPairsType.mix]

                for first_mention, second_mention in mention_types_dict[EvalPairsType.mix] + negative_pairs:
                    mentions_to_encode.add(first_mention.mention_id)
                    mentions_to_encode.add(second_mention.mention_id)

        return mentions_to_encode

    @staticmethod
    def group_topics_by_dataset(topics_to_datasets: dict[str, str]) -> dict[str, list[str]]:
        """Group topic ids by their dataset, keeping the first-appearance order of both."""
        topics_by_dataset = defaultdict(list)

        for topic_id, dataset_name in topics_to_datasets.items():
            topics_by_dataset[dataset_name].append(topic_id)

        return dict(topics_by_dataset)
