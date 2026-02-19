import json
import pickle
import random
from enum import StrEnum
from typing import Self

from cdcr_lexical_diversity_pairwise_scoring import logger
from cdcr_lexical_diversity_pairwise_scoring.constants import *
from cdcr_lexical_diversity_pairwise_scoring.dataobjs.dataset import DataSet
from cdcr_lexical_diversity_pairwise_scoring.dataobjs.mention_data import MentionuCDCR
from cdcr_lexical_diversity_pairwise_scoring.dataobjs.pairs_strategies import (
    AllPositivesStrategy,
    ContrastiveStrategy,
    EvaluationStrategy,
    MentionPairStrategy,
    UniformPositivesStrategy,
)
from cdcr_lexical_diversity_pairwise_scoring.dataobjs.topics import ScopeConfig, Topics


# TODO: remove duplication
class Split(StrEnum):
    train = "train"
    dev = "val"
    test = "test"
    na = "n/a"


class DatasetSetting(StrEnum):
    single = "single"
    excluding_target = "excluding_target"
    mix = "mix"


class EvalPairsType(StrEnum):
    events = "events"
    entities = "entities"
    mix = "mix"


class uCDCRDataSet(DataSet):
    def __init__(
        self,
        dataset_setting: DatasetSetting,
        dataset_names: list[str],
        dataset_root_folder: Path,
        type_of_pairs: MentionPairStrategy,
        negatives_per_positive: int | None,
        max_total_pairs: int | None,
        dataset_scope: ScopeConfig,
        split: Split,
    ):
        super(uCDCRDataSet, self).__init__(name="uCDCR")
        self.split = split
        self.dataset_setting = dataset_setting
        self.dataset_root_folder = dataset_root_folder

        # Type of pairs is more important than negatives_per_positive
        self.type_of_pairs = type_of_pairs
        self.negatives_per_positive = self._get_negatives_per_positive(
            negatives_per_positive,
            self.type_of_pairs,
            DEFAULT_RATIO,
        )

        self.max_total_pairs = max_total_pairs
        self.dataset_scope = dataset_scope

        (
            real_dataset_names,
            global_mention_ids_events,
            global_mention_ids_entities,
            global_mention_events,
            global_mention_entities,
        ) = self._build_events_and_entities_from_datasets(
            dataset_names,
            skip_empty=(self.split == Split.test),
            dataset_root_folder=self.dataset_root_folder,
            split_type=self.split,
        )

        self.dataset_names = real_dataset_names
        self.mention_ids_entities = global_mention_ids_entities
        self.mention_ids_events = global_mention_ids_events

        self.topics = Topics()
        self.topics.create_from_mention_list(
            global_mention_events + global_mention_entities,
            topic_scope=self.dataset_scope,
        )
        self.topics.convert_to_clusters()

        self.positive_pairs: list[tuple[MentionuCDCR, MentionuCDCR]] = []
        self.negative_pairs: list[tuple[MentionuCDCR, MentionuCDCR]] = []

    @staticmethod
    def _get_negatives_per_positive(
        negatives_per_positive: int | None,
        type_of_pairs: MentionPairStrategy,
        default_ratio: int,
    ):
        if type_of_pairs == MentionPairStrategy.all:
            logger.warning(
                f"Overriding negatives_per_positive variable to default {DEFAULT_RATIO} due to"
                f" 'all' option selected as mention pair strategy!"
            )
            return default_ratio

        elif negatives_per_positive is None:
            logger.info(f"Using {DEFAULT_RATIO} as default number of negatives per positives.")
            return default_ratio

        else:
            return negatives_per_positive

    @staticmethod
    def _read_mention_files(
        dataset_path: Path,
        dataset_name: str,
    ) -> tuple[list, list, list, list]:
        with open(dataset_path / "entity_mentions.json", encoding="utf-8") as file:
            entity_mentions = json.load(file)
            entity_mentions, entity_mention_ids = uCDCRDataSet._filter_and_update_mention_attributes(
                entity_mentions,
                dataset_name,
            )

        with open(dataset_path / "event_mentions.json", encoding="utf-8") as file:
            event_mentions = json.load(file)
            event_mentions, events_mention_ids = uCDCRDataSet._filter_and_update_mention_attributes(
                event_mentions,
                dataset_name,
            )

        return event_mentions, entity_mentions, events_mention_ids, entity_mention_ids

    @classmethod
    def _build_events_and_entities_from_datasets(
        cls,
        dataset_names: list[str],
        skip_empty: bool,
        dataset_root_folder: Path,
        split_type: Split,
    ):

        # Handle iterating
        real_dataset_names = []
        global_mention_ids_events = []
        global_mention_ids_entities = []
        global_mention_events = []
        global_mention_entities = []

        for dataset_name in dataset_names:
            split_folder = dataset_root_folder / dataset_name / split_type

            event_mentions, entity_mentions, mention_ids_events, mention_ids_entities = cls._read_mention_files(
                split_folder,
                dataset_name,
            )

            if skip_empty and len(event_mentions) + len(entity_mentions) == 0:
                logger.warning(f"No training data for {dataset_name}. Skipping it.")
                continue

            real_dataset_names.append(dataset_name)
            global_mention_ids_events.extend(mention_ids_events)
            global_mention_ids_entities.extend(mention_ids_entities)
            global_mention_events.extend(event_mentions)
            global_mention_entities.extend(entity_mentions)

        return (
            real_dataset_names,
            global_mention_ids_events,
            global_mention_ids_entities,
            global_mention_events,
            global_mention_entities,
        )

    def get_mix_pairs(self) -> list[tuple[MentionuCDCR, MentionuCDCR]]:
        """Returns all mention pairs combined"""
        all_pairs = self.positive_pairs + self.negative_pairs
        random.shuffle(all_pairs)
        return all_pairs

    def save_dataset(self, save_path: Path):
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("wb") as file:
            pickle.dump(self, file)
        logger.info(f"Saved positive pairs in {save_path}.")

    def generate_pairs(self):
        """Generate pairs depending on the method for the pair generation"""
        if self.split != Split.test:
            logger.info(f"Generating mention pairs using {self.type_of_pairs} strategy.")

        if self.split == Split.test:
            logger.info(f"Generating mention pairs for final evaluation.")
            positive_pairs, negative_pairs = EvaluationStrategy.create_pairs(
                topics=self.topics,
                mention_ids_events=self.mention_ids_events,
                mention_ids_entities=self.mention_ids_entities,
                exclude_singletons=EXCLUDE_SINGLETONS,
            )

        elif self.type_of_pairs in [MentionPairStrategy.all, MentionPairStrategy.random]:
            # TODO: make sure it is correct here and not vice versa
            if self.max_total_pairs is not None:
                positive_pairs, negative_pairs = UniformPositivesStrategy.create_pairs(
                    topics=self.topics,
                    max_total_pairs=self.max_total_pairs,
                    negatives_per_positive=self.negatives_per_positive,
                    dataset_names=self.dataset_names,
                )
            else:
                positive_pairs, negative_pairs = AllPositivesStrategy.create_pairs(
                    topics=self.topics,
                    negatives_per_positive=self.negatives_per_positive,
                    dataset_names=self.dataset_names,
                )

        elif self.type_of_pairs in [
            MentionPairStrategy.tfidf,
            MentionPairStrategy.embedding,
            MentionPairStrategy.encoder,
        ]:
            if self.max_total_pairs is None:
                if self.split == Split.train:
                    max_total_pairs = DEFAULT_TRAIN
                else:
                    max_total_pairs = DEFAULT_DEV
            else:
                max_total_pairs = self.max_total_pairs  # TODO

            positive_pairs, negative_pairs = ContrastiveStrategy.create_pairs(
                self.topics,
                max_total_pairs,
                self.type_of_pairs,
                self.negatives_per_positive,
                self.dataset_names,
                MIN_STD,  # TODO
            )

        else:
            raise NotImplementedError

        logger.info(f"Got positive pairs: {len(positive_pairs)}")
        logger.info(f"Got negative pairs: {len(negative_pairs)}")

        self.positive_pairs = positive_pairs
        self.negative_pairs = negative_pairs

    @staticmethod
    def _filter_and_update_mention_attributes(
        mentions,
        dataset_name: str,
    ):
        """Ensure that the keys will remain unique across the datasets"""
        mentions_new = []
        mention_ids = []
        for m in mentions:
            if dataset_name in ALLOWED_TOPICS:
                if m["topic"] not in ALLOWED_TOPICS[dataset_name]:
                    continue

            m["mention_id"] = f"{dataset_name}_{m['topic_id']}_{m['subtopic_id']}_{m['mention_id']}"
            m["dataset"] = dataset_name
            m["coref_chain"] = f"{dataset_name}_{m['topic_id']}_{m['coref_chain']}"
            m["subtopic_id"] = f"{dataset_name}_{m['subtopic_id']}"
            m["topic_id"] = f"{dataset_name}_{m['topic_id']}"
            mentions_new.append(m)
            mention_ids.append(m["mention_id"])
        return mentions_new, mention_ids

    @property
    def dataset_name(self):
        return self._build_dataset_name(
            split=self.split,
            type_of_pairs=self.type_of_pairs,
            dataset_scope=self.dataset_scope,
            max_total_pairs=self.max_total_pairs,
            negatives_per_positive=self.negatives_per_positive,
            dataset_names=self.dataset_names,
        )

    @staticmethod
    def _build_dataset_name(
        *,
        split: Split,
        type_of_pairs: MentionPairStrategy,
        dataset_scope: ScopeConfig,
        max_total_pairs: int | None,
        negatives_per_positive: int,
        dataset_names: list[str],
    ) -> str:
        return (
            f"{split}"
            f"_{type_of_pairs}"
            f"_{dataset_scope}"
            f"_{max_total_pairs}"
            f"_{negatives_per_positive}"
            f"_{'-'.join(dataset_names)}"
        )

    @classmethod
    def load_from_config(cls, config) -> tuple[Self, Self, Self]:
        result = []
        for split in [Split.train, Split.dev, Split.test]:
            if split == Split.train:
                dataset_name = cls._build_dataset_name(
                    split=split,
                    type_of_pairs=config.type_of_pairs,
                    dataset_scope=config.train_scope,
                    max_total_pairs=config.max_pairs_train,
                    negatives_per_positive=cls._get_negatives_per_positive(
                        config.ratio,
                        type_of_pairs=config.type_of_pairs,
                        default_ratio=DEFAULT_RATIO,
                    ),
                    dataset_names=config.train_dataset_names,
                )
            elif split == Split.dev:
                dataset_name = cls._build_dataset_name(
                    split=split,
                    type_of_pairs=config.dev_type_of_pairs,
                    dataset_scope=config.dev_scope,
                    max_total_pairs=config.max_pairs_dev,
                    negatives_per_positive=cls._get_negatives_per_positive(
                        config.ratio,
                        type_of_pairs=config.dev_type_of_pairs,
                        default_ratio=DEFAULT_RATIO,
                    ),
                    dataset_names=config.train_dataset_names,
                )
            else:
                dataset_name = cls._build_dataset_name(
                    split=split,
                    type_of_pairs=MentionPairStrategy.all,
                    dataset_scope=config.dev_scope,
                    max_total_pairs=None,
                    negatives_per_positive=cls._get_negatives_per_positive(
                        negatives_per_positive=None,
                        type_of_pairs=MentionPairStrategy.all,
                        default_ratio=DEFAULT_RATIO,
                    ),
                    dataset_names=config.train_dataset_names,
                )

            result.append(cls._from_pickle(PROJECT_ROOT / "resources/datasets" / f"{dataset_name}.pkl"))
        return tuple(result)

    @classmethod
    def _from_pickle(cls, dataset_pickle_path: Path) -> Self:
        """Load an instance of this class from a pickle file."""
        with dataset_pickle_path.open("rb") as file:
            obj = pickle.load(file)

        if not isinstance(obj, cls):
            raise TypeError(f"Pickle does not contain {cls.__name__}")

        return obj

    @classmethod
    def build_from_config(cls, config, split: Split) -> Self:
        if split == Split.train:
            dataset = uCDCRDataSet(
                dataset_setting=config.setting,
                dataset_names=config.train_dataset_names,
                dataset_root_folder=PROJECT_ROOT / config.dataset_folder,
                type_of_pairs=config.type_of_pairs,
                negatives_per_positive=config.ratio,
                max_total_pairs=config.max_pairs_train,
                dataset_scope=config.train_scope,
                split=split,
            )

        elif split == Split.dev:
            dataset = uCDCRDataSet(
                dataset_setting=config.setting,
                dataset_names=config.train_dataset_names,
                dataset_root_folder=PROJECT_ROOT / config.dataset_folder,
                type_of_pairs=config.dev_type_of_pairs,
                negatives_per_positive=config.ratio,
                max_total_pairs=config.max_pairs_dev,
                dataset_scope=config.dev_scope,
                split=split,
            )

        else:
            dataset = uCDCRDataSet(
                dataset_setting=config.setting,
                dataset_names=config.test_dataset_names,
                dataset_root_folder=PROJECT_ROOT / config.dataset_folder,
                type_of_pairs=MentionPairStrategy.all,
                negatives_per_positive=None,
                max_total_pairs=None,
                dataset_scope=config.test_scope,
                split=split,
            )
        return dataset
