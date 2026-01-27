import logging
import pickle
import random
import re
import json
from Typing import Tuple
from enum import IntEnum, StrEnum
from itertools import combinations
from pathlib import Path

from cdcr_lexical_diversity_pairwise_scoring.dataobjs.topics import Topic, ScopeConfig, Topics


logger = logging.getLogger(__name__)


# creating enumerations using class
class Split(StrEnum):
    train = "train"
    dev = "val"
    test = "test"
    na = "n/a"


class DatasetEnum(StrEnum):
    ecb = "ecb"
    wec = "wec"


class POLARITY(IntEnum):
    POSITIVE = 1
    NEGATIVE = 2


class DatasetSetting(StrEnum):
    single = "single"
    excluding_target = "excluding_target"
    mix = "mix"


class MentionPairStrategy(StrEnum):
    all = "all"
    random = "random"
    lemma = "same_lemma"
    embedding = "embedding"
    encoder = "encoder"


class DataSet:
    def __init__(self, name="DataSetSuper", ratio=-1):
        self.ratio = ratio
        self.name = name

    def load_pos_neg_pickle(
        self,
        positive_pairs_path: Path,
        negative_pairs_path: Path,
    ):
        pos_pairs = DataSet.load_pair_pickle(positive_pairs_path)
        neg_pairs = DataSet.load_pair_pickle(negative_pairs_path)

        if self.ratio > 0:
            if len(neg_pairs) > (len(pos_pairs) * self.ratio):
                neg_pairs = neg_pairs[0 : len(pos_pairs) * self.ratio]

        logger.info("Final pos pairs-" + str(len(pos_pairs)))
        logger.info("Final neg pairs-" + str(len(neg_pairs)))
        return self.create_features_from_pos_neg(pos_pairs, neg_pairs)

    @staticmethod
    def load_pair_pickle(
        pair_file_location: Path,
    ):
        logger.debug(f"Loading pairs file: {pair_file_location}")
        with pair_file_location.open("rb") as file:
            pairs = pickle.load(file)
        logger.debug(f"Loaded {len(pairs)} pairs in total.")
        return pairs

    @staticmethod
    def get_dataset(dataset_name: DatasetEnum, ratio=-1, split=Split.na):
        if dataset_name == DatasetEnum.ecb:
            return EcbDataSet(ratio=ratio)
        if dataset_name == DatasetEnum.wec:
            return WecDataSet(ratio=ratio, split=split)
        raise ValueError("Dataset name not supported-" + dataset_name)

    def get_pairwise_feat(self, data_file: Path, to_topics=ScopeConfig.subtopic):
        topics_ = Topics()
        topics_.create_from_file(data_file, keep_order=True)
        logger.info("Create pos/neg examples")
        # Create positive and negative pair within the same ECB+ topic
        positive_, negative_ = self.create_pos_neg_pairs(topics_, to_topics)

        if self.ratio > 0:
            if len(negative_) > (len(positive_) * self.ratio):
                negative_ = negative_[0 : len(positive_) * self.ratio]

        logger.info("pos-" + str(len(positive_)))
        logger.info("neg-" + str(len(negative_)))
        DataSet.validate_pairs(positive_, negative_)
        logger.debug(f"Created {len(positive_)} positive pairs and {len(negative_)} negative pairs.")
        return positive_, negative_

    @classmethod
    def validate_pairs(cls, pos_pairs, neg_pairs):
        for men1, men2 in pos_pairs:
            if men1.coref_chain != men2.coref_chain:
                raise ValueError("Error when validating positive pairs!")

        for men1, men2 in neg_pairs:
            if men1.coref_chain == men2.coref_chain:
                raise ValueError("Error when validating negative pairs!")

        logger.info("Validation Passed!")

    @classmethod
    def create_pos_neg_pairs(cls, topics, to_topic):
        raise NotImplementedError("Method implemented only in subclasses")

    def load_datasets(self, split_file):
        logger.info("Create Features:" + self.name)
        positive_, negative_ = self.get_pairwise_feat(split_file)
        split_feat = self.create_features_from_pos_neg(positive_, negative_)
        return split_feat

    @staticmethod
    def create_features_from_pos_neg(positive_exps, negative_exps):
        feats = list()
        feats.extend(positive_exps)
        feats.extend(negative_exps)
        # feats.extend(random.sample(negative_exps, len(positive_exps) * 2))
        random.shuffle(feats)
        logger.info("Total pairs examples-" + str(len(feats)))
        return feats

    @staticmethod
    def check_and_add_pair(_map, _pairs, mention1, mention2):
        if mention1 is None or mention2 is None:
            return False

        mentions_key1 = mention1.mention_id + "_" + mention2.mention_id
        mentions_key2 = mention2.mention_id + "_" + mention1.mention_id
        if mentions_key1 not in _map and mentions_key2 not in _map:
            _pairs.append((mention1, mention2))
            _map[mentions_key1] = True
            _map[mentions_key2] = True
            return True
        return False


def update_mention_id(mentions, dataset_name: str):
    """
    Ensure that the keys will remain unique across the datasets
    """
    mentions_new = []
    mention_ids = []
    for m in mentions:
        m["mention_id"] = f"{dataset_name}_{m['topic_id']}_{m['subtopic_id']}_{m['mention_id']}"
        m["dataset"] = dataset_name
        m["coref_chain"] = f"{dataset_name}_{m['coref_chain']}"
        m["subtopic_id"] = f"{dataset_name}_{m['subtopic_id']}"
        m["topic_id"] = f"{dataset_name}_{m['topic_id']}"
        mentions_new.append(m)
        mention_ids.append(m["mention_id"])
    return mentions_new, mention_ids


def read_mention_files(dataset_path: Path, dataset_name: str) -> Tuple[list, list]:
    with open(dataset_path / "entity_mentions.json", "r", encoding="utf-8") as file:
        entity_mentions = json.load(file)
        entity_mentions, entity_mention_ids = update_mention_id(entity_mentions, dataset_name)

    with open(dataset_path / "event_mentions.json", "r", encoding="utf-8") as file:
        event_mentions = json.load(file)
        event_mentions, events_mention_ids = update_mention_id(event_mentions, dataset_name)

    return event_mentions, entity_mentions, events_mention_ids, entity_mention_ids


class uCDCRDataSet(DataSet):
    def __init__(self, config, split: Split):
        super(uCDCRDataSet, self).__init__(name="uCDCR")
        self.split = split
        self.pairs = []
        mentions_event = []
        mentions_entity = []
        self.dataset_components = []
        self.setting = config.setting
        self.target_dataset = config.test_dataset_names[0] if self.setting in [DatasetSetting.single, DatasetSetting.excluding_target] else None
        dataset_folder = Path(config.dataset_folder)
        self.topics = Topics()
        # in case of test datasets, these lists of ids will help split into only events and only entities
        self.mention_ids_events, self.mention_ids_entities = [], []
        self.positive_pairs = []
        self.negative_pairs = []

        if split == Split.train:
            # save the attributes related to train
            self.type_of_pairs = MentionPairStrategy(config.type_of_pairs)
            self.ratio = config.ratio
            self.max_pairs = config.max_pairs_train
            self.dataset_scope = ScopeConfig(config.train_scope)

            # check if single, then other datasets are ignores
            if self.setting == DatasetSetting.single and len(config.train_dataset_names) > 1:
                logger.warning(f'More datasets provided in the "single" setting. Only the target dataset will be used as training data. ')
                config.train_dataset_names = config.test_dataset_names

            # read the train data
            for dataset_name in config.train_dataset_names:
                if self.setting == DatasetSetting.excluding_target and dataset_name in config.test_dataset_names:
                    continue

                split_folder = dataset_folder / dataset_name / split.value
                event_mentions, entity_mentions, mention_ids_events, mention_ids_entities = read_mention_files(
                    split_folder, dataset_name)
                self.mention_ids_events.extend(mention_ids_events)
                self.mention_ids_entities.extend(mention_ids_entities)
                if len(event_mentions) + len(event_mentions) == 0:
                    logger.warning(
                        f'No training data for {dataset_name}. Skipped ')
                else:
                    self.dataset_components.append(dataset_name)
                    mentions_event.extend(event_mentions)
                    mentions_entity.extend(entity_mentions)

        elif split == Split.dev:
            if self.setting == DatasetSetting.single and len(config.train_dataset_names) > 1:
                logger.warning(f'More datasets provided in the "single" setting. Only the target dataset will be used as dev data. ')
                config.train_dataset_names = config.test_dataset_names

            # save the attributes related to train
            self.type_of_pairs = MentionPairStrategy.all
            self.ratio = -1
            self.max_pairs = config.max_pairs_dev
            self.dataset_scope = ScopeConfig(config.dev_scope)

            # read the dev data
            for dataset_name in config.train_dataset_names:
                if self.setting == DatasetSetting.excluding_target and dataset_name in config.test_dataset_names:
                    continue

                split_folder = dataset_folder / dataset_name / split.value
                event_mentions, entity_mentions, mention_ids_events, mention_ids_entities = read_mention_files(
                    split_folder, dataset_name)
                self.mention_ids_events.extend(mention_ids_events)
                self.mention_ids_entities.extend(mention_ids_entities)
                self.dataset_components.append(dataset_name)
                mentions_event.extend(event_mentions)
                mentions_entity.extend(entity_mentions)
        else:
            # test
            self.type_of_pairs = MentionPairStrategy.all
            self.ratio = -1
            self.max_pairs = None
            self.dataset_scope = ScopeConfig(config.test_scope)

            # read the dev data
            if self.setting in [DatasetSetting.excluding_target, DatasetSetting.single] and len(config.test_dataset_names) > 1:
                logger.warning(f"The experiments do not have a concept for evaluation of several datasets while training on a single one yet. ")

            for dataset_name in config.test_dataset_names:
                split_folder = dataset_folder / dataset_name / split.value
                event_mentions, entity_mentions, mention_ids_events, mention_ids_entities = read_mention_files(split_folder, dataset_name)
                self.mention_ids_events.extend(mention_ids_events)
                self.mention_ids_entities.extend(mention_ids_entities)
                self.dataset_components.append(dataset_name)
                mentions_event.extend(event_mentions)
                mentions_entity.extend(entity_mentions)

        self.topics.create_from_mention_list(mentions_event + mentions_entity, topic_scope=self.dataset_scope)
        # generate clusters
        self.topics.convert_to_clusters()

    def generate_pairs(self):
        """
        Generate pairs depending on the method for the pair generation
        """
        # TODO check the given limit, the ratio, and the type of pairs, and dataset scope
        if self.ratio == -1 and self.type_of_pairs == MentionPairStrategy.all and self.max_pairs == None:
            self.create_all_pairs()
        pass

    def create_all_pairs(self):
        # create positive examples
        for topic in self.topics.topics_dict.values():
            for i, mention1 in enumerate(topic.mentions):
                if i + 1 == len(topic.mentions):
                    break
                # iterate over the upper triangle of the matrix
                for mention2 in topic.mentions[i + 1:]:
                    if mention1.coref_chain == mention2.coref_chain:
                        self.positive_pairs.append((mention1, mention2))
                    else:
                        self.negative_pairs.append((mention1, mention2))

    @staticmethod
    def from_ecb_subtopic_to_topic(topics):
        new_topics = Topics()
        for sub_topic in topics.topics_dict.values():
            id_num_groups = re.search(r"\b(\d+)\D+", str(sub_topic.topic_id))
            if id_num_groups is not None:
                id_num = id_num_groups.group(1)
                ret_topic = new_topics.get_topic_by_id(id_num)
                if ret_topic is None:
                    ret_topic = Topic(id_num)
                    new_topics.topics_dict[ret_topic.topic_id] = ret_topic

                ret_topic.mentions.extend(sub_topic.mentions)
            else:
                return topics

        return new_topics




class EcbDataSet(DataSet):
    def __init__(self, ratio=-1):
        super(EcbDataSet, self).__init__(ratio=ratio, name="ECB")

    @classmethod
    def create_pos_neg_pairs(cls, topics, to_topic):
        if to_topic == ScopeConfig.topic:
            topics = cls.from_ecb_subtopic_to_topic(topics)
        elif to_topic == ScopeConfig.corpus:
            topics.to_single_topic()

        # create positive examples
        positive_pairs = cls.create_pos_pairs(topics)
        # create negative examples
        negative_pairs = cls.create_neg_pairs(topics)
        random.shuffle(negative_pairs)

        return positive_pairs, negative_pairs

    @classmethod
    def create_pos_pairs(cls, topics):
        return cls.create_pairs(topics, POLARITY.POSITIVE)

    @classmethod
    def create_neg_pairs(cls, topics):
        return cls.create_pairs(topics, POLARITY.NEGATIVE)

    @classmethod
    def create_pairs(cls, topics, polarity):
        _map = dict()
        _pairs = list()
        # create positive examples
        for topic in topics.topics_dict.values():
            for mention1 in topic.mentions:
                for mention2 in topic.mentions:
                    if mention1.mention_id != mention2.mention_id:
                        if (polarity == POLARITY.POSITIVE and mention1.coref_chain == mention2.coref_chain) or (
                            polarity == POLARITY.NEGATIVE and mention1.coref_chain != mention2.coref_chain
                        ):
                            cls.check_and_add_pair(_map, _pairs, mention1, mention2)

        return _pairs

    @staticmethod
    def from_ecb_subtopic_to_topic(topics):
        new_topics = Topics()
        for sub_topic in topics.topics_dict.values():
            id_num_groups = re.search(r"\b(\d+)\D+", str(sub_topic.topic_id))
            if id_num_groups is not None:
                id_num = id_num_groups.group(1)
                ret_topic = new_topics.get_topic_by_id(id_num)
                if ret_topic is None:
                    ret_topic = Topic(id_num)
                    new_topics.topics_dict[ret_topic.topic_id] = ret_topic

                ret_topic.mentions.extend(sub_topic.mentions)
            else:
                return topics

        return new_topics


class WecDataSet(DataSet):
    def __init__(self, ratio=-1, split=Split.na, name="WEC"):
        super(WecDataSet, self).__init__(name=name, ratio=ratio)
        self.split = split

    def create_pos_neg_pairs(self, topics, sub_topics):
        if sub_topics == ScopeConfig.corpus:
            topics.to_single_topic()

        positive_pairs = WecDataSet.create_pos_pairs(topics)
        negative_pairs = self.create_neg_pairs(topics)

        return positive_pairs, negative_pairs

    @staticmethod
    def create_pos_pairs(topics):
        return EcbDataSet.create_pairs(topics, POLARITY.POSITIVE)

    def create_neg_pairs(self, topics):
        if self.split == Split.train:
            clusters = topics.convert_to_clusters()
            negative_pairs = self.create_neg_pairs_wec(clusters)
        else:
            negative_pairs = EcbDataSet.create_pairs(topics, POLARITY.NEGATIVE)

        return negative_pairs

    @classmethod
    def create_neg_pairs_wec(cls, clusters):
        all_mentions = []
        index = -1
        all_ment_index = -1
        found = True
        while found:
            found = False
            index += 1
            all_mentions.append([])
            all_ment_index += 1
            for _, mentions_list in clusters.items():
                if len(mentions_list) > index:
                    all_mentions[all_ment_index].append(mentions_list[index])
                    found = True

        # create WEC negative examples
        list_combined_pairs = list()
        for mention_list in all_mentions:
            if len(mention_list) > 2:
                cls.create_combinations(mention_list, list_combined_pairs)

        return list_combined_pairs

    @staticmethod
    def create_combinations(mentions, list_combined_pairs):
        MAX_SELECT = 3000000
        pair_list = list(combinations(mentions, 2))
        if len(pair_list) > MAX_SELECT:
            pair_list = random.sample(pair_list, MAX_SELECT)

        list_combined_pairs.extend(pair_list)
