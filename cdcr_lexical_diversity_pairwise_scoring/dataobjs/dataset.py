import pickle
import random
import re
from enum import IntEnum, StrEnum
from itertools import combinations

from cdcr_lexical_diversity_pairwise_scoring import logger
from cdcr_lexical_diversity_pairwise_scoring.constants import *
from cdcr_lexical_diversity_pairwise_scoring.dataobjs.topics import ScopeConfig, Topic, Topics


random.seed(42)


# creating enumenegatives_per_positivens using class
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


class DataSet:
    def __init__(self, name="DataSetSuper", negatives_per_positive=-1):
        self.negatives_per_positive = negatives_per_positive
        self.name = name

    def load_pos_neg_pickle(
        self,
        positive_pairs_path: Path,
        negative_pairs_path: Path,
    ):
        pos_pairs = DataSet.load_pair_pickle(positive_pairs_path)
        neg_pairs = DataSet.load_pair_pickle(negative_pairs_path)

        if self.negatives_per_positive > 0:
            if len(neg_pairs) > (len(pos_pairs) * self.negatives_per_positive):
                neg_pairs = neg_pairs[0 : len(pos_pairs) * self.negatives_per_positive]

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
    def get_dataset(dataset_name: DatasetEnum, negatives_per_positive=-1, split=Split.na):
        if dataset_name == DatasetEnum.ecb:
            return EcbDataSet(negatives_per_positive=negatives_per_positive)
        if dataset_name == DatasetEnum.wec:
            return WecDataSet(negatives_per_positive=negatives_per_positive, split=split)
        raise ValueError("Dataset name not supported-" + dataset_name)

    def get_pairwise_feat(self, data_file: Path, to_topics=ScopeConfig.subtopic):
        topics_ = Topics()
        topics_.create_from_file(data_file, keep_order=True)
        logger.info("Create pos/neg examples")
        # Create positive and negative pair within the same ECB+ topic
        positive_, negative_ = self.create_pos_neg_pairs(topics_, to_topics)

        if self.negatives_per_positive > 0:
            if len(negative_) > (len(positive_) * self.negatives_per_positive):
                negative_ = negative_[0 : len(positive_) * self.negatives_per_positive]

        logger.info("pos-" + str(len(positive_)))
        logger.info("neg-" + str(len(negative_)))
        self.validate_pairs(positive_, negative_)
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


class EcbDataSet(DataSet):
    def __init__(self, negatives_per_positive=-1):
        super(EcbDataSet, self).__init__(negatives_per_positive=negatives_per_positive, name="ECB")

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
    def __init__(self, negatives_per_positive=-1, split=Split.na, name="WEC"):
        super(WecDataSet, self).__init__(name=name, negatives_per_positive=negatives_per_positive)
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
