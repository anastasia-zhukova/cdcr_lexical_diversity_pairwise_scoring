import logging
import math
import pickle
import random
import re
import json
import pandas as pd
import numpy as np
import nltk
from typing import Tuple, Dict, Union, List
from enum import IntEnum, StrEnum
from itertools import combinations, chain
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from staticvectors import StaticVectors
from nltk.corpus import stopwords
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer

from cdcr_lexical_diversity_pairwise_scoring.dataobjs.topics import Topic, ScopeConfig, Topics
from cdcr_lexical_diversity_pairwise_scoring.constants import *

random.seed(42)
nltk.download("stopwords")
HF_HUB_DISABLE_SYMLINKS_WARNING = True

logger = logging.getLogger(__name__)

model = None
def get_sent_transformer_model(model_name: str):
    global model
    if model is None:
        model = SentenceTransformer(model_name)
    return model

model_fasttext = None
def get_fasttext_model(model_name: str):
    global model_fasttext
    if model_fasttext is None:
        model_fasttext = StaticVectors(model_name)
    return model_fasttext

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
    tfidf = "tfidf"
    embedding = "embedding"
    encoder = "encoder"


class EvalPairsType(StrEnum):
    events = "events"
    entities = "entities"
    mix = "mix"


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


def filter_and_update_mention_attributes(mentions, dataset_name: str):
    """
    Ensure that the keys will remain unique across the datasets
    """
    mentions_new = []
    mention_ids = []
    for m in mentions:
        if dataset_name in ALLOWED_TOPICS:
            if m["topic"] not in ALLOWED_TOPICS[dataset_name]:
                continue

        # m["mention_id"] = f"{dataset_name}_{m['topic_id']}_{m['subtopic_id']}_{m['mention_id']}"
        m["dataset"] = dataset_name
        m["coref_chain"] = f"{dataset_name}_{m['topic_id']}_{m['coref_chain']}"
        # m["subtopic_id"] = f"{dataset_name}_{m['subtopic_id']}"
        # m["topic_id"] = f"{dataset_name}_{m['topic_id']}"
        mentions_new.append(m)
        mention_ids.append(m["mention_id"])
    return mentions_new, mention_ids


def read_mention_files(dataset_path: Path, dataset_name: str) -> Tuple[list, list, list, list]:
    with open(dataset_path / "entity_mentions.json", "r", encoding="utf-8") as file:
        entity_mentions = json.load(file)
        entity_mentions, entity_mention_ids = filter_and_update_mention_attributes(entity_mentions, dataset_name)

    with open(dataset_path / "event_mentions.json", "r", encoding="utf-8") as file:
        event_mentions = json.load(file)
        event_mentions, events_mention_ids = filter_and_update_mention_attributes(event_mentions, dataset_name)

    return event_mentions, entity_mentions, events_mention_ids, entity_mention_ids


class uCDCRDataSet(DataSet):
    def __init__(self, config, split: Split):
        super(uCDCRDataSet, self).__init__(name="uCDCR")
        self.split = split
        mentions_event = []
        mentions_entity = []
        self.dataset_components = []
        self.setting = config.setting
        # if self.setting == DatasetSetting.single:
        #     self.target_datasets = [config.train_dataset_names[0]]
        # else:
        #     self.target_datasets = config.test_dataset_names
        dataset_folder = Path(config.dataset_folder)
        self.topics = Topics()
        # in case of test datasets, these lists of ids will help split into only events and only entities
        self.mention_ids_events, self.mention_ids_entities = [], []
        self.positive_pairs = []
        self.negative_pairs = []
        self.positive_pairs_eval_format = {}
        self.negative_pairs_eval_format = {}
        # to check later on hard and easy positives and negatives
        self.total_types = {}

        if split == Split.train:
            # save the attributes related to train
            self.type_of_pairs = MentionPairStrategy(config.type_of_pairs)
            self.ratio = config.ratio
            self.max_pairs = config.max_pairs_train
            self.dataset_scope = config.train_scope

            # check if single, then other datasets are ignores
            if self.setting == DatasetSetting.single and len(config.train_dataset_names) > 1:
                logger.warning(f'More datasets provided in the "single" setting. Only the first target dataset will be used as training data: {config.train_dataset_names[0]} ')
                config.train_dataset_names = config.train_dataset_names[0]

            # read the train data
            for dataset_name in config.train_dataset_names:
                if self.setting == DatasetSetting.excluding_target and dataset_name in config.test_dataset_names:
                    continue

                split_folder = dataset_folder / dataset_name / split.value
                event_mentions, entity_mentions, mention_ids_events, mention_ids_entities = read_mention_files(
                    split_folder, dataset_name)
                self.mention_ids_events.extend(mention_ids_events)
                self.mention_ids_entities.extend(mention_ids_entities)
                if len(event_mentions) + len(entity_mentions) == 0:
                    logger.warning(
                        f'No training data for {dataset_name}. Skipped ')
                else:
                    self.dataset_components.append(dataset_name)
                    mentions_event.extend(event_mentions)
                    mentions_entity.extend(entity_mentions)

        elif split == Split.dev:

            # save the attributes related to train
            self.type_of_pairs = config.dev_type_of_pairs
            if self.type_of_pairs == MentionPairStrategy.all:
                self.ratio = -1
            else:
                self.ratio = config.ratio
            self.max_pairs = config.max_pairs_dev
            self.dataset_scope = config.dev_scope

            # read the dev data
            for dataset_name in config.dev_dataset_names:
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
            self.dataset_scope = config.test_scope

            if self.dataset_scope == ScopeConfig.corpus and len(config.test_dataset_names) > 1:
                raise ValueError(f"The configuration of the {self.dataset_scope} and more than one test dataset can't be possible for the dataset preparation. Change the scope to {ScopeConfig.dataset}.")

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

    def get_mix_pairs(self):
        """
        Returns all mention pairs combined
        """
        mixed_list = self.positive_pairs + self.negative_pairs
        random.shuffle(mixed_list)
        return mixed_list

    def save_dataset(self, save_path: Path):
        with save_path.open("wb") as file:
            pickle.dump(self, file)
        logger.info(f"Saved positive pairs in {save_path}.")

    @classmethod
    def load(cls, path: Path) -> "uCDCRDataSet":
        """
        Load an instance of this class from a pickle file.
        """
        with path.open("rb") as f:
            obj = pickle.load(f)

        if not isinstance(obj, cls):
            raise TypeError(f"Pickle does not contain {cls.__name__}")

        return obj

    def generate_pairs(self):
        """
        Generate pairs depending on the method for the pair generation
        """
        logger.info(f"Generating mention pairs using {self.type_of_pairs} strategy. ")
        if self.type_of_pairs == MentionPairStrategy.all:
            # dev
            # both positives and negatives are capped given the max number and the ratio
            if self.max_pairs is not None and self.split == Split.dev:
                self.create_all_pairs_capped(same_mention_type=False)

            # extensive dev with all positives but negatives are capped to the ratio
            elif self.max_pairs is None and self.split == Split.dev:
                self.create_all_pairs_negatives_capped(same_mention_type=False)

            # test
            else:
                # three-level nested dictionary for the evaluation
                self.create_all_pairs_eval_format()

        if self.type_of_pairs == MentionPairStrategy.random:
            if self.max_pairs is not None:
                self.create_all_pairs_capped(same_mention_type=True)
            else:
                self.create_all_pairs_negatives_capped(same_mention_type=True)

        if self.type_of_pairs in [MentionPairStrategy.tfidf, MentionPairStrategy.embedding, MentionPairStrategy.encoder]:
            self.create_contrastive_pairs()

    def create_contrastive_pairs(self):
        """
        Builds mention pairs on the contrastive neighboor principle
        """
        if self.max_pairs is None:
            if self.split == Split.train:
                self.max_pairs = DEFAULT_TRAIN
            else:
                self.max_pairs = DEFAULT_DEV

        # compute max per dataset component
        if self.ratio > -1:
            positive_n_max = self.max_pairs // (1 + self.ratio) // len(self.dataset_components)
            ratio = self.ratio
        else:
            # make a default ratio 20 as in SOTA
            positive_n_max = self.max_pairs // (1 + DEFAULT_RATIO) // len(self.dataset_components)
            ratio = DEFAULT_RATIO

        used_up_n = {d: 0 for d in self.dataset_components}
        all_mention_pairs = {d: {} for d in self.dataset_components}
        not_used_mentions_pairs = {d: {} for d in self.dataset_components}

        shuffled_topics = list(self.topics.topic_clusters.items())
        random.shuffle(shuffled_topics)
        shuffled_clusters = dict(shuffled_topics)
        next_report_milestone = 0.1

        # for topic_id, clusters in self.topics.topic_clusters.items():
        for topic_id, clusters in shuffled_clusters.items():
            dataset = self.topics.topics_to_datasets[topic_id]
            if used_up_n[dataset] >= positive_n_max:
                continue

            shuffled_clusters = list(clusters.items())
            random.shuffle(shuffled_clusters)
            shuffled_clusters = dict(shuffled_clusters)

            logger.info(f"Encoding topic {topic_id}.")
            topic_embed_df, sim_df = self.encode_mentions(topic_id)
            mentions_topic_dict = {m.mention_id: m for m in self.topics.topics_dict[topic_id].mentions}

            for c_id, mentions in shuffled_clusters.items():
                if len(mentions) == 1:
                    # exclude singletons from being target mentions
                    continue

                if used_up_n[dataset] >= positive_n_max:
                    break

                # compute the threshold
                # some mentions could have been illuminated if no vector was computed for them, e.g., after the stopword removal in the embedding method
                mentions_cluster_dict = {m.mention_id: m for m in mentions if m.mention_id in topic_embed_df.index.to_list()}
                if not len(mentions_cluster_dict):
                    continue

                m_ids = list(mentions_cluster_dict)

                cluster_df = topic_embed_df.loc[m_ids]
                sim = cosine_similarity(cluster_df.values)
                threshold = float(np.mean(sim))
                std = float(np.std(sim))

                if std < MIN_STD:
                    # logger.warning(f"Cluster {c_id} has too little lexical variation. Only easy positives selected.")
                    all_mention_pairs[dataset][c_id] = {"pos_easy": set(), "pos_hard": set(), "neg_easy": set(),
                                                        "neg_hard": set()}
                    target_mention_id = cluster_df.index[0]
                    target_vector = cluster_df.loc[target_mention_id]
                    sim_target = cosine_similarity([target_vector], topic_embed_df.values)[0]
                    sim_series_target = pd.Series(sim_target, index=topic_embed_df.index)
                    sim_series_target = sim_series_target.drop(target_mention_id)

                    delta = MIN_STD / DENOM_DELTA
                    pos_easy = cluster_df.index[1:].to_list()

                    if threshold >= 1.0:
                        #  really no variation, so to get smth in, subtract delta
                        neg_hard = list(
                            set(sim_series_target[sim_series_target >= threshold - delta].index) - set(m_ids))
                    else:
                        neg_hard = list(set(sim_series_target[sim_series_target >= threshold + delta].index) - set(m_ids))

                    # form easies that are more remote than usually
                    neg_easy = list(set(sim_series_target[sim_series_target <= threshold - MIN_STD].index) - set(m_ids))
                    num_pairs_per_pos_type = len(pos_easy)
                    num_pairs_per_neg_type = num_pairs_per_pos_type * ratio

                    # use up all positives
                    all_mention_pairs[dataset][c_id]["pos_easy"].update(
                        set(uCDCRDataSet._make_pairs_with_target(target_mention_id, pos_easy, mentions_topic_dict,
                                                                 max_num=num_pairs_per_pos_type)))
                    # first use up all hard negatives, then the remaining take form easy negatives
                    hard_negative_pairs = uCDCRDataSet._make_pairs_with_target(target_mention_id, neg_hard,
                                                                               mentions_topic_dict,
                                                                               max_num=num_pairs_per_neg_type)
                    all_mention_pairs[dataset][c_id]["neg_hard"].update(set(hard_negative_pairs))
                    all_mention_pairs[dataset][c_id]["neg_easy"].update(
                        set(uCDCRDataSet._make_pairs_with_target(target_mention_id, neg_easy, mentions_topic_dict,
                                                                 max_num=num_pairs_per_neg_type - len(
                                                                     hard_negative_pairs))))

                else:
                    delta = std / DENOM_DELTA
                    # compute centroid
                    centroid = np.mean(cluster_df.values, axis=0)
                    sim_centroid = cosine_similarity([centroid], cluster_df.values)[0]
                    sim_series = pd.Series(sim_centroid, index=cluster_df.index)

                    # choose two target mentions: one close to centroid, one - faw away

                    for target_mention_id in [sim_series.idxmax(), sim_series.idxmin()]:
                        target_vector = cluster_df.loc[target_mention_id]
                        sim_target = cosine_similarity([target_vector], topic_embed_df.values)[0]
                        sim_series_target = pd.Series(sim_target, index=topic_embed_df.index)
                        sim_series_target = sim_series_target.drop(target_mention_id)

                        pos_easy = list(set(sim_series_target[sim_series_target >= threshold + delta].index).intersection(set(m_ids)))
                        pos_hard = list(set(sim_series_target[sim_series_target <= threshold - delta].index).intersection(set(m_ids)))
                        neg_hard = list(set(sim_series_target[sim_series_target >= threshold + delta].index) - set(m_ids))
                        neg_easy = list(set(sim_series_target[sim_series_target <= threshold - delta].index) - set(m_ids))
                        # neg_easy = sim_series_target[sim_series_target <= threshold - DELTA].sort_values(ascending=False).index.to_list()

                        # we use all positives and sample negatives
                        num_pairs_per_pos_type = len(pos_easy) + len(pos_hard)
                        num_pairs_per_neg_type = num_pairs_per_pos_type * ratio

                        if num_pairs_per_neg_type > len(neg_easy) + len(neg_hard):
                            # not enough negatives
                            continue

                        if c_id not in all_mention_pairs[dataset]:
                            all_mention_pairs[dataset][c_id] = {"pos_easy": set(), "pos_hard": set(), "neg_easy": set(),
                                                                "neg_hard": set()}
                        # use up all positives
                        all_mention_pairs[dataset][c_id]["pos_easy"].update(set(uCDCRDataSet._make_pairs_with_target(target_mention_id, pos_easy, mentions_topic_dict, max_num=num_pairs_per_pos_type)))
                        all_mention_pairs[dataset][c_id]["pos_hard"].update(set(uCDCRDataSet._make_pairs_with_target(target_mention_id, pos_hard, mentions_topic_dict, max_num=num_pairs_per_pos_type)))
                        # first use up all hard negatives, then the remaining take form easy negatives
                        hard_negative_pairs = uCDCRDataSet._make_pairs_with_target(target_mention_id, neg_hard, mentions_topic_dict, max_num=num_pairs_per_neg_type)
                        all_mention_pairs[dataset][c_id]["neg_hard"].update(set(hard_negative_pairs))
                        all_mention_pairs[dataset][c_id]["neg_easy"].update(set(uCDCRDataSet._make_pairs_with_target(target_mention_id, neg_easy, mentions_topic_dict, max_num=num_pairs_per_neg_type - len(hard_negative_pairs))))

                used_up_n[dataset] += len(all_mention_pairs[dataset][c_id]["pos_easy"]) + len(all_mention_pairs[dataset][c_id]["pos_hard"])
                if used_up_n[dataset] / positive_n_max >= next_report_milestone:
                    logger.info(f"Collected at least {next_report_milestone * 100}% of the {positive_n_max} positive pairs for {dataset}.")
                    next_report_milestone += 0.1

        self.total_types = {}
        for dataset, clusters_per_dataset in all_mention_pairs.items():
            self.total_types[dataset] = {"pos_easy": 0, "pos_hard": 0, "neg_easy": 0, "neg_hard": 0}
            for pair_types in list(clusters_per_dataset.values()):
                for p, pairs in pair_types.items():
                    self.total_types[dataset][p] += len(pairs)

                self.positive_pairs.extend(list(pair_types["pos_easy"]) + list(pair_types["pos_hard"]))
                self.negative_pairs.extend(list(pair_types["neg_easy"]) + list(pair_types["neg_hard"]))


    @classmethod
    def _make_pairs_with_target(cls, m_target_id, mention_ids, mention_dict, max_num):
        if not len(mention_ids):
            return []

        text_unique = {}
        for m_id in mention_ids:
            if mention_dict[m_id].tokens_str not in text_unique:
                text_unique[mention_dict[m_id].tokens_str] = []
            text_unique[mention_dict[m_id].tokens_str].append(m_id)

        text_unique = dict(sorted(text_unique.items(), key=lambda item: len(item[1])))
        if len(text_unique) > max_num:
            # take mentions with unique wording
            used_ids = [m_ids[0] for m_ids in list(text_unique.values())[:max_num]]
        else:
            used_ids = mention_ids[:max_num]

        return [(mention_dict[m_target_id], mention_dict[m_id]) for m_id in used_ids]

    def encode_mentions(self, topic_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.type_of_pairs == MentionPairStrategy.tfidf:
            embed_df, sim_df = self._encode_tfidf(topic_id)

        elif self.type_of_pairs == MentionPairStrategy.embedding:
            embed_df, sim_df = self._encode_embeddings(topic_id)

        elif self.type_of_pairs == MentionPairStrategy.encoder:
            embed_df, sim_df = self._encode_sentence_transformer(topic_id)
        else:
            raise NotImplementedError(f"A method for the mention pair creation {self.type_of_pairs} is not implemented.")
        return embed_df, sim_df

    def _encode_tfidf(self, topic_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        texts, ids = [], []

        for m in self.topics.topics_dict[topic_id].mentions:
            tokens_clean = self._remove_stopwords(m.tokens_text)
            if not len(tokens_clean):
                continue

            if m.mention_head_lemma not in tokens_clean:
                tokens_clean.append(m.mention_head_lemma)

            texts.append(" ".join(tokens_clean))
            ids.append(m.mention_id)

        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
        )

        X = vectorizer.fit_transform(texts)
        embeddings = X.toarray()
        embed_df = pd.DataFrame(embeddings, index=ids, columns=vectorizer.get_feature_names_out())
        sim = cosine_similarity(embeddings)
        sim_df = pd.DataFrame(sim, index=ids, columns=ids)
        return embed_df, sim_df

    def _encode_embeddings(self, topic_id: str)-> Tuple[pd.DataFrame, pd.DataFrame]:
        texts_dict, heads_dict = {}, {}
        cached_path = PROJECT_ROOT / "resources" / f"{EMBEDDING.replace('/', '_')}.h5"
        if cached_path.exists():
            existing_embed_df = pd.read_hdf(cached_path, key="df")
        else:
            existing_embed_df = pd.DataFrame()

        for m in self.topics.topics_dict[topic_id].mentions:
            tokens_clean = self._remove_stopwords(m.tokens_text)
            if not len(tokens_clean):
                continue
            texts_dict[m.mention_id] = tokens_clean
            heads_dict[m.mention_id] = m.mention_head

        mentions_to_index = list(set(texts_dict) - set(existing_embed_df.index.to_list()))
        texts_dict_to_use = {m: texts_dict[m] for m in mentions_to_index}

        embeddings = list()
        for m_id, tokens in texts_dict_to_use.items():
            vector = self._encode_sentence(tokens, heads_dict[m_id])
            embeddings.append(vector)

        embeddings = np.vstack(embeddings)
        new_embed_df = pd.DataFrame(embeddings, index=mentions_to_index)
        embed_df = pd.concat([existing_embed_df, new_embed_df])

        embed_df.to_hdf(
            cached_path,
            key="df",
            mode="w"  # "w" = overwrite, "a" = append
        )

        topic_embed_df = embed_df.loc[list(texts_dict)]
        sim = cosine_similarity(topic_embed_df.values)

        sim_df = pd.DataFrame(sim, index=topic_embed_df.index, columns=topic_embed_df.index)
        return topic_embed_df, sim_df


    def _remove_stopwords(self, tokens: List[str]) -> List[str]:
        tokens_clean = []
        stop_words = set(stopwords.words("english"))
        for t in tokens:
            if t not in stop_words and t not in punctuation:
                tokens_clean.append(t)
        return tokens_clean

    def _encode_sentence(self, tokens: List[str], head_token: str, head_weight: float = 2.0) -> np.array:
        """
        Encode a list of mention tokens with static embeddings
        """
        global model_fasttext
        model_fasttext = get_fasttext_model(EMBEDDING)
        vectors = model_fasttext.embeddings(tokens)

        weights = [head_weight if t == head_token else 1.0 for t in tokens]
        weights = np.array(weights, dtype=np.float32)

        # weighted mean
        embedding = np.average(vectors, axis=0, weights=weights)
        norm = np.linalg.norm(embedding)
        embedding = embedding / norm
        return embedding.astype(np.float32)


    def _encode_sentence_transformer(self, topic_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        model = get_sent_transformer_model(SENT_TRANSFOMER)
        cached_path = PROJECT_ROOT / "resources" / f"{SENT_TRANSFOMER.replace('/', '_')}.h5"
        if cached_path.exists():
            existing_embed_df = pd.read_hdf(cached_path, key="df")
        else:
            existing_embed_df = pd.DataFrame()

        texts_dict = {}
        for m in self.topics.topics_dict[topic_id].mentions:
            texts_dict[m.mention_id] = m.tokens_str

        mentions_to_index = list( set(texts_dict) - set(existing_embed_df.index.to_list()))
        texts = [texts_dict[m] for m in mentions_to_index]

        embeddings = model.encode(
            texts,
            batch_size=ENCODE_BATCH,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        new_embed_df = pd.DataFrame(embeddings, index=mentions_to_index)

        embed_df = pd.concat([existing_embed_df, new_embed_df])

        embed_df.to_hdf(
            cached_path,
            key="df",
            mode="w"  # "w" = overwrite, "a" = append
        )

        topic_embed_df = embed_df.loc[list(texts_dict)]
        sim = cosine_similarity(topic_embed_df.values)

        sim_df = pd.DataFrame(sim, index=topic_embed_df.index, columns=topic_embed_df.index)
        return topic_embed_df, sim_df


    def create_all_pairs_capped(self, same_mention_type: bool):
        """
        Created all mention pairs or the upper triangle, caps them, and only takes up as much as the limit allows
        """

        # compute max per dataset component
        if self.ratio > -1:
            ratio = self.ratio
            positive_n_max = self.max_pairs // (1 + self.ratio) // len(self.dataset_components)
        else:
            # make a default ratio 20 as in SOTA
            ratio = DEFAULT_RATIO
            positive_n_max = self.max_pairs  // (1 + DEFAULT_RATIO) // len(self.dataset_components)

        used_up_n = {d: 0 for d in self.dataset_components}
        not_used_mentions_pairs = {d: [] for d in self.dataset_components}
        used_topics = {}

        shuffled_clusters = list(self.topics.clusters.items())
        random.shuffle(shuffled_clusters)
        shuffled_clusters = dict(shuffled_clusters)

        for c_id, mentions in shuffled_clusters.items():
            topic_mentions_dict = {}
            # some clusters are cross-subtopic, so if the topic level is subtopic, we need to make sure that the positive pairs will be created on the level that we got from config
            for m in mentions:
                topic_id = self.topics.mention_to_topic[m.mention_id]
                if topic_id not in topic_mentions_dict:
                    topic_mentions_dict[topic_id] = []
                topic_mentions_dict[topic_id].append(m)

            for topic_id, topic_mentions in topic_mentions_dict.items():
                if len(topic_mentions) == 1:
                    continue

                dataset = topic_mentions[0].dataset
                if used_up_n[dataset] == positive_n_max:
                    continue

                triangle_n = len(topic_mentions) * (len(topic_mentions) - 1) // 2
                # limit positives to sqrt 6
                max_local_pairs = min(triangle_n, int(6 * math.sqrt(triangle_n)))

                # create the upper triangle
                triangle_pairs = list(combinations(topic_mentions, 2))
                random.shuffle(triangle_pairs)

                # limit to what is still there to take up
                max_to_take = min(max_local_pairs, positive_n_max - used_up_n[dataset])
                self.positive_pairs.extend(triangle_pairs[:max_to_take])
                not_used_mentions_pairs[dataset].extend(triangle_pairs[max_to_take:])
                used_up_n[dataset] += max_to_take

                if topic_id not in used_topics:
                    used_topics[topic_id] = list()

                used_topics[topic_id].append(set(m.mention_id for m in topic_mentions))

        for d in self.dataset_components:
            # if we didn't get enough positive pairs per dataset, take the missing pairs from the not used pairs
            if used_up_n[d] < positive_n_max:
                diff = positive_n_max - used_up_n[d]
                self.positive_pairs.extend(not_used_mentions_pairs[d][:diff])

        negative_n_max = (self.max_pairs - positive_n_max * len(self.dataset_components)) // len(self.dataset_components)
        self._create_capped_negatives({d: negative_n_max for d in self.dataset_components}, used_topics, ratio, same_mention_type)

    def create_all_pairs_eval_format(self):
        """
        Creates mention pairs on the level suitable for evaluation:
        - dataset_name
            - scope_name (e.g., subtopic)
                - events (e.g., pairs on the event level)
                - entities (e.g., pairs on the entity level)
                - mix (e.g., mixed pairs of events and entities level)
        """
        dataset_dict = {}
        for t, d in self.topics.topics_to_datasets.items():
            if d not in dataset_dict:
                dataset_dict[d] = []
            dataset_dict[d].append(t)

        for dataset, topics in dataset_dict.items():
            self.positive_pairs_eval_format[dataset] = {}
            self.negative_pairs_eval_format[dataset] = {}

            for topic_id in topics:
                self.positive_pairs_eval_format[dataset][topic_id] = {}
                self.negative_pairs_eval_format[dataset][topic_id] = {}

                for pairs_type in [EvalPairsType.events, EvalPairsType.entities, EvalPairsType.mix]:
                    self.positive_pairs_eval_format[dataset][topic_id][pairs_type.value] = []
                    self.negative_pairs_eval_format[dataset][topic_id][pairs_type.value] = []

                    if pairs_type == EvalPairsType.events:
                        mentions = [m for m in self.topics.topics_dict[topic_id].mentions if m.mention_id in self.mention_ids_events]
                    elif pairs_type == EvalPairsType.entities:
                        mentions = [m for m in self.topics.topics_dict[topic_id].mentions if m.mention_id in self.mention_ids_entities]
                    else:
                        mentions = self.topics.topics_dict[topic_id].mentions

                    if EXCLUDE_SINGLETONS:
                        mentions_eval = [m for m in mentions if not m.is_singleton]
                    else:
                        mentions_eval = mentions

                    triangle_pairs = list(combinations(mentions_eval, 2))

                    for mention1, mention2 in triangle_pairs:
                        if mention1.coref_chain == mention2.coref_chain:
                            self.positive_pairs_eval_format[dataset][topic_id][pairs_type.value].append((mention1, mention2))
                        else:
                            self.negative_pairs_eval_format[dataset][topic_id][pairs_type.value].append((mention1, mention2))

    def create_all_pairs(self) -> Tuple[Dict[str, int], dict]:
        """
        Creates all permutations of the pairs in the upper triangle and splits into positive and negatives
        """
        positive_counter = {d: 0 for d in self.dataset_components}
        for topic in self.topics.topics_dict.values():
            triangle_pairs = list(combinations(topic.mentions, 2))
            for mention1, mention2 in triangle_pairs:
                if mention1.coref_chain == mention2.coref_chain:
                    self.positive_pairs.append((mention1, mention2))
                    positive_counter[mention1.dataset] += 1
                else:
                    self.negative_pairs.append((mention1, mention2))

        used_topics = {}
        for topic, clusters in self.topics.topic_clusters.items():
            used_topics[topic] = []

            for c_id, mentions in clusters.items():
                used_topics[topic].append({m.mention_id for m in mentions})

        return positive_counter, used_topics


    def create_all_pairs_negatives_capped(self, same_mention_type: bool):
        """
        Just in case we want all positives but not overwhelm with negatives. Does not balance per dataset,
        so the number of negatives will be proportional to the number of positives per dataset.
        """
        positive_counter_datasets, used_topics = self.create_all_pairs()
        ratio = self.ratio if self.ratio > -1 else DEFAULT_RATIO
        negative_counter_datasets = {d: ratio * pos for d, pos in positive_counter_datasets}
        self._create_capped_negatives(negative_counter_datasets, used_topics, ratio, same_mention_type)


    def _create_capped_negatives(self, max_topic_negatives: dict, used_topics: dict, ratio: int, same_mention_type: bool):
        """
        Stratified negative creation based on the positive pairs
        """
        negative_pairs_dataset = {d: [] for d in list(max_topic_negatives)}

        for t_id, topic in self.topics.topics_dict.items():
            dataset = self.topics.topics_to_datasets[t_id]

            if t_id not in used_topics:
                continue

            mention_ids_topic = {m.mention_id: m for m in topic.mentions}
            for positive_mentions_cluster in used_topics[t_id]:

                if same_mention_type:
                    # only entities as negatives
                    if list(positive_mentions_cluster)[0] in self.mention_ids_entities:
                        negative_mentions = [m_id for m_id in set(self.mention_ids_entities) - positive_mentions_cluster if m_id in mention_ids_topic]
                    else:
                        negative_mentions = [m_id for m_id in set(self.mention_ids_events) - positive_mentions_cluster if m_id in mention_ids_topic]
                else:
                    # all other mentions outside a cluster are negatives
                    negative_mentions = list(set(mention_ids_topic) - positive_mentions_cluster)

                # all combinations of the positive mentions
                pos_num_pairs = len(positive_mentions_cluster) * (len(positive_mentions_cluster) - 1) // 2
                neg_num_pairs_max = pos_num_pairs * ratio

                negative_pairs = []
                for pos in list(positive_mentions_cluster):
                    if len(negative_pairs) == neg_num_pairs_max:
                        break

                    for neg in negative_mentions:
                        if len(negative_pairs) == neg_num_pairs_max:
                            break

                        negative_pairs.append((mention_ids_topic[pos], mention_ids_topic[neg]))

                negative_pairs_dataset[dataset].extend(negative_pairs)

        for d, negatives in negative_pairs_dataset.items():
            random.shuffle(negatives)
            self.negative_pairs.extend(negatives[:max_topic_negatives[d]])
        a = 0



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
