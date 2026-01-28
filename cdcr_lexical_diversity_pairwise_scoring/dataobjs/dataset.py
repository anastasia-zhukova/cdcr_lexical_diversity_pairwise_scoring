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
from itertools import combinations
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from staticvectors import StaticVectors
from nltk.corpus import stopwords
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer

from cdcr_lexical_diversity_pairwise_scoring.dataobjs.topics import Topic, ScopeConfig, Topics
from cdcr_lexical_diversity_pairwise_scoring.constants import DEFAULT_RATIO, DEFAULT_DEV, DEFAULT_TRAIN, SENT_TRANSFOMER, ENCODE_BATCH, DELTA

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
def get_fasttext_model(model_name: str = "neuml/fasttext"):
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
    lemma = "lemma"
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


def read_mention_files(dataset_path: Path, dataset_name: str) -> Tuple[list, list, list, list]:
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
        if self.type_of_pairs == MentionPairStrategy.all:
            # dev
            if self.max_pairs is not None and self.split == Split.dev:
                self.create_all_pairs_capped()

            # extensive dev
            elif self.max_pairs is None and self.split == Split.dev:
                self.create_all_pairs_negatives_capped()

            # test
            else:
                self.create_all_pairs()

        if self.type_of_pairs == MentionPairStrategy.random:
            if self.max_pairs is not None:
                self.create_all_pairs_capped()
            else:
                self.create_all_pairs_negatives_capped()

        if self.type_of_pairs in [MentionPairStrategy.lemma, MentionPairStrategy.embedding, MentionPairStrategy.encoder]:
            self.create_contrastive_pairs()

    def create_contrastive_pairs(self):
        # todo custom case if max n is not provided
        if self.max_pairs is None:
            if self.split == Split.train:
                self.max_pairs = DEFAULT_TRAIN
            else:
                self.max_pairs = DEFAULT_DEV

        # compute max per dataset component
        if self.ratio > -1:
            positive_n_max = self.max_pairs * len(self.dataset_components) // (1 + self.ratio)
            ratio = self.ratio
        else:
            # make a default ratio 20 as in SOTA
            positive_n_max = self.max_pairs * len(self.dataset_components) // (1 + DEFAULT_RATIO)
            ratio = DEFAULT_RATIO

        used_up_n = {d: 0 for d in self.dataset_components}
        all_mention_pairs = {d: {} for d in self.dataset_components}

        for topic_id, clusters in self.topics.topic_clusters.items():
            dataset = self.topics.topics_to_datasets[topic_id]
            if used_up_n[dataset] >= positive_n_max:
                break

            shuffled_clusters = list(clusters.items())
            random.shuffle(shuffled_clusters)
            shuffled_clusters = dict(shuffled_clusters)

            topic_embed_df, sim_df = self.encode_mentions(topic_id)

            for c_id, mentions in shuffled_clusters.items():
                if used_up_n[dataset] >= positive_n_max:
                    break

                # compute the threshold
                # some mentions could have been illuminated if no vector was computed for them, e.g., after the stopword removal in the embedding method
                mentions_dict = {m["mention_id"]: m for m in mentions if m["mention_id"] in topic_embed_df.index.to_list()}
                m_ids = list(mentions_dict)

                cluster_df = topic_embed_df.loc[m_ids]
                sim = cosine_similarity(cluster_df.values)
                threshold = float(np.mean(sim))

                # compute centroid
                centroid = np.mean(cluster_df.values, axis=0)
                sim_centroid = cosine_similarity(centroid, cluster_df.values)[0]
                sim_series = pd.Series(sim_centroid, index=cluster_df.index)

                # choose two target mentions: one close to centroid, one - faw away

                for target_mention_id in [sim_series.idxmax(), sim_series.idxmin()]:
                    target_vector = cluster_df.loc[target_mention_id]
                    sim_target = cosine_similarity(target_vector, topic_embed_df.values)[0]
                    sim_series_target = pd.Series(sim_target, index=topic_embed_df.index)
                    sim_series_target = sim_series_target.drop(target_mention_id)

                    pos_easy = sim_series_target[sim_series_target >= threshold + DELTA].index.to_list()
                    pos_hard = sim_series_target[sim_series_target <= threshold - DELTA].index.to_list()
                    neg_hard = sim_series_target[sim_series_target >= threshold + DELTA].index.to_list()
                    neg_easy = sim_series_target[sim_series_target <= threshold - DELTA].index.to_list()
                    # neg_easy = sim_series_target[sim_series_target <= threshold - DELTA].sort_values(ascending=False).index.to_list()
                    num_pairs_per_pos_type = min(len(pos_easy), len(pos_hard))
                    num_pairs_per_neg_type = min(len(neg_easy), len(neg_hard))

                    if num_pairs_per_pos_type == 0 or num_pairs_per_neg_type:
                        # can't support the balance
                        continue

                    if num_pairs_per_pos_type * ratio > num_pairs_per_neg_type:
                        # can't support the balance
                        continue

                    if c_id not in all_mention_pairs[dataset]:
                        all_mention_pairs[dataset][c_id] = {"pos_easy": [], "pos_hard": [], "neg_easy": [],
                                                            "neg_hard": []}
                    all_mention_pairs[dataset][c_id]["pos_easy"] = uCDCRDataSet._make_pairs_with_target(target_mention_id, pos_easy[:num_pairs_per_pos_type], mentions_dict)
                    all_mention_pairs[dataset][c_id]["pos_hard"] = uCDCRDataSet._make_pairs_with_target(target_mention_id, pos_hard[:num_pairs_per_pos_type], mentions_dict)
                    all_mention_pairs[dataset][c_id]["neg_easy"] = uCDCRDataSet._make_pairs_with_target(target_mention_id, neg_easy[:num_pairs_per_neg_type], mentions_dict)
                    all_mention_pairs[dataset][c_id]["neg_hard"] = uCDCRDataSet._make_pairs_with_target(target_mention_id, neg_hard[:num_pairs_per_neg_type], mentions_dict)

                used_up_n[dataset] += len(all_mention_pairs[dataset][c_id]["pos_easy"])

        for clusters_per_dataset in list(all_mention_pairs.values()):
            for pair_types in list(clusters_per_dataset.values()):
                self.positive_pairs.extend(pair_types["pos_easy"])
                self.positive_pairs.extend(pair_types["pos_hard"])
                self.negative_pairs.extend(pair_types["neg_easy"])
                self.negative_pairs.extend(pair_types["neg_hard"])


    @classmethod
    def _make_pairs_with_target(cls, m_target_id, mention_ids, mention_dict):
        return [(mention_dict[m_target_id], mention_dict[m_id]) for m_id in mention_ids]

    def encode_mentions(self, topic_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.type_of_pairs == MentionPairStrategy.lemma:
            embed_df, sim_df = self._encode_lemmas(topic_id)

        elif self.type_of_pairs == MentionPairStrategy.embedding:
            embed_df, sim_df = self._encode_embeddings(topic_id)

        elif self.type_of_pairs == MentionPairStrategy.encoder:
            embed_df, sim_df = self._encode_sentence_transformer(topic_id)
        else:
            raise NotImplementedError(f"A method for the mention pair creation {self.type_of_pairs} is not implemented.")
        return embed_df, sim_df


    def _encode_lemmas(self, topic_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        texts, ids = [], []

        for m in self.topics.topics_dict[topic_id]:
            tokens_clean = self._remove_stopwords(m["tokens_text"])
            if not len(tokens_clean):
                continue

            if m["mention_head_lemma"] not in tokens_clean:
                tokens_clean.append(m["mention_head_lemma"])
            texts.append(" ".join(tokens_clean))
            ids.append(m["mention_id"])

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
        texts, ids, heads = [], [], []

        for m in self.topics.topics_dict[topic_id]:
            tokens_clean = self._remove_stopwords(m["tokens_text"])
            if not len(tokens_clean):
                continue
            texts.append(tokens_clean)
            ids.append(m["mention_id"])
            heads.append(m["mention_head"])

        embeddings = list()
        for tokens, head in zip(texts, heads):
            vector = self._encode_sentence(tokens, head)
            embeddings.append(vector)

        embeddings = np.vstack(embeddings)
        embed_df = pd.DataFrame(embeddings, index=ids)
        sim = cosine_similarity(embeddings)

        sim_df = pd.DataFrame(sim, index=ids, columns=ids)
        return embed_df, sim_df

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
        model_fasttext = get_fasttext_model()
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
        texts, ids = [], []
        for m in self.topics.topics_dict[topic_id]:
            texts.append(m["tokens_str"])
            ids.append(m["mention_id"])

        embeddings = model.encode(
            texts,
            batch_size=ENCODE_BATCH,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        embed_df = pd.DataFrame(embeddings, index=ids)
        sim = cosine_similarity(embeddings)

        sim_df = pd.DataFrame(sim, index=ids, columns=ids)
        return embed_df, sim_df


    def create_all_pairs_capped(self):
        """
        Created all mention pairs or the upper triangle, caps them, and only takes up as much as the limit allows
        """

        # compute max per dataset component
        if self.ratio > -1:
            positive_n_max = self.max_pairs * len(self.dataset_components) // (1 + self.ratio)
        else:
            # make a default ratio 20 as in SOTA
            positive_n_max = self.max_pairs * len(self.dataset_components) // (1 + DEFAULT_RATIO)

        used_up_n = {d: 0 for d in self.dataset_components}
        not_used_mentions_pairs = {d: [] for d in self.dataset_components}

        for topic_id, clusters in self.topics.topic_clusters.items():
            shuffled_clusters = list(clusters.items())
            random.shuffle(shuffled_clusters)
            shuffled_clusters = dict(shuffled_clusters)

            for c_id, mentions in shuffled_clusters.items():
                dataset = mentions[0]["dataset"]
                if used_up_n[dataset] == positive_n_max:
                    break

                triangle_n = len(mentions) * (len(mentions) - 1) // 2
                # limit positives to sqrt 6
                max_local_pairs = min(triangle_n, int(6 * math.sqrt(triangle_n)))

                # create the upper triangle
                triangle_pairs = list(combinations(mentions, 2))
                random.shuffle(triangle_pairs)

                # limit to what is still there to take up
                max_to_take = min(max_local_pairs, positive_n_max - used_up_n[dataset])
                self.positive_pairs.extend(triangle_pairs[:max_to_take])
                not_used_mentions_pairs[dataset].extend(triangle_pairs[max_to_take:])
                used_up_n[dataset] += max_to_take

        for d in self.dataset_components:
            # if we didn't get enough positive pairs per dataset, take the missing pairs from the not used pairs
            if used_up_n[d] < positive_n_max:
                diff = positive_n_max - used_up_n[d]
                self.positive_pairs.extend(not_used_mentions_pairs[d][:diff])

        negative_n_max = (self.max_pairs - positive_n_max * len(self.dataset_components)) // len(self.dataset_components)
        self._create_capped_negatives({d: negative_n_max for d in self.dataset_components})


    def create_all_pairs(self) -> Dict[str, int]:
        """
        Creates all permutations of the pairs in the upper triangle and splits into positive and negatives
        """
        self.positive_pairs, self.negative_pairs = [], []
        positive_counter = {d: 0 for d in self.dataset_components}
        for topic in self.topics.topics_dict.values():
            triangle_pairs = list(combinations(topic.mentions, 2))
            for mention1, mention2 in triangle_pairs:
                if mention1.coref_chain == mention2.coref_chain:
                    self.positive_pairs.append((mention1, mention2))
                    positive_counter[mention1["dataset"]] += 1
                else:
                    self.negative_pairs.append((mention1, mention2))
        return positive_counter


    def create_all_pairs_negatives_capped(self):
        """
        Just in case we want all positives but not overwhelm with negatives. Does not balance per dataset,
        so the number of negatives will be proportional to the number of positives per dataset.
        """
        positive_counter_datasets = self.create_all_pairs()
        ratio = self.ratio if self.ratio > -1 else DEFAULT_RATIO
        negative_counter_datasets = {d: ratio * pos for d, pos in positive_counter_datasets}
        self._create_capped_negatives(negative_counter_datasets)


    def _create_capped_negatives(self, max_topic_negatives: dict):
        """
        Creates all permutations of the pairs in the upper triangle and splits into positive and negatives
        """
        self.negative_pairs = []
        for topic in self.topics.topics_dict.values():
            negative_pairs_topic = []
            triangle_pairs = list(combinations(topic.mentions, 2))
            for mention1, mention2 in triangle_pairs:
                if mention1.coref_chain != mention2.coref_chain:
                    negative_pairs_topic.append((mention1, mention2))
                    if len(negative_pairs_topic) == max_topic_negatives[mention1["dataset"]]:
                        break

            self.negative_pairs.extend(negative_pairs_topic)




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
