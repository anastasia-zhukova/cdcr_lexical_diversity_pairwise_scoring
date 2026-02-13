import random
from collections import defaultdict
from dataclasses import dataclass, field
from string import punctuation

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from staticvectors import StaticVectors

from cdcr_lexical_diversity_pairwise_scoring import logger
from cdcr_lexical_diversity_pairwise_scoring.constants import *
from cdcr_lexical_diversity_pairwise_scoring.dataobjs.pairs_strategies.enum import MentionPairStrategy


nltk.download("stopwords")
_STOP_WORDS = frozenset(stopwords.words("english"))
TOKEN_FILTER = _STOP_WORDS | frozenset(punctuation)
MODEL_FASTTEXT = StaticVectors(EMBEDDING)
MODEL_SENTENCE_TRANSFORMER = SentenceTransformer(SENT_TRANSFOMER)


# TODO: rename
@dataclass
class MyDataclass:
    easy_positive_pairs: set[str] = field(default_factory=set)
    easy_negative_pairs: set[str] = field(default_factory=set)
    hard_positive_pairs: set[str] = field(default_factory=set)
    hard_negative_pairs: set[str] = field(default_factory=set)


class ContrastiveStrategy:
    @staticmethod
    def _get_max_positives_per_dataset_uniformly(
        max_total_pairs: int,
        negatives_per_positive: int,
        n_datasets: int,
    ) -> int:
        positive_n_max = max_total_pairs // (1 + negatives_per_positive) // n_datasets
        return positive_n_max

    @staticmethod
    def _make_pairs_with_target(
        mention_target_id: int,
        mention_ids: list[int],
        mention_dict: dict,
        max_pairs: int,
    ) -> set:
        if len(mention_ids) == 0 or mention_ids is None:
            return set()

        target_mention = mention_dict[mention_target_id]

        text_unique: dict[str, list] = defaultdict(list)
        for mention_id in mention_ids:
            text_unique[mention_dict[mention_id].tokens_str].append(mention_id)

        text_unique = dict()
        if len(text_unique) > max_pairs:
            # take mentions with unique wording
            used_ids = [
                mention_ids[0]
                for _, mention_ids in sorted(text_unique.items(), key=lambda key_value: len(key_value[1]))[:max_pairs]
            ]
        else:
            used_ids = mention_ids[:max_pairs]

        return {(target_mention, mention_dict[m_id]) for m_id in used_ids}

    @staticmethod
    def create_pairs(
        topics,
        max_total_pairs: int,
        pairs_strategy: MentionPairStrategy,
        negatives_per_positive: int,
        dataset_names: list[str],
        min_std: float,
    ) -> tuple[list, list]:
        """Builds mention pairs on the contrastive neighbour principle"""

        max_positives_per_dataset = ContrastiveStrategy._get_max_positives_per_dataset_uniformly(
            negatives_per_positive,
            max_total_pairs,
            len(dataset_names),
        )

        used_up_n = dict.fromkeys(dataset_names, 0)
        all_mention_pairs = {d: {} for d in dataset_names}

        shuffled_topics = list(topics.topic_clusters.items())
        random.shuffle(shuffled_topics)
        shuffled_topics = dict(shuffled_topics)
        next_report_milestone = 0.1

        for topic_id, clusters in shuffled_topics.items():
            dataset = topics.topics_to_datasets[topic_id]
            if used_up_n[dataset] >= max_positives_per_dataset:
                continue

            shuffled_clusters = list(clusters.items())
            random.shuffle(shuffled_clusters)
            shuffled_clusters = dict(shuffled_clusters)

            logger.debug(f"Encoding topic {topic_id}.")
            topic_embed_df, sim_df = ContrastiveStrategy.encode_mentions(pairs_strategy, topics, topic_id)
            mentions_topic_dict = {mention.mention_id: mention for mention in topics.topics_dict[topic_id].mentions}

            for c_id, mentions in shuffled_clusters.items():
                # exclude singletons from being target mentions
                if len(mentions) == 1:
                    continue

                if used_up_n[dataset] >= max_positives_per_dataset:
                    break

                # compute the threshold
                # some mentions could have been illuminated if no vector was computed for them,
                # e.g. after the stopword removal in the embedding method
                allowed_mention_ids = topic_embed_df.index.to_list()
                mention_ids = [mention.mention_id for mention in mentions if mention.mention_id in allowed_mention_ids]
                if len(mention_ids) == 0:
                    continue

                cluster_df = topic_embed_df.loc[mention_ids]
                cosine_similarity_ = cosine_similarity(cluster_df.values)
                threshold = float(np.mean(cosine_similarity_))
                std = float(np.std(cosine_similarity_))

                if std < min_std:
                    # There are no hard positives in this case, so all positives are considered easy ones.
                    delta = MIN_STD / DENOM_DELTA
                    target_mention_id = cluster_df.index[0]
                    target_vector = cluster_df.iloc[0]
                    sim_target = cosine_similarity(target_vector.reshape(1, -1), topic_embed_df.values)[0]
                    sim_series_target = pd.Series(sim_target, index=topic_embed_df.index).drop(target_mention_id)

                    easy_positives_ids = cluster_df.index[1:].to_list()

                    # Find hard negatives
                    hard_negatives_ids = (
                        # really no variation, so to get smth in, subtract delta
                        list(
                            set(sim_series_target[sim_series_target >= threshold - delta].index) - set(mention_ids),
                        )
                        if threshold >= 1.0
                        else list(
                            set(sim_series_target[sim_series_target >= threshold + delta].index) - set(mention_ids),
                        )
                    )

                    # form easies that are more remote than usually
                    easy_negatives_ids = list(
                        set(sim_series_target[sim_series_target <= threshold - MIN_STD].index) - set(mention_ids)
                    )

                    num_pairs_per_positive_type = len(easy_positives_ids)
                    num_pairs_per_negative_type = num_pairs_per_positive_type * negatives_per_positive

                    # use up all positives
                    # TODO: rewrite to a separate function
                    easy_positive_pairs = ContrastiveStrategy._make_pairs_with_target(
                        target_mention_id,
                        easy_positives_ids,
                        mentions_topic_dict,
                        max_pairs=num_pairs_per_positive_type,
                    )
                    # first use up all hard negatives, then the remaining take form easy negatives
                    hard_negative_pairs = ContrastiveStrategy._make_pairs_with_target(
                        target_mention_id,
                        hard_negatives_ids,
                        mentions_topic_dict,
                        max_pairs=num_pairs_per_negative_type,
                    )
                    easy_negative_pairs = ContrastiveStrategy._make_pairs_with_target(
                        target_mention_id,
                        easy_negatives_ids,
                        mentions_topic_dict,
                        max_pairs=num_pairs_per_negative_type - len(hard_negative_pairs),
                    )

                    used_up_n[dataset] -= all_mention_pairs[dataset][c_id].easy_positive_pairs
                    # TODO: check and remove update if there is no update actually.
                    all_mention_pairs[dataset][c_id] = MyDataclass()
                    all_mention_pairs[dataset][c_id].easy_positive_pairs.update(easy_positive_pairs)
                    all_mention_pairs[dataset][c_id].easy_negative_pairs.update(easy_negative_pairs)
                    all_mention_pairs[dataset][c_id].hard_negative_pairs.update(hard_negative_pairs)

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

                        easy_positives_ids = list(
                            set(sim_series_target[sim_series_target >= threshold + delta].index).intersection(
                                set(mention_ids),
                            ),
                        )
                        hard_positives_ids = list(
                            set(sim_series_target[sim_series_target <= threshold - delta].index).intersection(
                                set(mention_ids),
                            ),
                        )
                        easy_negatives_ids = list(
                            set(sim_series_target[sim_series_target <= threshold - delta].index) - set(mention_ids),
                        )
                        hard_negatives_ids = list(
                            set(sim_series_target[sim_series_target >= threshold + delta].index) - set(mention_ids),
                        )

                        # we use all positives and sample negatives
                        num_pairs_per_positive_type = len(easy_positives_ids) + len(hard_positives_ids)
                        num_pairs_per_negative_type = num_pairs_per_positive_type * negatives_per_positive

                        if num_pairs_per_negative_type > len(easy_negatives_ids) + len(hard_negatives_ids):
                            # not enough negatives
                            continue

                        if c_id not in all_mention_pairs[dataset]:
                            all_mention_pairs[dataset][c_id] = MyDataclass()

                        # use up all positives
                        easy_positive_pairs = ContrastiveStrategy._make_pairs_with_target(
                            target_mention_id,
                            easy_positives_ids,
                            mentions_topic_dict,
                            max_pairs=num_pairs_per_positive_type,
                        )
                        hard_positive_pairs = ContrastiveStrategy._make_pairs_with_target(
                            target_mention_id,
                            hard_negatives_ids,
                            mentions_topic_dict,
                            max_pairs=num_pairs_per_positive_type,
                        )
                        # first use up all hard negatives, then the remaining take form easy negatives
                        hard_negative_pairs = ContrastiveStrategy._make_pairs_with_target(
                            target_mention_id,
                            hard_negatives_ids,
                            mentions_topic_dict,
                            max_pairs=num_pairs_per_negative_type,
                        )
                        easy_negative_pairs = ContrastiveStrategy._make_pairs_with_target(
                            target_mention_id,
                            easy_negatives_ids,
                            mentions_topic_dict,
                            max_pairs=num_pairs_per_negative_type - len(hard_negative_pairs),
                        )

                        used_up_n[dataset] -= (
                            all_mention_pairs[dataset][c_id].easy_positive_pairs
                            + all_mention_pairs[dataset][c_id].hard_positive_pairs
                        )

                        all_mention_pairs[dataset][c_id].easy_positive_pairs.update(easy_positive_pairs)
                        all_mention_pairs[dataset][c_id].easy_negative_pairs.update(easy_negative_pairs)
                        all_mention_pairs[dataset][c_id].hard_positive_pairs.update(hard_positive_pairs)
                        all_mention_pairs[dataset][c_id].hard_negative_pairs.update(hard_negative_pairs)

                    used_up_n[dataset] += (
                        all_mention_pairs[dataset][c_id].easy_positive_pairs
                        + all_mention_pairs[dataset][c_id].hard_positive_pairs
                    )

                if used_up_n[dataset] / max_positives_per_dataset >= next_report_milestone:
                    logger.info(
                        f"Collected at least {next_report_milestone * 100:1f}% of the {max_positives_per_dataset} positive pairs for {dataset}.",
                    )
                    next_report_milestone += 0.1

        positive_pairs: list[tuple] = []
        negative_pairs: list[tuple] = []

        for dataset, clusters_per_dataset in all_mention_pairs.items():
            for pair_types in clusters_per_dataset.values():
                positive_pairs.extend(pair_types.positive_easy)
                positive_pairs.extend(pair_types.positive_hard)
                negative_pairs.extend(pair_types.negative_easy)
                negative_pairs.extend(pair_types.negative_hard)

        return positive_pairs, negative_pairs

    @staticmethod
    def encode_mentions(
        pairs_strategy: MentionPairStrategy,
        topics,
        topic_id: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if pairs_strategy == MentionPairStrategy.tfidf:
            embed_df, sim_df = ContrastiveStrategy._encode_tfidf(topics, topic_id)

        elif pairs_strategy == MentionPairStrategy.embedding:
            embed_df, sim_df = ContrastiveStrategy._encode_embeddings(topics, topic_id)

        elif pairs_strategy == MentionPairStrategy.encoder:
            embed_df, sim_df = ContrastiveStrategy._encode_sentence_transformer(topics, topic_id)
        else:
            raise NotImplementedError(
                f"A method for the mention pair creation {pairs_strategy} is not implemented.",
            )
        return embed_df, sim_df

    @staticmethod
    def _encode_tfidf(
        topics,
        topic_id: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        mentions = topics.topics_dict[topic_id].mentions
        texts: list[str] = []
        mention_ids: list[str] = []

        for mention in mentions:
            tokens = ContrastiveStrategy._remove_stopwords(mention.tokens_text)
            if len(tokens) == 0:
                continue

            head = mention.mention_head_lemma
            if head is not None and head not in tokens:
                tokens.append(head)

            texts.append(" ".join(tokens))
            mention_ids.append(mention.mention_id)

        vectorizer = TfidfVectorizer(lowercase=True)
        embeddings = vectorizer.fit_transform(texts)
        embeddings_df = pd.DataFrame.sparse.from_spmatrix(
            embeddings,
            index=mention_ids,
            columns=vectorizer.get_feature_names_out(),
        )
        similarity = cosine_similarity(embeddings)
        similarity_df = pd.DataFrame(similarity, index=mention_ids, columns=mention_ids)
        return embeddings_df, similarity_df

    @staticmethod
    def _encode_embeddings(
        topics,
        topic_id: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        cached_path = PROJECT_ROOT / "resources" / f"{EMBEDDING.replace('/', '_')}.h5"
        mentions = topics.topics_dict[topic_id].mentions

        mention_ids: list[str] = []
        tokens_by_id: dict[str, list[str]] = {}
        heads_by_id: dict[str, str] = {}

        for mention in mentions:
            tokens = ContrastiveStrategy._remove_stopwords(mention.tokens_text)
            if len(tokens) == 0:
                continue

            mention_id = mention.mention_id
            mention_ids.append(mention_id)
            tokens_by_id[mention_id] = tokens
            heads_by_id[mention_id] = mention.mention_head

        if cached_path.exists():
            logger.info(f"Loading existing embeddings dataframe from {cached_path}.")
            existing_embeddings_df = pd.read_hdf(cached_path, key="df")
        else:
            existing_embeddings_df = pd.DataFrame()

        cache_index = set(existing_embeddings_df.index)
        mention_ids_to_encode: list[str] = [mention_id for mention_id in mention_ids if mention_id not in cache_index]

        if len(mention_ids_to_encode) == 0:
            embeddings_df = existing_embeddings_df
        else:
            new_vectors = [
                ContrastiveStrategy._encode_sentence(tokens_by_id[mention_id], heads_by_id[mention_id])
                for mention_id in mention_ids_to_encode
            ]
            new_embeddings_df = pd.DataFrame(
                np.vstack(new_vectors),
                index=mention_ids_to_encode,
            )
            embeddings_df = pd.concat([existing_embeddings_df, new_embeddings_df], axis=0)
            embeddings_df.to_hdf(cached_path, key="df", mode="w")  # "w" = overwrite, "a" = append

        topic_embeddings_df = embeddings_df.loc[mention_ids]
        similarity = cosine_similarity(topic_embeddings_df.values)

        similarity_df = pd.DataFrame(similarity, index=topic_embeddings_df.index, columns=topic_embeddings_df.index)
        return topic_embeddings_df, similarity_df

    @staticmethod
    def _remove_stopwords(tokens: list[str]) -> list[str]:
        return [t for t in tokens if t not in TOKEN_FILTER]

    @staticmethod
    def _encode_sentence(
        tokens: list[str],
        head_token: str,
        head_weight: float = 2.0,
    ) -> np.ndarray:
        """Encode a list of mention tokens with static embeddings"""
        vectors = MODEL_FASTTEXT.embeddings(tokens)

        weights = np.array(
            [head_weight if t == head_token else 1.0 for t in tokens],
            dtype=np.float32,
        )

        embedding = np.average(vectors, axis=0, weights=weights)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.astype(np.float32)

    @staticmethod
    def _encode_sentence_transformer(
        topics,
        topic_id: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        cached_path = PROJECT_ROOT / "resources" / f"{SENT_TRANSFOMER.replace('/', '_')}.h5"
        mentions = topics.topics_dict[topic_id].mentions

        mention_ids: list[str] = []
        text_by_id: dict[str, str] = {}

        for mention in mentions:
            mention_id = mention.mention_id
            mention_ids.append(mention_id)
            text_by_id[mention_id] = mention.tokens_str

        if cached_path.exists():
            logger.info(f"Loading existing embeddings dataframe from {cached_path}.")
            existing_embeddings_df = pd.read_hdf(cached_path, key="df")
        else:
            existing_embeddings_df = pd.DataFrame()

        cache_index = set(existing_embeddings_df.index)
        mention_ids_to_encode: list[str] = [mention_id for mention_id in mention_ids if mention_id not in cache_index]

        if len(mention_ids_to_encode) == 0:
            embeddings_df = existing_embeddings_df
        else:
            texts = [text_by_id[mention_id] for mention_id in mention_ids_to_encode]
            new_vectors = MODEL_SENTENCE_TRANSFORMER.encode(
                texts,
                batch_size=ENCODE_BATCH,
                convert_to_numpy=True,
                show_progress_bar=True,
                normalize_embeddings=True,
            )
            new_embeddings_df = pd.DataFrame(
                new_vectors,
                index=mention_ids_to_encode,
            )
            embeddings_df = pd.concat([existing_embeddings_df, new_embeddings_df], axis=0)
            embeddings_df.to_hdf(cached_path, key="df", mode="w")  # "w" = overwrite, "a" = append

        topic_embeddings_df = embeddings_df.loc[mention_ids]
        similarity = cosine_similarity(topic_embeddings_df.values)
        similarity_df = pd.DataFrame(similarity, index=topic_embeddings_df.index, columns=topic_embeddings_df.index)
        return topic_embeddings_df, similarity_df
