import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations

from cdcr_lexical_diversity_pairwise_scoring import logger
from cdcr_lexical_diversity_pairwise_scoring.dataobjs.mention_data import MentionuCDCR
from cdcr_lexical_diversity_pairwise_scoring.dataobjs.topics import Topics


@dataclass
class Cluster:
    mention_ids_in_cluster: set[str] = field(default_factory=set)
    mentions_in_cluster: list[MentionuCDCR] = field(default_factory=list)


class UniformPositivesStrategy:
    @staticmethod
    def create_pairs(
        topics: Topics,
        max_total_pairs: int,
        negatives_per_positive: int,
        dataset_names: list[str],
    ) -> tuple[
        list[tuple[MentionuCDCR, MentionuCDCR]],
        list[tuple[MentionuCDCR, MentionuCDCR]],
    ]:
        """Created all mention pairs or the upper triangle, caps them, and only takes up as much as the limit allows"""
        # compute max per dataset component
        positive_pairs = []

        positive_n_max = UniformPositivesStrategy._get_max_positives_per_dataset_uniformly(
            max_total_pairs,
            negatives_per_positive,
            len(dataset_names),
        )
        used_up_n = {dataset_name: 0 for dataset_name in dataset_names}
        not_used_mentions_pairs = {d: [] for d in dataset_names}

        shuffled_clusters = list(topics.clusters.items())
        random.shuffle(shuffled_clusters)

        clusters_in_topics = defaultdict(list)
        for _, cluster_mentions in shuffled_clusters:
            topic_id_to_mentions = defaultdict(list)
            # Some clusters are cross-subtopic, so if the topic level is subtopic,
            #   we need to make sure that the positive pairs will be created on the level
            #   that we got from config
            for mention in cluster_mentions:
                topic_id = topics.mention_to_topic[mention.mention_id]
                topic_id_to_mentions[topic_id].append(mention)
            topic_id_to_mentions = dict(topic_id_to_mentions)

            for topic_id, topic_mentions in topic_id_to_mentions.items():
                if len(topic_mentions) == 1:
                    continue

                dataset = topic_mentions[0].dataset
                if used_up_n[dataset] >= positive_n_max:
                    continue

                triangle_n = len(topic_mentions) * (len(topic_mentions) - 1) // 2
                # limit positives to sqrt 6
                max_local_pairs = min(triangle_n, int(6 * math.sqrt(triangle_n)))

                # create the upper triangle
                triangle_pairs = list(combinations(topic_mentions, 2))
                random.shuffle(triangle_pairs)

                # limit to what is still there to take up
                max_to_take = min(max_local_pairs, positive_n_max - used_up_n[dataset])
                positive_pairs.extend(triangle_pairs[:max_to_take])
                not_used_mentions_pairs[dataset].extend(triangle_pairs[max_to_take:])
                used_up_n[dataset] += max_to_take

                # Remove duplicates to form a cluster quazi-set
                cluster = Cluster()
                for mention in topic_mentions:
                    if mention.mention_id not in cluster.mention_ids_in_cluster:
                        cluster.mention_ids_in_cluster.add(mention.mention_id)
                        cluster.mentions_in_cluster.append(mention)

                clusters_in_topics[topic_id].append(cluster)

        for d in dataset_names:
            # if we didn't get enough positive pairs per dataset, take the missing pairs from the not used pairs
            if used_up_n[d] < positive_n_max:
                diff = positive_n_max - used_up_n[d]
                positive_pairs.extend(not_used_mentions_pairs[d][:diff])

        negative_n_max = (max_total_pairs - positive_n_max * len(dataset_names)) // len(dataset_names)
        negative_pairs = UniformPositivesStrategy._create_sampled_capped_negatives(
            topics,
            {dataset_name: negative_n_max for dataset_name in dataset_names},
            clusters_in_topics,
            negatives_per_positive,
        )
        return positive_pairs, negative_pairs

    @staticmethod
    def _get_max_positives_per_dataset_uniformly(
        max_total_pairs: int,
        negatives_per_positive: int,
        n_datasets: int,
    ) -> int:
        return max_total_pairs // (1 + negatives_per_positive) // n_datasets

    @staticmethod
    def _create_sampled_capped_negatives(
        topics,
        max_topic_negatives: dict,
        clusters_in_topics: dict[int, list[Cluster]],
        negatives_per_positive: int,
    ) -> list[tuple[MentionuCDCR, MentionuCDCR]]:
        """Stratified negative creation based on the positive pairs"""
        negative_pairs_per_dataset = {d: [] for d in list(max_topic_negatives)}
        negative_pairs = []

        for topic_id, topic in topics.topics_dict.items():
            dataset = topics.topics_to_datasets[topic_id]

            if topic_id not in clusters_in_topics:
                continue

            mention_ids_to_mentions = {mention.mention_id: mention for mention in topic.mentions}

            for cluster in clusters_in_topics[topic_id]:
                # all other mentions outside a cluster are negatives
                negative_candidates_ids = set(mention_ids_to_mentions.keys()) - cluster.mention_ids_in_cluster

                number_of_positive_pairs = (
                    len(cluster.mentions_in_cluster) * (len(cluster.mentions_in_cluster) - 1) // 2
                )  # Essentially everybody with everybody except with itself.
                required_number_of_negative_pairs = number_of_positive_pairs * negatives_per_positive

                # we will need a permutation of the positive mentions with some selected negatives,
                #   so the number is the target number of pairs given a cluster divided by the cluster size
                required_number_of_negative_candidates = required_number_of_negative_pairs // len(
                    cluster.mentions_in_cluster
                )
                logger.debug(
                    f"Require {required_number_of_negative_candidates} negative candidates,"
                    f" have {len(negative_candidates_ids)}"
                )
                # We will sample from list of ids, then we will convert back to mentions via a dict
                selected_negative_candidates_ids = random.sample(
                    list(negative_candidates_ids),
                    # TODO: we have to do this to avoid situation when there is not enough candidates,
                    #   but this leads to "undertaking" the objects.
                    min(len(negative_candidates_ids), required_number_of_negative_candidates),
                )
                selected_negative_candidates = [
                    mention_ids_to_mentions[mention_id] for mention_id in selected_negative_candidates_ids
                ]
                negative_pairs_local = [
                    (a, b) for a in cluster.mentions_in_cluster for b in selected_negative_candidates
                ]
                negative_pairs_per_dataset[dataset].extend(negative_pairs_local)

        for dataset, negatives in negative_pairs_per_dataset.items():
            random.shuffle(negatives)
            negative_pairs.extend(
                negatives[: max_topic_negatives[dataset]]
            )  # TODO: why do we sample here? Seems like there is not point for this.

        return negative_pairs
