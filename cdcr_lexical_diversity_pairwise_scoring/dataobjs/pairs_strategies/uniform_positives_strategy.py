import math
import random
from itertools import combinations


class UniformPositivesStrategy:
    @staticmethod
    def create_pairs(
        topics,
        max_total_pairs: int | None,
        negatives_per_positive: int,
        dataset_names: list[str],
    ) -> tuple[list, list]:
        """Created all mention pairs or the upper triangle, caps them, and only takes up as much as the limit allows"""
        # compute max per dataset component
        positive_pairs = []

        positive_n_max = UniformPositivesStrategy._get_max_positives_per_dataset_uniformly(
            negatives_per_positive,
            max_total_pairs,
            len(dataset_names),
        )
        used_up_n = dict.fromkeys(dataset_names, 0)
        not_used_mentions_pairs = {d: [] for d in dataset_names}

        shuffled_clusters = list(topics.clusters.items())
        random.shuffle(shuffled_clusters)
        shuffled_clusters = dict(shuffled_clusters)

        used_topics = {}
        for c_id, mentions in shuffled_clusters.items():
            topic_mentions_dict = {}
            # Some clusters are cross-subtopic, so if the topic level is subtopic,
            #   we need to make sure that the positive pairs will be created on the level
            #   that we got from config
            for mention in mentions:
                topic_id = topics.mention_to_topic[mention.mention_id]
                if topic_id not in topic_mentions_dict:
                    topic_mentions_dict[topic_id] = []
                topic_mentions_dict[topic_id].append(mention)

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
                positive_pairs.extend(triangle_pairs[:max_to_take])
                not_used_mentions_pairs[dataset].extend(triangle_pairs[max_to_take:])
                used_up_n[dataset] += max_to_take

                if topic_id not in used_topics:
                    used_topics[topic_id] = list()

                used_topics[topic_id].append(set(m.mention_id for m in topic_mentions))

        for d in dataset_names:
            # if we didn't get enough positive pairs per dataset, take the missing pairs from the not used pairs
            if used_up_n[d] < positive_n_max:
                diff = positive_n_max - used_up_n[d]
                positive_pairs.extend(not_used_mentions_pairs[d][:diff])

        negative_n_max = (max_total_pairs - positive_n_max * len(dataset_names)) // len(dataset_names)
        negative_pairs = UniformPositivesStrategy._create_sampled_capped_negatives(
            topics,
            {dataset_name: negative_n_max for dataset_name in dataset_names},
            used_topics,
            negatives_per_positive,
        )
        return positive_pairs, negative_pairs

    @staticmethod
    def _get_max_positives_per_dataset_uniformly(
        max_total_pairs: int,
        negatives_per_positive: int,
        n_datasets: int,
    ) -> int:
        positive_n_max = max_total_pairs // (1 + negatives_per_positive) // n_datasets
        return positive_n_max

    @staticmethod
    def _create_sampled_capped_negatives(
        topics,
        max_topic_negatives: dict,
        topic_cluster_membership: dict,
        negatives_per_positive: int,
    ):
        """Stratified negative creation based on the positive pairs"""
        negative_pairs_per_dataset = {d: [] for d in list(max_topic_negatives)}
        negative_pairs = []

        for topic_id, topic in topics.topics_dict.items():
            dataset = topics.topics_to_datasets[topic_id]

            if topic_id not in topic_cluster_membership:
                continue

            mention_ids = {mention.mention_id for mention in topic.mentions}

            for cluster in topic_cluster_membership[topic_id]:
                # all other mentions outside a cluster are negatives
                negative_candidates = mention_ids - cluster

                number_of_positive_pairs = (
                    len(cluster) * (len(cluster) - 1) // 2
                )  # Essentially everybody with everybody except with itself.
                required_number_of_negative_pairs = number_of_positive_pairs * negatives_per_positive

                # we will need a permutation of the positive mentions with some selected negatives, so the number is the target number of pairs given a cluster divided by the cluster size
                required_number_of_negative_candidates = required_number_of_negative_pairs // len(cluster)
                selected_negative_candidates = random.sample(
                    list(negative_candidates),
                    required_number_of_negative_candidates,
                )
                negative_pairs = [(a, b) for a in cluster for b in selected_negative_candidates]
                negative_pairs_per_dataset[dataset].extend(negative_pairs)

        for dataset, negatives in negative_pairs_per_dataset.items():
            random.shuffle(negatives)
            negative_pairs.extend(
                negatives[: max_topic_negatives[dataset]]
            )  # TODO: why do we extend and then sample here? are we supposed to do that?

        return negative_pairs
