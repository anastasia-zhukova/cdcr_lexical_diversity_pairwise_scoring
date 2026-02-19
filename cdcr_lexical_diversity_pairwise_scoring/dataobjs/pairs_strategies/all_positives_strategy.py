import random
from itertools import combinations

from cdcr_lexical_diversity_pairwise_scoring import logger


class AllPositivesStrategy:
    @staticmethod
    def create_pairs(
        topics,
        negatives_per_positive: int,
        dataset_names: list[str],
    ) -> tuple[list, list]:
        # TODO: we first make all possible negative pairs, then we EXTEND them with sampled ones: this sounds wrong.
        all_positives, _, number_of_positives_per_dataset = AllPositivesStrategy._create_all_possible_pairs(
            topics,
            dataset_names,
        )
        number_of_allowed_negatives_per_dataset = {
            dataset_name: number_of_positives * negatives_per_positive
            for dataset_name, number_of_positives in number_of_positives_per_dataset.items()
        }
        logger.debug(f"Got {number_of_allowed_negatives_per_dataset=}")

        topic_cluster_membership = AllPositivesStrategy._get_topic_cluster_memberships(topics)
        negative_pairs = AllPositivesStrategy._create_sampled_capped_negatives(
            topics,
            number_of_allowed_negatives_per_dataset,
            topic_cluster_membership,
            negatives_per_positive,
        )

        return all_positives, negative_pairs

    @staticmethod
    def _get_topic_cluster_memberships(topics) -> dict[str, list[set[str]]]:
        result = {}
        for topic_id, clusters in topics.topic_clusters.items():
            result[topic_id] = [{mention.mention_id for mention in mentions} for mentions in clusters.values()]

        return result

    @staticmethod
    def _create_all_possible_pairs(
        topics,
        dataset_names: list[str],
    ) -> tuple[list[tuple], list[tuple], dict[str, int]]:
        """Creates all permutations of the pairs in the upper triangle and splits into positive and negatives"""
        number_of_positives_per_dataset: dict[str, int] = {c: 0 for c in dataset_names}
        positive_pairs = []
        negative_pairs = []

        for topic in topics.topics_dict.values():
            for mention_1, mention_2 in combinations(topic.mentions, 2):
                if mention_1.coref_chain == mention_2.coref_chain:
                    positive_pairs.append((mention_1, mention_2))
                    number_of_positives_per_dataset[mention_1.dataset] += 1
                else:
                    negative_pairs.append((mention_1, mention_2))

        return positive_pairs, negative_pairs, number_of_positives_per_dataset

    @staticmethod
    def _create_sampled_capped_negatives(
        topics,
        max_topic_negatives: dict,
        topic_cluster_membership: dict,
        negatives_per_positive: int,
    ):
        """Stratified negative creation based on the positive pairs"""
        negative_pairs_per_dataset = {d: [] for d in list(max_topic_negatives)}

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
                logger.debug(
                    f"Require {required_number_of_negative_candidates} negative candidates,"
                    f" have {len(negative_candidates)}"
                )
                selected_negative_candidates = random.sample(
                    list(negative_candidates),
                    # TODO: we have to do this to avoid situation when there is not enough candidates,
                    #   but this leads to "undertaking" the objects.
                    min(len(negative_candidates), required_number_of_negative_candidates),
                )
                negative_pairs = [(a, b) for a in cluster for b in selected_negative_candidates]
                negative_pairs_per_dataset[dataset].extend(negative_pairs)

        for dataset, negatives in negative_pairs_per_dataset.items():
            random.shuffle(negatives)
            negative_pairs.extend(
                negatives[: max_topic_negatives[dataset]]
            )  # TODO: why do we extend and then sample here? are we supposed to do that?

        return negative_pairs
