from collections import defaultdict
from enum import StrEnum
from itertools import combinations
from typing import Container


# TODO: remove duplication
class EvalPairsType(StrEnum):
    events = "events"
    entities = "entities"
    mix = "mix"


class EvaluationStrategy:
    @staticmethod
    def create_pairs(
        topics,
        mention_ids_events: Container[str],
        mention_ids_entities: Container[str],
        exclude_singletons: bool,
    ) -> tuple[dict, dict]:
        topic_ids_by_dataset = defaultdict(list)
        for topic_id, dataset in topics.topics_to_datasets.items():
            topic_ids_by_dataset[dataset].append(topic_id)
        topic_ids_by_dataset = dict(topic_ids_by_dataset)

        overall_positive_pairs = {}
        overall_negative_pairs = {}
        for dataset, dataset_topic_ids in topic_ids_by_dataset.items():
            overall_positive_pairs[dataset] = {}
            overall_negative_pairs[dataset] = {}

            for topic_id in dataset_topic_ids:
                overall_positive_pairs[dataset][topic_id] = {}
                overall_negative_pairs[dataset][topic_id] = {}

                for pair_type in [EvalPairsType.events, EvalPairsType.entities, EvalPairsType.mix]:
                    mentions = EvaluationStrategy._filter_mentions(
                        topic_id,
                        topics,
                        pair_type,
                        mention_ids_events,
                        mention_ids_entities,
                        exclude_singletons,
                    )

                    positives_pairs = []
                    negative_pairs = []

                    for mention_1, mention_2 in combinations(mentions, 2):
                        if mention_1.coref_chain == mention_2.coref_chain:
                            positives_pairs.append((mention_1, mention_2))
                        else:
                            negative_pairs.append((mention_1, mention_2))

                    overall_positive_pairs[dataset][topic_id][pair_type] = positives_pairs
                    overall_negative_pairs[dataset][topic_id][pair_type] = negative_pairs

        return overall_positive_pairs, overall_negative_pairs

    @staticmethod
    def _filter_mentions(
        topic_id: str,
        topics,
        pair_type: str,
        mention_ids_events,
        mention_ids_entities,
        exclude_singletons: bool,
    ):
        all_mentions = topics.topics_dict[topic_id].mentions
        if pair_type == EvalPairsType.events.value:
            mentions = [mention for mention in all_mentions if mention.mention_id in mention_ids_events]
        elif pair_type == EvalPairsType.entities.value:
            mentions = [mention for mention in all_mentions if mention.mention_id in mention_ids_entities]
        else:
            mentions = all_mentions

        if exclude_singletons:
            mentions = [mention for mention in mentions if not mention.is_singleton]
        return mentions
