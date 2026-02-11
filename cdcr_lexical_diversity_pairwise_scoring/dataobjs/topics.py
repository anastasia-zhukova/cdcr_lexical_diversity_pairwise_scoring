import json
import logging
from enum import IntEnum, StrEnum
from pathlib import Path

from cdcr_lexical_diversity_pairwise_scoring.dataobjs.mention_data import MentionData, MentionuCDCR


logger = logging.getLogger(__name__)


class ScopeConfig(StrEnum):
    subtopic = "subtopic"
    topic = "topic"
    dataset = "dataset"
    corpus = "corpus"


class Topic:
    def __init__(self, topic_id: str):
        self.topic_id = topic_id
        self.mentions = []


class Topics:
    def __init__(self):
        self.topics_dict = dict()
        self.keep_order = False
        self.clusters = {}
        self.topic_clusters = dict()
        self.topics_to_datasets = dict()
        self.mention_to_topic = dict()

    def topic_id_exists(self, id_to_search):
        if id_to_search in self.topics_dict:
            return True
        return False

    def get_topic_by_id(self, id_to_search):
        if id_to_search in self.topics_dict:
            return self.topics_dict[id_to_search]
        return None

    def create_from_file(
        self,
        mentions_file_path: Path,
        keep_order: bool = True,
    ) -> None:
        """Args:
        keep_order: whether to keep original mentions order or not (default = False)
        mentions_file_path: this topic mentions json file

        """
        self.keep_order = keep_order
        with mentions_file_path.open("r", encoding="utf-8") as file:
            mentions = json.load(file)

        self.topics_dict = self.order_mentions_by_topics(mentions)
        self.convert_to_clusters()


    def create_from_mention_list(self, mentions, topic_scope: ScopeConfig):
        mentions_class_list = MentionuCDCR.read_mentions(mentions)
        if topic_scope == ScopeConfig.corpus:
            self.topics_dict["uCDCR"] = Topic("uCDCR")
            self.topics_dict["uCDCR"].mentions = mentions
        else:
            for m in mentions_class_list:
                if topic_scope == ScopeConfig.subtopic:
                    topic_id = m.subtopic_id
                elif topic_scope == ScopeConfig.topic:
                    topic_id = m.topic_id
                else:
                    # dataset level
                    topic_id = m.dataset

                if topic_id not in self.topics_dict:
                    self.topics_dict[topic_id] = Topic(topic_id)

                self.topics_dict[topic_id].mentions.append(m)
                self.topics_to_datasets[topic_id] = m.dataset
                self.mention_to_topic[m.mention_id] = topic_id
        logger.info(f"Dataset contains {len(mentions_class_list)} mentions.")
        logger.info(f"Dataset contains {len(self.topics_dict)} topics (defined by the {topic_scope.value} scope from the config).")


    def order_mentions_by_topics(self, mentions: list[dict]) -> dict[str, Topic]:
        """Order mentions to documents topics
        Args:
            mentions: json mentions file

        Returns:
            List[Topic] of the mentions separated by their documents topics

        """
        running_index = 0
        topics = dict()
        for mention_line in mentions:
            mention = MentionData._read_json_mention_data_line(mention_line)

            if self.keep_order:
                if mention.mention_index == -1:
                    mention.mention_index = running_index
                    running_index += 1

            topic_id = mention.topic_id

            if topic_id not in topics:
                topics[topic_id] = Topic(topic_id)

            topics[topic_id].mentions.append(mention)

        return topics

    def to_single_topic(self):
        new_topic = Topic("-1")

        for topic in self.topics_dict.values():
            for ment in topic.mentions:
                new_topic.mentions.append(ment)

        self.topics_dict.clear()
        self.topics_dict["-1"] = new_topic

    def convert_to_clusters(self):
        if not len(self.clusters):
            for t_id, topic in self.topics_dict.items():
                self.topic_clusters[t_id] = {}

                for mention in topic.mentions:
                    if mention.coref_chain not in self.clusters:
                        self.clusters[mention.coref_chain] = list()
                    if mention.coref_chain not in self.topic_clusters[t_id]:
                        self.topic_clusters[t_id][mention.coref_chain] = list()
                    self.clusters[mention.coref_chain].append(mention)
                    self.topic_clusters[t_id][mention.coref_chain].append(mention)
                # break
            logger.info(f"Dataset contains {len(self.clusters)} clusters (some clusters are cross-topic).")
            return self.clusters
        else:
            return self.clusters
