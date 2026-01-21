from cdcr_lexical_diversity_pairwise_scoring.dataobjs.mention_data import MentionData


class Cluster(object):
    def __init__(self, coref_chain: int = -1) -> None:
        """
        Object represent a set of mentions with same coref chain id

        Args:
            coref_chain (int): the cluster id/coref_chain value
        """
        self.mentions = []
        self.cluster_strings = []
        self.merged = False
        self.coref_chain = coref_chain
        self.mentions_corefs = set()

    def add_mention(self, mention: MentionData) -> None:
        if mention is not None:
            mention.predicted_coref_chain = self.coref_chain
            self.mentions.append(mention)
            self.cluster_strings.append(mention.tokens_str)
            self.mentions_corefs.add(mention.coref_chain)


class Clusters(object):
    cluster_coref_chain = 0

    def __init__(
        self,
        topic_id: str,
        mentions: list[MentionData] | None = None,
    ) -> None:
        self.clusters_list = []
        self.topic_id = topic_id
        self.set_initial_clusters(mentions)

    def set_initial_clusters(
        self,
        mentions: list[MentionData] | None = None,
    ) -> None:
        if mentions is not None:
            for mention in mentions:
                cluster = Cluster(Clusters.cluster_coref_chain)
                cluster.add_mention(mention)
                self.clusters_list.append(cluster)
                Clusters.cluster_coref_chain += 1

    @classmethod
    def inc_cluster_coref_chain(cls, value: int) -> None:
        cls.cluster_coref_chain += value

    @classmethod
    def get_cluster_coref_chain(cls) -> int:
        return cls.cluster_coref_chain

    @staticmethod
    def from_mentions_to_predicted_clusters(mentions: list[MentionData]) -> dict:
        clusters = {}
        for mention in mentions:
            if mention.predicted_coref_chain not in clusters:
                clusters[mention.predicted_coref_chain] = []
            clusters[mention.predicted_coref_chain].append(mention)

        return clusters

    @staticmethod
    def from_mentions_to_gold_clusters(mentions):
        clusters = dict()
        for mention in mentions:
            if mention.coref_chain not in clusters:
                clusters[mention.coref_chain] = list()
            clusters[mention.coref_chain].append(mention)

        return clusters
