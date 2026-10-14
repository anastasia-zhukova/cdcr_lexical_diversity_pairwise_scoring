import logging
import time

from sklearn.cluster import AgglomerativeClustering

from cdcr_lexical_diversity_pairwise_scoring.dataobjs.cluster import Clusters


logger = logging.getLogger(__name__)


def agglomerative_clustering(pred_matrix, topic, average_link_threshold):
    start = time.time()
    n_clusters = run_clustering(pred_matrix, topic, average_link_threshold)

    took = time.time() - start
    logger.info(f"Produced {n_clusters} clusters using agglomerative clustering, took: {took:.4f} sec")

    return topic.mentions


def run_clustering(pred_matrix, topic, average_link_threshold):
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=average_link_threshold,
    ).fit(pred_matrix)

    # Mutates the original topic's mentions' predicted_coref_chain-s: this is why it is not returned.
    # TODO: fight mutation
    for i in range(len(topic.mentions)):
        topic.mentions[i].predicted_coref_chain = clustering.labels_[i] + Clusters.get_cluster_coref_chain()

    n_clusters = max(clustering.labels_) + 1
    Clusters.inc_cluster_coref_chain(n_clusters)
    return n_clusters
