import pickle
import random

import hydra
import torch
from hydra.core.config_store import ConfigStore
from tqdm import tqdm

from cdcr_lexical_diversity_pairwise_scoring import logger
from cdcr_lexical_diversity_pairwise_scoring.configs import EmbeddingsCacheBuildingConfig
from cdcr_lexical_diversity_pairwise_scoring.constants import CACHED_VECTOR_PATH, PROJECT_ROOT
from cdcr_lexical_diversity_pairwise_scoring.dataobjs.ucdcr_dataset import uCDCRDataSet
from cdcr_lexical_diversity_pairwise_scoring.utils.embed_utils import EmbedTransformersGenerics


SAVE_CACHE_FREQUENCY = 1000


def encode_dataset_mentions(
    dataset: uCDCRDataSet,
    embed_model: EmbedTransformersGenerics,
):
    n_topics = len(dataset.topics.topics_dict)
    if CACHED_VECTOR_PATH.exists():
        logger.info("Loading cached vectors.")
        with CACHED_VECTOR_PATH.open("rb") as file:
            encoded_mentions = pickle.load(file)
    else:
        encoded_mentions = {}

    number_of_mentions = 0
    for i, (topic_id, topic) in enumerate(dataset.topics.topics_dict.items()):
        for mention in tqdm(
            topic.mentions,
            desc=f"Encoding mentions of topic {topic_id} ({i}/{n_topics - 1})",
            leave=False,
        ):
            if mention.mention_id in encoded_mentions:
                continue

            hidden, first_token, last_token, mention_size = embed_model.get_mention_full_rep(mention)
            encoded_mentions[mention.mention_id] = (hidden.cpu(), first_token.cpu(), last_token.cpu(), mention_size)

            number_of_mentions += 1
            if number_of_mentions % SAVE_CACHE_FREQUENCY == 0:
                with CACHED_VECTOR_PATH.open("wb") as file:
                    pickle.dump(encoded_mentions, file)

    with CACHED_VECTOR_PATH.open("wb") as file:
        pickle.dump(encoded_mentions, file)


cs = ConfigStore.instance()
cs.store(name="preprocess_embeddings", node=EmbeddingsCacheBuildingConfig)


@hydra.main(version_base="1.3", config_path=str(PROJECT_ROOT / "config"), config_name="preprocess_embeddings")
def main(config: EmbeddingsCacheBuildingConfig) -> None:
    torch.manual_seed(0)
    random.seed(0)
    if config.use_cuda:
        torch.cuda.manual_seed(0)

    embeddings_model = EmbedTransformersGenerics(
        max_surrounding_context=config.max_surrounding_context,
        use_cuda=config.use_cuda,
    )

    datasets = uCDCRDataSet.load_from_config(config.data_config)
    for dataset in datasets:
        encode_dataset_mentions(dataset, embeddings_model)


if __name__ == "__main__":
    main()
