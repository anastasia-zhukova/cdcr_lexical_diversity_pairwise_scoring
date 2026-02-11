"""Usage:
    preprocess_embed.py <File> [<File2>] [<File3>]
    preprocess_embed.py <File> [<File2>] [<File3>] [--max=<x>]
    preprocess_embed.py <File> [<File2>] [<File3>] [--cuda=<y>]
    preprocess_embed.py <File> [<File2>] [<File3>] [--max=<x>] [--cuda=<y>]

Options:
    -h --help     Show this screen.
    --max=<x>   Maximum surrounding context [default: 250]
    --cuda=<y>  True/False - Whether to use cuda device or not [default: True].

"""

import multiprocessing
import pickle
import random
import time
import hydra
import json
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm

import hydra
import torch
from hydra.core.config_store import ConfigStore

from cdcr_lexical_diversity_pairwise_scoring import logger
from cdcr_lexical_diversity_pairwise_scoring.utils.io_utils import get_dataset_config_name
from cdcr_lexical_diversity_pairwise_scoring.dataobjs.topics import Topics
from cdcr_lexical_diversity_pairwise_scoring.utils.embed_utils import EmbedTransformersGenerics
from cdcr_lexical_diversity_pairwise_scoring.preprocess_gen_pairs import Config
from cdcr_lexical_diversity_pairwise_scoring.constants import PROJECT_ROOT, USE_CUDA, CACHED_VECTOR_PATH
from cdcr_lexical_diversity_pairwise_scoring.dataobjs.dataset import uCDCRDataSet

CONFIG_NAME = "preprocess_test"
cs = ConfigStore.instance()
cs.store(name=CONFIG_NAME, node=Config)
torch.manual_seed(0)
random.seed(0)
CACHE_FREQUENCY = 1000


def encode_dataset_mentions(dataset: uCDCRDataSet, embed_model: EmbedTransformersGenerics):
    topic_num = len(dataset.topics.topics_dict)

    if CACHED_VECTOR_PATH.exists():
        with open(CACHED_VECTOR_PATH, "rb") as file:
            encoded_mentions = pickle.load(file)
    else:
        encoded_mentions = {}

    m_num = 0
    for i, (topic_id, topic) in enumerate(dataset.topics.topics_dict.items()):

        for mention in tqdm(topic.mentions, desc=f"Encoding mentions of topic {topic_id} ({i}/{topic_num - 1})"):
            if mention.mention_id in encoded_mentions:
                continue

            hidden, first_token, last_token, mention_size = embed_model.get_mention_full_rep(mention)
            encoded_mentions[mention.mention_id] = (hidden.cpu(), first_token.cpu(), last_token.cpu(), mention_size)
            m_num += 1
            if m_num % CACHE_FREQUENCY == 0:
               with CACHED_VECTOR_PATH.open("wb") as file:
                    pickle.dump(encoded_mentions, file)
@dataclass
class Config:
    file_1: Path | None
    file_2: Path | None
    file_3: Path | None
    use_cuda: bool
    max_context: int


def extract_feature_dict(
    topics: Topics,
    embed_model: EmbedTransformersGenerics,
) -> dict:
    result_train = {}
    topic_count = len(topics.topics_dict)
    for i, topic in enumerate(topics.topics_dict.values()):
        mention_count = len(topic.mentions)
        for mention in topic.mentions:
            start = time.time()
            hidden, first_token, last_token, mention_size = embed_model.get_mention_full_rep(mention)
            time_took = time.time() - start

            result_train[mention.mention_id] = (hidden.cpu(), first_token.cpu(), last_token.cpu(), mention_size)
            mention_count -= 1
            if mention_count > 0:
                # TODO: refactor to not overflood.
                logger.info(
                    f"Remaining {mention_count} Mentions in Topic {i+1}/{topic_count}."
                    f" Last Mention took {time_took} seconds",
                )
            else:
                logger.info(f"Finished Topic {i+1}/{topic_count}. Last Mention took {time_took} seconds")

    with CACHED_VECTOR_PATH.open("wb") as file:
        pickle.dump(encoded_mentions, file)


def encode_dataset(
    dataset_file: Path,
    max_surrounding_context: int = -1,
):
    embed_model = EmbedTransformersGenerics(
        max_surrounding_context=max_surrounding_context,
        use_cuda=USE_CUDA,
    )
    # name = multiprocessing.current_process().name
    # logger.info(f"Starting {name}")
    logger.info(f"Encoding mentions from {dataset_file}...")

    # basename = dataset_file.stem
    # dirname = dataset_file.parent
    with open(dataset_file, "rb") as file:
        dataset = pickle.load(file)

    encode_dataset_mentions(dataset, embed_model)
    logger.info(f"Finished encoding mentions from {dataset_file}.")


def main(config_name) -> None:
    if USE_CUDA:
        torch.cuda.manual_seed(0)

    file_name = get_dataset_config_name(config_name)
    dataset_file_path = PROJECT_ROOT / "config" / file_name
    with open(dataset_file_path, "r", encoding="utf-8") as file:
        dataset_dict = json.load(file)

    logger.info(f"Processing files from {dataset_file_path}")
    for split, dataset_path in dataset_dict.items():
        encode_dataset(dataset_path)


if __name__ == "__main__":
    main()
