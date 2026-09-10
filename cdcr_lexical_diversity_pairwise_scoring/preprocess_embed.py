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

import pickle
import random
import json
from pathlib import Path
from tqdm import tqdm
import hydra
import torch

from cdcr_lexical_diversity_pairwise_scoring import logger
from cdcr_lexical_diversity_pairwise_scoring.utils.io_utils import get_dataset_info_save_path, get_encoding_cache_file
from cdcr_lexical_diversity_pairwise_scoring.utils.embed_utils import EmbedTransformersGenerics
from cdcr_lexical_diversity_pairwise_scoring.preprocess_gen_pairs import Config
from cdcr_lexical_diversity_pairwise_scoring.constants import PROJECT_ROOT, CONFIG_NAME
from cdcr_lexical_diversity_pairwise_scoring.dataobjs.dataset import uCDCRDataSet

torch.manual_seed(0)
random.seed(0)


def encode_dataset_mentions(dataset: uCDCRDataSet, split: str, embed_model: EmbedTransformersGenerics, config: Config):

    device = torch.device("cuda" if torch.cuda.is_available() and config.use_cuda else "cpu")
    topic_num = len(dataset.topics.topics_dict)

    # encoding only mentions from the created pairs
    mentions_to_encode = set()
    for mention_pair in dataset.positive_pairs + dataset.negative_pairs:
        mentions_to_encode.add(mention_pair[0].mention_id)
        mentions_to_encode.add(mention_pair[1].mention_id)

    if len(dataset.positive_pairs_eval_format):
        for dataset_name, topic_dict in dataset.positive_pairs_eval_format.items():
            for topic_id, mention_types_dict in topic_dict.items():
                for mention_pair in mention_types_dict["mix"] + dataset.negative_pairs_eval_format[dataset_name][topic_id]["mix"]:
                    mentions_to_encode.add(mention_pair[0].mention_id)
                    mentions_to_encode.add(mention_pair[1].mention_id)

    encoded_mentions = {}
    cached_vector_path = None

    for i, (topic_id, dataset_name) in enumerate(dataset.topics.topics_to_datasets.items()):
        cached_vector_path_next = get_encoding_cache_file(split, dataset_name, config.language_model)

        # if it is the same dataset, do not reread the existing file
        if cached_vector_path_next != cached_vector_path:
            if cached_vector_path_next.exists():
                with open(cached_vector_path_next, "rb") as file:
                    encoded_mentions = pickle.load(file)

        cached_vector_path = cached_vector_path_next
        topic = dataset.topics.topics_dict[topic_id]

        for mention in tqdm(topic.mentions, desc=f"Encoding mentions of topic {topic_id} ({i}/{topic_num - 1})"):
            if mention.mention_id in encoded_mentions:
                continue

            if mention.mention_id not in mentions_to_encode:
                continue

            hidden, first_token, last_token, mention_size = embed_model.get_mention_full_rep(mention)
            encoded_mentions[mention.mention_id] = (hidden.to(device), first_token.to(device), last_token.to(device), mention_size)

            with cached_vector_path.open("wb") as file:
                pickle.dump(encoded_mentions, file)


def encode_dataset(
    dataset_file: Path,
    split: str,
    config: Config,
    max_surrounding_context: int = -1,
):
    embed_model = EmbedTransformersGenerics(
        bert_model_name=config.language_model,
        max_surrounding_context=max_surrounding_context,
        use_cuda=config.use_cuda,
    )

    logger.info(f"Encoding mentions from {dataset_file}...")
    with open(dataset_file, "rb") as file:
        dataset = pickle.load(file)

    encode_dataset_mentions(dataset, split, embed_model, config)
    logger.info(f"Finished encoding mentions from {dataset_file}.")


@hydra.main(version_base="1.3", config_path=str(PROJECT_ROOT / "config"), config_name=CONFIG_NAME)
def main(config: Config) -> None:
    if config.use_cuda:
        torch.cuda.manual_seed(0)

    dataset_file_path = get_dataset_info_save_path()
    with open(dataset_file_path, "r", encoding="utf-8") as file:
        dataset_dict = json.load(file)

    logger.info(f"Processing files from {dataset_file_path}")
    for split, dataset_path in dataset_dict.items():
        # no need for the multithreading especially because I am caching files into one pickle file, which we need then to check for the IO rights
        encode_dataset(dataset_path, split, config)


if __name__ == "__main__":
    main()
