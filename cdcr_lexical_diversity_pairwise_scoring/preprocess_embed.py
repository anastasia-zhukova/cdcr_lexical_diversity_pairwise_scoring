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
import hydra
import torch
from datetime import datetime
import pandas as pd

from cdcr_lexical_diversity_pairwise_scoring import logger
from cdcr_lexical_diversity_pairwise_scoring.utils.io_utils import get_dataset_info_save_path, get_evaluation_result_path
from cdcr_lexical_diversity_pairwise_scoring.utils.embed_utils import EmbedTransformersGenerics
from cdcr_lexical_diversity_pairwise_scoring.preprocess_gen_pairs import Config
from cdcr_lexical_diversity_pairwise_scoring.constants import PROJECT_ROOT, CONFIG_NAME
from cdcr_lexical_diversity_pairwise_scoring.dataobjs.dataset import Split, uCDCRDataSet
from cdcr_lexical_diversity_pairwise_scoring.utils.encoding_cache import ENCODING_PAIR_TYPE, DatasetMentionEncoder

torch.manual_seed(0)
random.seed(0)


def encode_dataset_mentions(dataset: uCDCRDataSet, split: str, embed_model: EmbedTransformersGenerics, config: Config):
    encoder = DatasetMentionEncoder(embed_model=embed_model, language_model=config.language_model, split=split)
    timings = encoder.encode(dataset)

    # track the time the encoding of the test sets took (topics where something was encoded)
    if split == Split.test:
        save_filename = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_inference_time.csv'
        inference_time_df = pd.DataFrame(
            [
                {
                    "experiment": CONFIG_NAME,
                    "dataset": timing.dataset,
                    "topic": timing.topic_id,
                    "pair_type": ENCODING_PAIR_TYPE,
                    "mentions": timing.mentions_in_topic,
                    "inference_time": timing.seconds,
                }
                for timing in timings
                if timing.encoded_mentions > 0
            ],
        )
        inference_time_df.to_csv(get_evaluation_result_path() / save_filename)


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
