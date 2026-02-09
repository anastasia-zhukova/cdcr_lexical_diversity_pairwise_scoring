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
from dataclasses import dataclass
from pathlib import Path

import hydra
import torch
from hydra.core.config_store import ConfigStore

from cdcr_lexical_diversity_pairwise_scoring import logger
from cdcr_lexical_diversity_pairwise_scoring.constants import PROJECT_ROOT
from cdcr_lexical_diversity_pairwise_scoring.dataobjs.topics import Topics
from cdcr_lexical_diversity_pairwise_scoring.utils.embed_utils import EmbedTransformersGenerics


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

    return result_train


def worker(
    resource_file: Path,
    max_surrounding_context: int,
    use_cuda: bool,
):
    embed_model = EmbedTransformersGenerics(
        max_surrounding_context=max_surrounding_context,
        use_cuda=use_cuda,
    )
    name = multiprocessing.current_process().name
    logger.info(f"Starting {name}")

    basename = resource_file.stem
    dirname = resource_file.parent
    save_to = dirname / (basename + "_roberta_large.pickle")

    topics = Topics()
    topics.create_from_file(resource_file, keep_order=True)
    train_feat = extract_feature_dict(topics, embed_model)
    with save_to.open("wb") as file:
        pickle.dump(train_feat, file)

    logger.info(f"Finished {basename}")


cs = ConfigStore.instance()
cs.store(name="preprocess_embed_config", node=Config)


@hydra.main(version_base="1.3", config_path=str(PROJECT_ROOT / "config"), config_name="preprocess_embed_config")
def main(config: Config) -> None:
    multiprocessing.set_start_method("spawn")

    all_files = list()
    if config.file_1:
        all_files.append(PROJECT_ROOT / config.file_1)
    if config.file_2:
        all_files.append(PROJECT_ROOT / config.file_2)
    if config.file_3:
        all_files.append(PROJECT_ROOT / config.file_3)

    torch.manual_seed(0)
    random.seed(0)
    if config.use_cuda:
        torch.cuda.manual_seed(0)

    logger.info(f"Processing files {all_files}")
    jobs = []
    for resource_file in all_files:
        job = multiprocessing.Process(target=worker, args=(resource_file, config.max_context, config.use_cuda))
        jobs.append(job)
        job.start()

    for job in jobs:
        job.join()


if __name__ == "__main__":
    main()
