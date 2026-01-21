"""
Usage:
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
from pathlib import Path

import torch
from docopt import docopt

from cdcr_lexical_diversity_pairwise_scoring import logger
from cdcr_lexical_diversity_pairwise_scoring.dataobjs.topics import Topics
from cdcr_lexical_diversity_pairwise_scoring.utils.embed_utils import EmbedTransformersGenerics


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
                logger.info(
                    f"Remaining {mention_count} Mentions in Topic {i+1}/{topic_count}."
                    f" Last Mention took {time_took} seconds"
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
        max_surrounding_contx=max_surrounding_context,
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


def main(arguments):
    multiprocessing.set_start_method("spawn")
    _file1 = arguments.get("<File>")
    _file2 = arguments.get("<File2>")
    _file3 = arguments.get("<File3>")
    max_surrounding_context = int(arguments.get("--max"))
    use_cuda = True if arguments.get("--cuda").lower() == "true" else False

    all_files = list()
    if _file1:
        all_files.append(_file1)
    if _file2:
        all_files.append(_file2)
    if _file3:
        all_files.append(_file3)

    torch.manual_seed(0)
    random.seed(0)
    if use_cuda:
        torch.cuda.manual_seed(0)

    logger.info(f"Processing files {all_files}")
    jobs = []
    for resource_file in all_files:
        job = multiprocessing.Process(target=worker, args=(resource_file, max_surrounding_context, use_cuda))
        jobs.append(job)
        job.start()

    for job in jobs:
        job.join()


if __name__ == "__main__":
    arguments = docopt(__doc__, argv=None, help=True, version=None, options_first=False)
    main(arguments)
