"""

Usage:
    preprocess_gen_pairs.py <File> --dataset=<dataset>
    preprocess_gen_pairs.py <File> --dataset=<dataset> [--split=<set>]
    preprocess_gen_pairs.py <File> --dataset=<dataset> [--split=<set>] [--ratio=<x>] [--topic=<type>]

Options:
    -h --help     Show this screen.
    --dataset=<dataset>   wec/ecb - which dataset to generate for [default: wec]
    --split=<set>    dev/test/train/na (split=na => doesnt matter) [default: na].
    --ratio=<x>  ratio of positive:negative, were negative is the controlled list (ratio=-1 => no ratio) [default: -1]
    --topic=<type>  subtopic/topic/corpus - relevant only to ECB+, take pairs only from the same sub-topic, topic or corpus wide [default: corpus]

"""

import pickle
import random
from pathlib import Path

from docopt import docopt

from cdcr_lexical_diversity_pairwise_scoring import logger
from cdcr_lexical_diversity_pairwise_scoring.dataobjs.dataset import Split, DataSet
from cdcr_lexical_diversity_pairwise_scoring.dataobjs.topics import TopicConfig


def generate_pairs(
    event_validation_file: str,
    dataset: DataSet,
    topic_config: int,  # TODO: confit should not be presented as int.
):
    positive, negative = dataset.get_pairwise_feat(
        data_file=event_validation_file,
        to_topics=topic_config,
    )
    logger.debug(f"Created {len(positive)} positive pairs and {len(negative)} negative pairs.")
    validate_pairs(positive, negative)

    dirname = Path(event_validation_file).parent
    basename = Path(event_validation_file).stem

    positive_file_path = dirname / f"{basename}_PosPairs.pickle"
    with positive_file_path.open("wb") as file:
        pickle.dump(positive, file)
    logger.info(f"Saved positive pairs in {positive_file_path}.")

    negative_file_path = dirname / f"{basename}_NegPairs.pickle"
    with negative_file_path.open("wb") as file:
        pickle.dump(negative, file)
    logger.info(f"Saved negative pairs in {negative_file_path}.")


def validate_pairs(pos_pairs, neg_pairs):
    for men1, men2 in pos_pairs:
        if men1.coref_chain != men2.coref_chain:
            raise ValueError("Error when validating positive pairs!")

    for men1, men2 in neg_pairs:
        if men1.coref_chain == men2.coref_chain:
            raise ValueError("Error when validating negative pairs!")

    logger.info("Validation Passed!")


def main(arguments):
    _event_validation_file = arguments.get("<File>")
    _ratio = int(arguments.get("--ratio"))
    _split_arg = arguments.get("--split").lower()
    _topic_arg = arguments.get("--topic")
    _dataset_arg = arguments.get("--dataset")

    # TODO: raise error when not in train/dev/test?
    split = Split[_split_arg]

    topic_config = TopicConfig[_topic_arg]

    random.seed(0)
    dataset = DataSet.get_dataset(_dataset_arg, ratio=_ratio, split=split)

    if _dataset_arg == "wec" and split == Split.train and _ratio == -1:
        logger.warning("Selected WEC dataset for train with a -1 ratio will generate all possible negative pairs!!")

    logger.info("Generating pairs for file-/" + _event_validation_file)
    generate_pairs(
        event_validation_file=_event_validation_file,
        dataset=dataset,
        topic_config=topic_config,
    )


if __name__ == "__main__":
    arguments = docopt(
        __doc__,
        argv=None,
        help=True,
        version=None,
        options_first=False,
    )
    main(arguments)
