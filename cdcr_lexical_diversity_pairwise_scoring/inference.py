"""
Usage:
    inference.py --tpf=<TestPosFile> --tnf=<testNegFile> --te=<TestEmbed> --mf=<ModelFile> [--cuda=<b>]

Options:
    -h --help       Show this screen.
    --cuda=<y>      True/False - Whether to use cuda device or not [default: True]

"""

import logging
import ntpath
import os

import torch
from docopt import docopt
from cdcr_lexical_diversity_pairwise_scoring.train import accuracy_on_dataset
from cdcr_lexical_diversity_pairwise_scoring.utils.log_utils import create_logger_with_fh
from cdcr_lexical_diversity_pairwise_scoring.dataobjs.dataset import EcbDataSet
from cdcr_lexical_diversity_pairwise_scoring.utils.embed_utils import EmbedFromFile

logger = logging.getLogger(__name__)


def main(arguments):
    dataset_arg = arguments.get("--dataset")
    model_file = arguments.get("--mf")
    event_test_file_pos = arguments.get("--tpf")
    event_test_file_neg = arguments.get("--tnf")
    embed_file = arguments.get("--te")
    use_cuda = True if arguments.get("--cuda").lower() == "true" else False

    # TODO: only supports ECB?
    dataset = EcbDataSet()

    # TODO: replace for better logger.
    log_param_str = os.path.dirname(model_file) + "/inference_" + ntpath.basename(model_file)
    create_logger_with_fh(log_param_str)

    logger.info(f"Loading the model from {model_file}")
    pairwise_model = torch.load(model_file)
    embed_utils = EmbedFromFile(embed_file)
    pairwise_model.set_embed_utils(embed_utils)
    pairwise_model.eval()

    # TODO: replace for Path
    positive_pairs = dataset.load_pair_pickle(event_test_file_pos)
    negative_pairs = dataset.load_pair_pickle(event_test_file_neg)
    split_feat = dataset.create_features_from_pos_neg(positive_pairs, negative_pairs)

    accuracy_on_dataset("", 0, pairwise_model, split_feat)


if __name__ == "__main__":
    arguments = docopt(__doc__, argv=None, help=True, version=None, options_first=False)
    main(arguments)
