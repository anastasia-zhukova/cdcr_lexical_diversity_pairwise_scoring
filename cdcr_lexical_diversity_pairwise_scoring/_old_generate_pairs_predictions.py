"""Usage:
    _old_generate_pairs_predictions.py --tmf=<TestMentionsFile> --tef=<TestEmbedFile> --mf=<ModelFile> --out=<OurPredFile>
            [--cuda=<y>] [--topic=<type>] [--em=<ExtractMethod>]

Options:
    -h --help                   Show this screen.
    --cuda=<y>                  True/False - Whether to use cuda device or not [default: True]
    --topic=<type>              subtopic/topic/corpus - relevant only to ECB+, take pairs only from the same sub-topic, topic or corpus wide [default: subtopic]
    --em=<ExtractMethod>        pairwise/head_lemma/exact_string - model type to run [default: pairwise]
"""

import pickle
import random
from itertools import product
from pathlib import Path

import numpy as np
import torch
from docopt import docopt

from cdcr_lexical_diversity_pairwise_scoring import logger
from cdcr_lexical_diversity_pairwise_scoring.coref_system.pairwise_model_kenton import PairwiseModelKenton
from cdcr_lexical_diversity_pairwise_scoring.coref_system.relation_extraction import (
    HeadLemmaRelationExtractor,
    RelationTypeEnum,
)
from cdcr_lexical_diversity_pairwise_scoring.dataobjs.dataset import ScopeConfig
from cdcr_lexical_diversity_pairwise_scoring.dataobjs.topics import Topics
from cdcr_lexical_diversity_pairwise_scoring.utils.embed_utils import EmbedFromFile


MAX_ALLOWED_BATCH_SIZE = 20000


def generate_prediction_matrix(
    model: PairwiseModelKenton | HeadLemmaRelationExtractor,
    topic,
):
    all_pairs = list(product(topic.mentions, repeat=2))
    pairs_chunks = [all_pairs]
    if len(all_pairs) > MAX_ALLOWED_BATCH_SIZE:
        pairs_chunks = [
            all_pairs[i : i + MAX_ALLOWED_BATCH_SIZE] for i in range(0, len(all_pairs), MAX_ALLOWED_BATCH_SIZE)
        ]
    predictions = np.empty(0)
    with torch.no_grad():
        for chunk in pairs_chunks:
            chunk_predictions, _ = model.predict(chunk, bs=len(chunk))
            predictions = np.append(predictions, chunk_predictions.detach().cpu().numpy())
    predictions = 1 - predictions
    pred_matrix = predictions.reshape(len(topic.mentions), len(topic.mentions))
    return pred_matrix


def predict_and_save(
    event_topics: Topics,
    model: PairwiseModelKenton | HeadLemmaRelationExtractor,
    output_file: Path,
):
    all_predictions = []
    for topic in event_topics.topics_dict.values():
        logger.info(f"Evaluating Topic No {topic.topic_id}")
        predictions = generate_prediction_matrix(model, topic)
        all_predictions.append((topic, predictions))

    with output_file.open("wb") as file:
        pickle.dump(all_predictions, file)


def get_pairwise_model(
    model_file_path: Path,
    embeddings_file_path: Path,
    use_cuda: bool,
):
    pairwise_model = torch.load(model_file_path)
    pairwise_model.set_embed_utils(EmbedFromFile(embeddings_file_path))

    if use_cuda:
        pairwise_model.cuda()

    pairwise_model.eval()
    return pairwise_model


def main(arguments):
    _mentions_file = arguments.get("--tmf")
    _embed_file = arguments.get("--tef")
    _model_file = arguments.get("--mf")
    _outfile = arguments.get("--out")
    _use_cuda = True if arguments.get("--cuda").lower() == "true" else False
    _topic_arg = arguments.get("--topic")
    _extract_method_str = arguments.get("--em")

    _topic_config = ScopeConfig[_topic_arg]
    _extract_method = RelationTypeEnum[_extract_method_str]

    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)

    logger.info(f"loading model from {_model_file}.")
    event_topics = Topics()
    event_topics.create_from_file(_mentions_file, True)

    if _topic_config == ScopeConfig.corpus and len(event_topics.topics_dict) > 1:
        event_topics.to_single_topic()

    _cluster_algo = None
    if _extract_method == RelationTypeEnum.pairwise:
        model = get_pairwise_model(
            model_file_path=_model_file,
            embeddings_file_path=_embed_file,
            use_cuda=_use_cuda,
        )
    # same-lemma-head baseline
    elif _extract_method == RelationTypeEnum.same_head_lemma:
        model = HeadLemmaRelationExtractor()
    else:
        raise NotImplementedError

    logger.info(f"Running agglomerative clustering with model: {model.__name__}")
    predict_and_save(event_topics=event_topics, model=model, output_file=_outfile)


if __name__ == "__main__":
    arguments = docopt(__doc__, argv=None, help=True, version=None, options_first=False)
    main(arguments)
