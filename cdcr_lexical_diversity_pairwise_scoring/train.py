"""Usage:
    train.py [--bs=<x>] [--lr=<y>] [--ratio=<z>] [--itr=<k>]
                    [--cuda=<b>] [--ft=<b1>] [--wd=<t>] [--hidden=<w>] [--dataset=<d>]

Options:
    -h --help       Show this screen.
    --bs=<x>        Batch size [default: 32]
    --lr=<y>        Learning rate [default: 5e-4]
    --ratio=<z>     Ratio of positive:negative, were negative is the controlled list (ratio=-1 => no ratio) [default: -1]
    --itr=<k>       Number of iterations [default: 10]
    --cuda=<y>      True/False - Whether to use cuda device or not [default: True]
    --ft=<b1>       Fine-tune the LM or not [default: False]
    --wd=<t>        Adam optimizer Weight-decay [default: 0.01]
    --hidden=<w>    hidden layers size [default: 150]
    --dataset=<d>   wec/ecb - which dataset to generate for [default: wec]

"""

import logging
import pickle
import random
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from docopt import docopt
from hydra.core.config_store import ConfigStore

from cdcr_lexical_diversity_pairwise_scoring.coref_system.pairwise_model_kenton import PairwiseModelKenton
from cdcr_lexical_diversity_pairwise_scoring.dataobjs.dataset import Split
from cdcr_lexical_diversity_pairwise_scoring.utils.embed_utils import EmbedFromFile
from cdcr_lexical_diversity_pairwise_scoring.utils.eval_utils import get_confusion_matrix, precision_recall_f1
from cdcr_lexical_diversity_pairwise_scoring.utils.io_utils import create_and_get_path, get_dataset_name, get_model_name
from cdcr_lexical_diversity_pairwise_scoring.utils.log_utils import create_logger_with_fh
from cdcr_lexical_diversity_pairwise_scoring.constants import PROJECT_ROOT, CACHED_VECTOR_PATH, USE_CUDA
from cdcr_lexical_diversity_pairwise_scoring.preprocess_gen_pairs import Config

torch.manual_seed(1234)
random.seed(1234)
np.random.seed(1234)
logger = logging.getLogger(__name__)

CONFIG_NAME = "preprocess_test"
cs = ConfigStore.instance()
cs.store(name=CONFIG_NAME, node=Config)


def train_pairwise(
    pairwise_model: PairwiseModelKenton,
    train,
    validation,
    batch_size: int,
    epochs: int = 4,
    lr: float = 1e-5,
    model_out=None,
    weight_decay: float = 0.01,
):
    loss_func = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(pairwise_model.parameters(), lr, weight_decay=weight_decay)
    dataset_size = len(train)

    best_result_so_far = -1

    for epoch in range(epochs):
        pairwise_model.train()
        end_index = batch_size
        random.shuffle(train)

        cumulative_loss = 0.0
        current_batch = 1
        # TODO: rewrite using dataloader.
        for start_index in range(0, dataset_size, batch_size):
            end_index = min(end_index, dataset_size)

            optimizer.zero_grad()

            batch_features = train[start_index:end_index].copy()
            bs = end_index - start_index
            prediction, gold_labels = pairwise_model(batch_features, bs)

            loss = loss_func(prediction, gold_labels.reshape(-1, 1).float())
            loss.backward()
            optimizer.step()

            cumulative_loss += loss.item()
            end_index += batch_size
            current_batch += 1

            if current_batch % 100 == 0:
                report = "%d: %d: loss: %.10f:" % (epoch + 1, end_index, cumulative_loss / current_batch)
                logger.info(report)

        pairwise_model.eval()
        _, _, _, dev_f1 = accuracy_on_dataset("Dev", epoch + 1, pairwise_model, validation)

        if best_result_so_far < dev_f1:
            logger.info("Found better model saving")
            torch.save(pairwise_model, model_out + "iter_" + str(epoch + 1))
            best_result_so_far = dev_f1

    return best_result_so_far


def accuracy_on_dataset(
    evaluation_set_name: str,
    epoch: int,
    pairwise_model: PairwiseModelKenton,
    features,
    batch_size: int = 10000,
):
    all_labels, all_predictions = run_inference(pairwise_model, features, batch_size=batch_size)
    accuracy = torch.mean((all_labels == all_predictions).float())
    tn, fp, fn, tp = get_confusion_matrix(all_labels, all_predictions)
    precision, recall, f1 = precision_recall_f1(tp, fp, fn)

    logger.info(
        "%s: %d: Accuracy: %.10f: precision: %.10f: recall: %.10f: f1: %.10f"
        % (evaluation_set_name + "-Acc", epoch, accuracy.item(), precision, recall, f1),
    )

    return accuracy, precision, recall, f1


def run_inference(
    pairwise_model: PairwiseModelKenton,
    features,
    round_pred: bool = True,
    batch_size: int = 10000,
) -> tuple[torch.Tensor, torch.Tensor]:
    dataset_size = len(features)
    labels = []
    predictions = []

    for start_index in range(0, dataset_size, batch_size):
        end_index = min(start_index + batch_size, dataset_size)
        batch_features = features[start_index:end_index].copy()
        current_batch_size = end_index - start_index

        # TODO: model should not work with real labels.
        batch_predictions, batch_label = pairwise_model.predict(batch_features, current_batch_size)

        if round_pred:
            batch_predictions = torch.round(batch_predictions.reshape(-1)).long()

        predictions.append(batch_predictions.detach().cpu())
        labels.append(batch_label.detach().cpu())

    all_labels = torch.cat(labels)
    all_predictions = torch.cat(predictions)

    return all_labels, all_predictions


def init_basic_training_resources(
    embeddings_path: Path,
    dataset_config_file: Path,
    hidden_size: int,
    use_cuda: bool = USE_CUDA,
) -> tuple[
    ...,  # TODO
    ...,  # TODO
    PairwiseModelKenton,
    str,
    str
]:
    # load encoded mentions
    embed_utils = EmbedFromFile(embeddings_path)
    # init a model
    pairwise_model = PairwiseModelKenton(embed_utils.embed_size, hidden_size, 1, embed_utils, use_cuda)

    # load datasets
    with open(dataset_config_file, "r") as file:
        dataset_config = json.load(file)

    with open(dataset_config[Split.train.value], "rb") as file:
        train_dataset = pickle.load(file)

    with open(dataset_config[Split.dev.value], "rb") as file:
        dev_dataset = pickle.load(file)

    train_feat = train_dataset.get_mix_pairs()
    validation_feat = dev_dataset.get_mix_pairs()

    if use_cuda:
        torch.cuda.manual_seed(1234)
        pairwise_model.cuda()

    return train_feat, validation_feat, pairwise_model, "_".join(train_dataset.dataset_components), "_".join(dev_dataset.dataset_components)


def main(arguments, config_name: str):
    start_time = datetime.now()
    dt_string = start_time.strftime("%d%m%Y_%H%M%S")
    _output_folder = create_and_get_path("checkpoints/" + dt_string)
    _batch_size = int(arguments.get("--bs", 16))
    _learning_rate = float(arguments.get("--lr", 5e-4))
    _iterations = int(arguments.get("--itr", 10))
    _use_cuda = True if arguments.get("--cuda").lower() == "true" else False
    _fine_tune = True if arguments.get("--ft").lower() == "true" else False
    _weight_decay = float(arguments.get("--wd", 0.01))
    _hidden_size = int(arguments.get("--hidden", 150))

    model_name = get_model_name(config_name)
    _model_file = _output_folder / model_name

    file_name = get_dataset_name(config_name)
    dataset_file_path = PROJECT_ROOT / "config" / file_name

    _event_train_feat, _event_validation_feat, _pairwise_model, _train_names, _dev_names = init_basic_training_resources(
        embeddings_path=CACHED_VECTOR_PATH,
        dataset_config_file=dataset_file_path,
        hidden_size=_hidden_size,
    )

    log_params_str = (
        "ds_"
        + _train_names
        + "_lr_"
        + str(_learning_rate)
        + "_bs_"
        + str(_batch_size)
        + "_itr"
        + str(_iterations)
    )
    # TODO: replace with simple logger.
    create_logger_with_fh(str(_output_folder) + "/"  f"train_{log_params_str}")

    # TODO: prettify
    logger.info(
        "train_set="
        + _train_names
        + ", lr="
        + str(_learning_rate)
        + ", bs="
        + str(_batch_size)
        + ", itr="
        + str(_iterations)
        + ", hidden_s="
        + str(_hidden_size)
        + ", weight_decay="
        + str(_weight_decay),
    )

    train_pairwise(
        _pairwise_model,
        _event_train_feat,
        _event_validation_feat,
        _batch_size,
        _iterations,
        _learning_rate,
        model_out=_model_file,
        weight_decay=_weight_decay,
    )


if __name__ == "__main__":
    arguments = docopt(__doc__, argv=None, help=True, version=None, options_first=False)
    # TODO Sergei: proper reading as arg the config file for the experiment + the config for the training parameters
    config_name = CONFIG_NAME
    main(arguments, config_name)
