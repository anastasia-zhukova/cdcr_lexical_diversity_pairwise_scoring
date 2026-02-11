import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from tqdm import tqdm

from cdcr_lexical_diversity_pairwise_scoring import logger
from cdcr_lexical_diversity_pairwise_scoring.constants import PROJECT_ROOT
from cdcr_lexical_diversity_pairwise_scoring.coref_system.pairwise_model_kenton import PairwiseModelKenton
from cdcr_lexical_diversity_pairwise_scoring.dataobjs.dataset import DataSet, DatasetEnum, Split
from cdcr_lexical_diversity_pairwise_scoring.utils.embed_utils import EmbedFromFile
from cdcr_lexical_diversity_pairwise_scoring.utils.eval_utils import get_confusion_matrix, precision_recall_f1
from cdcr_lexical_diversity_pairwise_scoring.utils.io_utils import create_and_get_path
from cdcr_lexical_diversity_pairwise_scoring.utils.log_utils import create_logger_with_fh


@dataclass
class Config:
    # required files
    train_positive_file: Path = MISSING
    train_negative_file: Path = MISSING
    dev_positive_file: Path = MISSING
    dev_negative_file: Path = MISSING
    train_embeddings_file: Path = MISSING
    dev_embeddings_file: Path = MISSING

    # outputs / identity
    model_name: str = MISSING
    dataset_name: DatasetEnum = MISSING

    # training params
    batch_size: int = MISSING
    learning_rate: float = MISSING
    negative_positive_ratio: int = MISSING
    training_iterations: int = MISSING
    use_cuda: bool = MISSING
    fine_tune: bool = MISSING
    weight_decay: float = MISSING
    hidden_size: int = MISSING


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
        pbar = tqdm(total=dataset_size)
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
            pbar.update(1)

            if current_batch % 100 == 0:
                pbar.set_postfix(loss=f"{cumulative_loss / current_batch:.10f}")

        pbar.close()
        logger.info(f"Finished training for epoch {epoch}. Final loss: {cumulative_loss / current_batch:.10f}")

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
    train_embeddings_path: Path,
    dev_embeddings_path: Path,
    dataset_type: DatasetEnum,
    train_positive_file_path: Path,
    train_negative_file_path: Path,
    dev_positive_file_path: Path,
    dev_negative_file_path: Path,
    ratio: int,
    hidden_size: int,
    use_cuda: bool,
) -> tuple[
    ...,  # TODO
    ...,  # TODO
    PairwiseModelKenton,
]:
    torch.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)

    embed_utils = EmbedFromFile([train_embeddings_path, dev_embeddings_path], use_cuda)
    pairwise_model = PairwiseModelKenton(embed_utils.embed_size, hidden_size, 1, embed_utils, use_cuda)

    train_dataset = DataSet.get_dataset(dataset_type, ratio=ratio, split=Split.train)
    dev_dataset = DataSet.get_dataset(dataset_type, split=Split.dev)
    train_feat = train_dataset.load_pos_neg_pickle(train_positive_file_path, train_negative_file_path)
    validation_feat = dev_dataset.load_pos_neg_pickle(dev_positive_file_path, dev_negative_file_path)

    if use_cuda:
        logger.info("Using cuda.")
        torch.cuda.manual_seed(1234)
        pairwise_model.cuda()

    return train_feat, validation_feat, pairwise_model


cs = ConfigStore.instance()
cs.store(name="train_config", node=Config)


@hydra.main(version_base="1.3", config_path=str(PROJECT_ROOT / "config"), config_name="train_config")
def main(config: Config):
    start_time = datetime.now()
    dt_string = start_time.strftime("%d%m%Y_%H%M%S")
    output_folder = create_and_get_path("checkpoints/" + dt_string)

    log_params_str = (
        "ds_"
        + config.dataset_name
        + "_lr_"
        + str(config.learning_rate)
        + "_bs_"
        + str(config.batch_size)
        + "_r"
        + str(config.negative_positive_ratio)
        + "_itr"
        + str(config.training_iterations)
    )
    # TODO: replace with simple logger.
    create_logger_with_fh(output_folder / ("train_" + log_params_str))

    # TODO: prettify
    logger.info(
        "train_set="
        + config.dataset_name
        + ", lr="
        + str(config.learning_rate)
        + ", bs="
        + str(config.batch_size)
        + ", ratio=1:"
        + str(config.negative_positive_ratio)
        + ", itr="
        + str(config.training_iterations)
        + ", hidden_s="
        + str(config.hidden_size)
        + ", weight_decay="
        + str(config.weight_decay),
    )

    _event_train_feat, _event_validation_feat, _pairwise_model = init_basic_training_resources(
        train_embeddings_path=PROJECT_ROOT / config.train_embeddings_file,
        dev_embeddings_path=PROJECT_ROOT / config.dev_embeddings_file,
        dataset_type=DatasetEnum(config.dataset_name),
        train_positive_file_path=PROJECT_ROOT / config.train_positive_file,
        train_negative_file_path=PROJECT_ROOT / config.train_negative_file,
        dev_positive_file_path=PROJECT_ROOT / config.dev_positive_file,
        dev_negative_file_path=PROJECT_ROOT / config.dev_negative_file,
        ratio=config.negative_positive_ratio,
        hidden_size=config.hidden_size,
        use_cuda=config.use_cuda,
    )

    train_pairwise(
        _pairwise_model,
        _event_train_feat,
        _event_validation_feat,
        config.batch_size,
        config.training_iterations,
        config.learning_rate,
        model_out=config.model_name,
        weight_decay=config.weight_decay,
    )


if __name__ == "__main__":
    main()
