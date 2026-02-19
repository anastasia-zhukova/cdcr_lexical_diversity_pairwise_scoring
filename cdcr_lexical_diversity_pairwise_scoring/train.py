import json
import random
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from tqdm import tqdm

from cdcr_lexical_diversity_pairwise_scoring import logger
from cdcr_lexical_diversity_pairwise_scoring.configs import ModelTrainConfig
from cdcr_lexical_diversity_pairwise_scoring.constants import CACHED_VECTOR_PATH, PROJECT_ROOT
from cdcr_lexical_diversity_pairwise_scoring.coref_system.pairwise_model_kenton import PairwiseModelKenton
from cdcr_lexical_diversity_pairwise_scoring.dataobjs.ucdcr_dataset import uCDCRDataSet
from cdcr_lexical_diversity_pairwise_scoring.utils.embed_utils import EmbedFromFile
from cdcr_lexical_diversity_pairwise_scoring.utils.eval_utils import get_confusion_matrix, precision_recall_f1
from cdcr_lexical_diversity_pairwise_scoring.utils.log_utils import create_logger_with_fh


def train_pairwise(
    pairwise_model: PairwiseModelKenton,
    train,
    validation,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    model_out: Path,
):
    loss_func = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(pairwise_model.parameters(), lr, weight_decay=weight_decay)
    dataset_size = len(train)
    model_save_path = ""

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
            pbar.update(len(batch_features))

            if current_batch % 100 == 0:
                pbar.set_postfix(loss=f"{cumulative_loss / current_batch:.10f}")

        pbar.close()
        logger.info(f"Finished training for epoch {epoch}. Final loss: {cumulative_loss / current_batch:.10f}")

        pairwise_model.eval()
        _, _, _, dev_f1 = accuracy_on_dataset("Dev", epoch + 1, pairwise_model, validation, batch_size=batch_size)

        if best_result_so_far < dev_f1:
            logger.info("Found better model saving")
            model_name = model_out.stem + "_iter_" + str(epoch + 1) + model_out.suffix
            model_save_path = model_out.parent / model_name
            torch.save(pairwise_model, model_save_path)
            best_result_so_far = dev_f1

    return best_result_so_far, model_save_path


def accuracy_on_dataset(
    evaluation_set_name: str,
    epoch: int,
    pairwise_model: PairwiseModelKenton,
    features,
    batch_size: int,
):
    all_labels, all_predictions = run_inference(pairwise_model, features, batch_size=batch_size)
    logger.info("Got labels.")
    logger.info(f"Shape true labels: {all_labels}")
    logger.info(f"Shape predicted labels: {all_predictions}")
    accuracy = torch.mean((all_labels == all_predictions).float())
    logger.info("Accuracy calculated")
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
    batch_size: int,
    round_pred: bool = True,
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
    train_dataset: uCDCRDataSet,
    dev_dataset: uCDCRDataSet,
    hidden_size: int,
    use_cuda: bool,
) -> tuple[
    ...,  # TODO
    ...,  # TODO
    PairwiseModelKenton,
    str,
    str,
]:
    embed_utils = EmbedFromFile(embeddings_path, use_cuda)
    pairwise_model = PairwiseModelKenton(embed_utils.embed_size, hidden_size, 1, embed_utils, use_cuda)

    train_feat = train_dataset.get_mix_pairs()
    validation_feat = dev_dataset.get_mix_pairs()

    if use_cuda:
        pairwise_model.cuda()

    return (
        train_feat,
        validation_feat,
        pairwise_model,
        "_".join(train_dataset.dataset_names),
        "_".join(dev_dataset.dataset_names),
    )


cs = ConfigStore.instance()
cs.store(name="train_config", node=ModelTrainConfig)


@hydra.main(version_base="1.3", config_path=str(PROJECT_ROOT / "config"), config_name="train_config")
def main(config: ModelTrainConfig):
    if config.use_cuda:
        torch.cuda.manual_seed(1234)

    data_config_name = HydraConfig.get().runtime.choices["data_config"]

    torch.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)

    start_time = datetime.now()
    dt_string = start_time.strftime("%d%m%Y_%H%M%S")
    output_folder = PROJECT_ROOT / f"checkpoints/{data_config_name}/{dt_string}"
    output_folder.mkdir(parents=True, exist_ok=True)

    train_dataset, dev_dataset, _ = uCDCRDataSet.load_from_config(config.data_config)
    logger.debug(train_dataset.get_mix_pairs())

    log_params_str = (
        f"dc_{data_config_name}"
        + "lr_"
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
        + str(data_config_name)
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

    _event_train_feat, _event_validation_feat, _pairwise_model, _train_names, _dev_names = (
        init_basic_training_resources(
            embeddings_path=CACHED_VECTOR_PATH,
            train_dataset=train_dataset,
            dev_dataset=dev_dataset,
            hidden_size=config.hidden_size,
            use_cuda=config.use_cuda,
        )
    )

    eval_res, best_model_path = train_pairwise(
        _pairwise_model,
        _event_train_feat,
        _event_validation_feat,
        config.batch_size,
        config.training_iterations,
        config.learning_rate,
        model_out=output_folder / "model.pt",
        weight_decay=config.weight_decay,
    )
    run_results = {"eval_dev": eval_res, "model": best_model_path}

    # TODO Sergei proper saving to config with the best model
    with (output_folder / "results.json").open("w") as file:
        json.dump(run_results, file)


if __name__ == "__main__":
    main()
