"""Usage:
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
import json
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from cdcr_lexical_diversity_pairwise_scoring import logger
from cdcr_lexical_diversity_pairwise_scoring.constants import PROJECT_ROOT, DEFAULT_RATIO, CONFIG_NAME
from cdcr_lexical_diversity_pairwise_scoring.dataobjs.dataset import Split, uCDCRDataSet, MentionPairStrategy, DatasetSetting
from cdcr_lexical_diversity_pairwise_scoring.dataobjs.topics import ScopeConfig
from cdcr_lexical_diversity_pairwise_scoring.utils.io_utils import create_dataset_save_path, get_dataset_info_save_path


@dataclass
class Config:
    dataset_folder: Path
    setting: DatasetSetting
    type_of_pairs: MentionPairStrategy
    ratio: int
    max_pairs_train: int
    train_scope: ScopeConfig
    train_dataset_names: List[str]
    dev_dataset_names: List[str]
    dev_scope: ScopeConfig
    dev_type_of_pairs: MentionPairStrategy
    max_pairs_dev: int
    test_scope: ScopeConfig
    test_dataset_names: List[str]
    language_model: str
    batch_size: int = MISSING
    learning_rate: float = MISSING
    training_iterations: int = MISSING
    use_cuda: bool = MISSING
    fine_tune: bool = MISSING
    weight_decay: float = MISSING
    hidden_size: int = MISSING


cs = ConfigStore.instance()
cs.store(name=CONFIG_NAME, node=Config)


@hydra.main(version_base="1.3", config_path=str(PROJECT_ROOT / "config"), config_name=CONFIG_NAME)
def main(config: Config) -> None:
    logger.debug(config)
    random.seed(0)
    dataset_dict = {}

    for split in [Split.train, Split.dev, Split.test]:
        logger.info(f"Generating pairs for {split.value} split")
        config.dataset_folder = PROJECT_ROOT / config.dataset_folder
        dataset = uCDCRDataSet(config, split)

        if split == Split.train:
            scope = config.train_scope
            max_pairs = config.max_pairs_train
            type_of_pairs = config.type_of_pairs
            ratio = config.ratio

        elif split == Split.dev:
            scope = config.dev_scope
            max_pairs = config.max_pairs_dev
            type_of_pairs = config.dev_type_of_pairs
            if type_of_pairs == MentionPairStrategy.all:
                ratio = -1
            else:
                ratio = config.ratio
            # type_of_pairs = MentionPairStrategy.all

        else:
            scope = config.test_scope
            max_pairs = None
            ratio = -1
            type_of_pairs = MentionPairStrategy.all

        save_path = create_dataset_save_path(split, type_of_pairs, scope, max_pairs, ratio, dataset.dataset_components)
        if save_path.exists():
            logger.info(f"A dataset for {split.value} with the same config (path {str(save_path)}) already exists. Skipped.")
            dataset_dict[split.value] = str(save_path)
            continue

        dataset.generate_pairs()
        dataset.save_dataset(save_path)
        dataset_dict[split.value] = str(save_path)

    # save all paths to the created datasets for this experiment
    save_path = get_dataset_info_save_path()
    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(dataset_dict, file)

    logger.info(f"The paths to the created datasets for the current experiment config is saved in: {save_path}")


if __name__ == "__main__":
    # TODO Sergei: proper reading specific config to each experiment
    main()
