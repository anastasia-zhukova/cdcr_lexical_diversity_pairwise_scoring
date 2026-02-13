import random
from dataclasses import dataclass
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore

from cdcr_lexical_diversity_pairwise_scoring import logger
from cdcr_lexical_diversity_pairwise_scoring.constants import PROJECT_ROOT
from cdcr_lexical_diversity_pairwise_scoring.dataobjs.topics import ScopeConfig
from cdcr_lexical_diversity_pairwise_scoring.dataobjs.ucdcr_dataset import (
    DatasetSetting,
    MentionPairStrategy,
    Split,
    uCDCRDataSet,
)


# TODO Sergei: fix the configs to be pluggable for each experiment
CONFIG_NAME = "preprocess_E4-2-4"


@dataclass
class PreprocessGenPairsConfig:
    dataset_folder: Path  # Global for all run
    setting: DatasetSetting  # Global for all run
    ratio: int  # Global for all run
    ##
    train_scope: ScopeConfig
    dev_scope: ScopeConfig
    test_scope: ScopeConfig
    ##
    type_of_pairs: MentionPairStrategy
    dev_type_of_pairs: MentionPairStrategy
    # test_type_of_pairs is always set to MentionPairStrategy.all
    ##
    max_pairs_train: int
    max_pairs_dev: int
    # max_pairs_test is always set to None
    ##
    train_dataset_names: list[str]  # Shared with dev_dataset_name
    # dev_dataset_names is the same as train_dataset_names
    test_dataset_names: list[str]


cs = ConfigStore.instance()
cs.store(name=CONFIG_NAME, node=PreprocessGenPairsConfig)


@hydra.main(version_base="1.3", config_path=str(PROJECT_ROOT / "config"), config_name=CONFIG_NAME)
def main(config: PreprocessGenPairsConfig) -> None:
    logger.debug(config)
    random.seed(0)
    dataset_dict = {}

    if config.setting == DatasetSetting.single:
        if len(config.train_dataset_names) > 1:
            raise ValueError(f"There should only be one train/dev dataset for {DatasetSetting.single} setting!")
        if len(config.test_dataset_names) > 1:
            raise ValueError(f"There should only be one test dataset for {DatasetSetting.single} setting!")
        if config.train_dataset_names != config.test_dataset_names:
            raise ValueError(
                f"For {DatasetSetting.single} setting, train/dev dataset and test dataset should be the same,"
                f" found {config.train_dataset_names} and {config.test_dataset_names}"
            )
    elif config.setting == DatasetSetting.excluding_target:
        for dataset_name in config.test_dataset_names:
            if dataset_name in config.train_dataset_names:
                raise ValueError(f"Target dataset is detected in train/ datasets list: {dataset_name}")

    if config.test_scope == ScopeConfig.corpus and len(config.test_dataset_names) > 1:
        raise ValueError(
            f"The configuration  of the {config.test_scope} and more than one test dataset"
            f" can't be possible for the dataset preparation. Change the scope to {ScopeConfig.dataset}.",
        )

    for split in [Split.train, Split.dev, Split.test]:
        logger.info(f"Generating pairs for {split.value} split")
        dataset = uCDCRDataSet.build_from_config(config, split)
        dataset.generate_pairs()
        # TODO: refactor inside dataset
        save_path = PROJECT_ROOT / "resources/datasets" / f"{dataset.dataset_name}.pkl"
        dataset.save_dataset(save_path)
        dataset_dict[split.value] = str(save_path)

    logger.info(f"The paths to the created datasets for the current experiment config is saved")


if __name__ == "__main__":
    main()
