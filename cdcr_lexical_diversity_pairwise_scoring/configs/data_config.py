from dataclasses import dataclass
from pathlib import Path

from cdcr_lexical_diversity_pairwise_scoring.dataobjs.topics import ScopeConfig
from cdcr_lexical_diversity_pairwise_scoring.dataobjs.ucdcr_dataset import (
    DatasetSetting,
    MentionPairStrategy,
)


@dataclass
class DataConfig:
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
    max_pairs_dev: int | None
    # max_pairs_test is always set to None
    ##
    train_dataset_names: list[str]  # Shared with dev_dataset_name
    # dev_dataset_names is the same as train_dataset_names
    test_dataset_names: list[str]
