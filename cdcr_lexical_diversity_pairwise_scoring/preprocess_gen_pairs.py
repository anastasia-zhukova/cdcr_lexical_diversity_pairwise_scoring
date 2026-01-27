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

import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

import hydra
from hydra.core.config_store import ConfigStore

from cdcr_lexical_diversity_pairwise_scoring import logger
from cdcr_lexical_diversity_pairwise_scoring.constants import PROJECT_ROOT
from cdcr_lexical_diversity_pairwise_scoring.dataobjs.dataset import DataSet, DatasetEnum, Split, uCDCRDataSet
from cdcr_lexical_diversity_pairwise_scoring.dataobjs.topics import ScopeConfig


@dataclass
class Config:
    dataset_folder: Path
    setting: str
    type_of_pairs: str
    ratio: int
    max_pairs_train: int
    train_scope: ScopeConfig
    train_dataset_names: List[str]
    dev_scope: ScopeConfig
    max_pairs_dev: int
    test_scope: ScopeConfig
    test_dataset_names: List[str]


cs = ConfigStore.instance()
cs.store(name="preprocess_gen_pairs_config", node=Config)


@hydra.main(version_base="1.3", config_path=str(PROJECT_ROOT / "config"), config_name="preprocess_gen_pairs_config")
def main(config: Config) -> None:
    logger.debug(config)
    random.seed(0)

    for split in [Split.train, Split.dev, Split.test]:
        dataset = uCDCRDataSet(config, split)

        if split == Split.train:
            scope = config.train_scope.value
            max_pairs = config.max_pairs_train
            type_of_pairs = config.type_of_pairs
        elif split.value == Split.dev:
            scope = config.dev_scope.value
            max_pairs = config.max_pairs_dev
            type_of_pairs = "all"
        else:
            scope = config.test_scope.value
            max_pairs = "all"
            type_of_pairs = "all"

        save_path = PROJECT_ROOT / "resources" / f"{split.value}_{type_of_pairs}_{scope}_{max_pairs}_{'-'.join(dataset.dataset_components)}.pickle"
        if save_path.exists():
            logger.info(f"A dataset for {split} with the same config already exists. Skipped.")

        logger.info(f"Generating pairs for file: {split}")
        # TODO
        dataset.generate_pairs()
        # TODO
        dataset.save_dataset(save_path)

    # TODO save dataset
    # dirname = Path(dataset_folder).parent
    # basename = Path(dataset_folder).stem
    #
    # positive_file_path = dirname / f"{basename}_PosPairs.pickle"
    # with positive_file_path.open("wb") as file:
    #     pickle.dump(positive, file)
    # logger.info(f"Saved positive pairs in {positive_file_path}.")
    #
    # negative_file_path = dirname / f"{basename}_NegPairs.pickle"
    # with negative_file_path.open("wb") as file:
    #     pickle.dump(negative, file)
    # logger.info(f"Saved negative pairs in {negative_file_path}.")


if __name__ == "__main__":
    main()
