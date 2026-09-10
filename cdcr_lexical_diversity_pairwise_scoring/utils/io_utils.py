import json
import os
from typing import Union, List
from pathlib import Path

from cdcr_lexical_diversity_pairwise_scoring.constants import PROJECT_ROOT, CONFIG_NAME
from cdcr_lexical_diversity_pairwise_scoring.dataobjs.mention_data import MentionData, MentionuCDCR
from cdcr_lexical_diversity_pairwise_scoring.dataobjs.dataset import Split, MentionPairStrategy, ScopeConfig


def create_dataset_save_path(split: Split, type_of_pairs: MentionPairStrategy, scope: ScopeConfig, max_pairs: Union[int, None], ratio: int, dataset_components: List[str]):
    return PROJECT_ROOT / "experiment_cache_results" / CONFIG_NAME / f"{split.value}_{type_of_pairs.value}_{scope.value}_{max_pairs}_{ratio}_{'-'.join(dataset_components)}.pickle"

def get_dataset_info_save_path():
    return PROJECT_ROOT / "experiment_cache_results" / CONFIG_NAME / "datasets.json"

def get_model_info_save_path():
    return PROJECT_ROOT / "experiment_cache_results" / CONFIG_NAME / "model.json"

def get_conll_files_root_path():
    return PROJECT_ROOT / "experiment_cache_results" / CONFIG_NAME / "predictions_conll"

def get_predicted_cluster_path():
    return PROJECT_ROOT / "experiment_cache_results" / CONFIG_NAME / "predictions_json" / "results.json"

def get_evaluation_result_path():
    return PROJECT_ROOT / "evaluation_results"

def get_encoding_cache_file(split: str, dataset: str, language_model: str):
    return PROJECT_ROOT / "experiment_cache_results" / f"cached_{split}_{dataset}_{language_model}.pickle"


def write_coref_scorer_results(
    mentions: list[MentionuCDCR],
    output_file: Path,
    topic_id: str,
    save_predicted: True
) -> None:
    mentions.sort(key=lambda x: x.mention_index)

    with output_file.open("w") as file:
        file.write(f"#begin document ({topic_id}); part 000")
        for mention in mentions:
            if save_predicted:
                file.write(f"\n{topic_id}\t({mention.predicted_coref_chain})")
            else:
                file.write(f"\n{topic_id}\t({mention.coref_chain})")
        file.write("\n#end document")


def write_coref_scorer_results_simple(
    chain_values_per_mention: list[str] | list[int],
    mentions: list[str],
    mention_type: str,
    output_file: Path,
    dataset: str,
    topic_id: str,
    predictions: bool = True
) -> None:
    if predictions:
        file_path = output_file / "response.conll"
    else:
        file_path = output_file / "key.conll"
    with file_path.open("w") as file:
        file.write(f"#begin document ({dataset}/{mention_type}/{topic_id}); part 000")

        for mention_id, chain in zip(mentions, chain_values_per_mention):
            # ECBplusMETAm/entities/36ecb
            file.write(f"\n{dataset}/{mention_type}/{topic_id}\t{mention_id}\t({chain})")
        file.write("\n#end document")


def write_mention_to_json(out_file: str, mentions: list):
    mentions.sort(key=lambda x: x.mention_index)
    with open(out_file, "w+") as output:
        json.dump(mentions, output, default=lambda x: x.__dict__, indent=4, sort_keys=True, ensure_ascii=False)


def create_and_get_path(path_to_create):
    path_to = PROJECT_ROOT / path_to_create
    if not os.path.exists(path_to):
        os.makedirs(path_to)
    return path_to
