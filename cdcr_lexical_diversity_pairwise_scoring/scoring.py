import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from cdcr_lexical_diversity_pairwise_scoring import logger
from cdcr_lexical_diversity_pairwise_scoring.constants import PROJECT_ROOT
from cdcr_lexical_diversity_pairwise_scoring.utils.io_utils import get_experiment_name


MUC = "_MUC"
B3 = "_B3"
CEAF_M = "_CEAF_M"
CEAF_E = "_CEAF_E"
BLANC = "_BLANC"
CONLL = "_CONLL"
P = "P"
R = "R"
F1 = "F1"
TRUE_LABEL, PRED_LABEL = "label_true", "label_pred"
SPLIT = "split"

CONFIG_NAME = "preprocess_test"


def run_scorer(key_file_path: Path, response_file_path: Path):
    result_file = PROJECT_ROOT / "resources" / "conll_res.txt"

    scorer_command = (
        f'perl {PROJECT_ROOT / "scorer" / "scorer.pl"} '
        f"all {key_file_path} "
        f"{response_file_path} none > {result_file} \n"
    )

    processes = []
    # LOGGER.info('Run CoNLL scorer perl command for CDCR')
    processes.append(subprocess.Popen(scorer_command, shell=True))
    while processes:
        status = processes[0].poll()
        if status is not None:
            processes.pop(0)

    # LOGGER.info('Running CoNLL scorers has been completed.')

    f1_dict = {}
    metrics_list = [MUC, B3, CEAF_M, CEAF_E, BLANC]
    params = [R, P, F1]
    i = 0
    with open(result_file) as ins:
        for line in ins:
            new_line = line.strip()
            if new_line.find("F1:") != -1:
                if i >= len(metrics_list):
                    break
                f1_dict[metrics_list[i]] = {}
                if new_line.find("Coreference") != -1:
                    j = 0
                    for value in new_line.replace("\t", " ").split(" "):
                        if "%" not in value:
                            continue
                        param = params[j]
                        f1_dict[metrics_list[i]][param] = round(float(value[:-1]), 2)
                        j += 1
                    i += 1

    # metrics_df = pd.DataFrame()
    avg_f1 = []
    # conll_count = 0
    output_dict = {}
    for metrics_name, metrics_vals in f1_dict.items():
        if metrics_name not in [MUC, B3, CEAF_E]:
            continue

        avg_f1.append(metrics_vals[F1])
        for m, v in metrics_vals.items():
            output_dict[m + metrics_name] = float(format(v, ".3f"))

    if avg_f1:
        f1_conll = np.mean(avg_f1)
    else:
        logger.warning("CoNLL score was not calculated. Most likely, perl is not installed.")
        f1_conll = 0

    output_dict[F1 + CONLL] = float(format(f1_conll, ".3f"))
    return output_dict


def main(config_name: str = None):
    """Computes CoNLL scores for the experiment that we want or for everything"""
    now_ = datetime.now()
    save_filename = f'{now_.strftime("%Y-%m-%d_%H-%M-%S")}_conll_evaluation.csv'
    save_filename_topics = f'{now_.strftime("%Y-%m-%d_%H-%M-%S")}_conll_evaluation_topics.csv'
    summary_folder = PROJECT_ROOT / "evaluation_results" / "output_files"
    result_path = PROJECT_ROOT / "evaluation_results" / "input_files"

    if config_name is not None:
        experiment_target = [get_experiment_name(config_name)]
    else:
        # score all files with experiments
        experiment_target = [item.name for item in result_path.iterdir()]

    summary_df = pd.DataFrame()
    summary_df_subtopic = pd.DataFrame()

    for experiment_path in result_path.iterdir():
        experiment = experiment_path.name

        if experiment not in experiment_target:
            continue

        logger.info(f"Scoring experiment {experiment}")

        for dataset_path in experiment_path.iterdir():
            dataset = dataset_path.name

            for pair_type_path in dataset_path.iterdir():
                pair_type = pair_type_path.name
                pair_topic_df = pd.DataFrame()

                for topic_path in pair_type_path.iterdir():
                    topic = topic_path.name

                    summary_dict = {
                        "experiment": experiment_path.name,
                        "dataset": dataset_path.name,
                        "topic": topic_path.name,
                        "mention_type": pair_type_path.name,
                    }
                    key_path = topic_path / "key.key_conll"
                    response_path = topic_path / "response.response_conll"
                    conll_f1_dict = run_scorer(key_file_path=key_path, response_file_path=response_path)
                    summary_dict.update(conll_f1_dict)
                    pair_topic_df = pd.concat(
                        [
                            pair_topic_df,
                            pd.DataFrame(summary_dict, index=[f"{experiment}\\{dataset}\\{pair_type}\\{topic}"]),
                        ],
                        axis=0,
                    )

                summary_df_subtopic = pd.concat([summary_df_subtopic, pair_topic_df])
                summary_df = pd.concat(
                    [
                        summary_df,
                        pd.DataFrame(pair_topic_df.mean(), columns=[f"{experiment}\\{dataset}\\{pair_type}\\all"]).T,
                    ],
                    axis=0,
                )
                summary_df.to_csv(summary_folder / save_filename)
                summary_df_subtopic.to_csv(summary_folder / save_filename_topics)

    summary_df.to_csv(summary_folder / save_filename)
    save_filename_topics = f'{now_.strftime("%Y-%m-%d_%H-%M-%S")}_conll_evaluation_topics.csv'
    summary_df_subtopic.to_csv(summary_folder / save_filename_topics)


if __name__ == "__main__":
    # TODO Sergei fix reading the config names :) Here can be optional (None altenatively) if I want to run the scorer for all expriment data that I have
    config_name = CONFIG_NAME
    main(config_name)
