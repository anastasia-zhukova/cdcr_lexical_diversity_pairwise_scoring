import json

import logging
import pickle
import time
import os
from pathlib import Path
from os import path
from typing import List

from cdcr_lexical_diversity_pairwise_scoring.dataobjs.mention_data import MentionData

logger = logging.getLogger(__name__)


LIBRARY_PATH = Path(path.realpath(__file__)).parent.parent.parent


def load_json_file(file_path):
    """load a file into a json object"""
    try:
        with open(file_path, encoding="utf-8") as small_file:
            return json.load(small_file)
    except OSError as e:
        print(e)
        print("trying to read file in blocks")
        with open(file_path, encoding="utf-8") as big_file:
            json_string = ""
            while True:
                block = big_file.read(64 * (1 << 20))  # Read 64 MB at a time;
                json_string = json_string + block
                if not block:  # Reached EOF
                    break
            return json.loads(json_string)


def write_coref_scorer_results(
    mentions: list[MentionData],
    output_file: str,
) -> None:
    """
    :param mentions: List[MentionData]
    :param output_file: str
    :return:
    """
    mentions.sort(key=lambda x: x.mention_index)
    output = open(output_file, "w")
    output.write("#begin document (ECB+/ecbplus_all); part 000\n")
    for mention in mentions:
        output.write("ECB+/ecbplus_all\t" + "(" + str(mention.predicted_coref_chain) + ")\n")
    output.write("#end document")
    output.close()


def write_mention_to_json(out_file: str, mentions: List):
    mentions.sort(key=lambda x: x.mention_index)
    with open(out_file, "w+") as output:
        json.dump(mentions, output, default=default, indent=4, sort_keys=True, ensure_ascii=False)


def create_and_get_path(path_to_create):
    path_to = str(LIBRARY_PATH) + "/" + path_to_create
    if not os.path.exists(path_to):
        os.makedirs(path_to)
    return path_to


def default(o):
    return o.__dict__
