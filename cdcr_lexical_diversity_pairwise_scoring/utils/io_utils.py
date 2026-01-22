import json
import os
from pathlib import Path

from cdcr_lexical_diversity_pairwise_scoring.constants import PROJECT_ROOT
from cdcr_lexical_diversity_pairwise_scoring.dataobjs.mention_data import MentionData


def write_coref_scorer_results(
    mentions: list[MentionData],
    output_file: Path,
) -> None:
    mentions.sort(key=lambda x: x.mention_index)

    with output_file.open("w") as file:
        file.write("#begin document (ECB+/ecbplus_all); part 000")
        for mention in mentions:
            file.write(f"\nECB+/ecbplus_all\t({mention.predicted_coref_chain})")
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
