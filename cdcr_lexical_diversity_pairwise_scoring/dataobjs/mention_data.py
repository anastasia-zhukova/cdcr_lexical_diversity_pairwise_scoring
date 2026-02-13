import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

from cdcr_lexical_diversity_pairwise_scoring.utils.string_utils import SpacySyntaxAnalyzer


@dataclass(slots=True)
class MentionuCDCR:
    coref_chain: str
    mention_id: str
    tokens_str: str
    description: str
    coref_type: str
    mention_type: str
    mention_full_type: str
    tokens_text: list[str]
    tokens_number: list[int]
    mention_head: str
    mention_head_id: int
    mention_head_pos: str
    mention_head_lemma: str
    mention_ner: str
    sent_id: int
    topic: str
    topic_id: str
    subtopic_id: str
    subtopic: str
    doc_id: str
    doc: str
    mention_context: list[str]
    mention_context_start_end_id: list[int]
    tokens_number_context: list[int]
    mention_head_id_context: int
    is_singleton: bool
    conll_doc_key: str
    # --
    mention_index: int = None
    dataset: str = None

    def __repr__(self):
        return f"{self.tokens_str}"

    @classmethod
    def read_mentions(cls, mentions: list[dict]) -> list[Self]:
        return [cls(**mention, mention_index=i) for i, mention in enumerate(mentions)]


class MentionData:
    def __init__(
        self,
        mention_id,
        topic_id: str,
        doc_id: str,
        sent_id: int,
        tokens_numbers: list[int],
        tokens_str: str,
        mention_context: list[str],
        mention_head: str,
        mention_head_lemma: str,
        coref_chain: str,
        mention_type: Literal["HUM", "NON", "TIM", "LOC", "ACT", "NEG"] = "NA",  # TODO: what about na?
        coref_link: str = "NA",  # TODO
        predicted_coref_chain: str = None,
        mention_pos: str = None,
        mention_ner: str = None,
        mention_index: int = -1,
        gen_lemma: bool = False,
    ) -> None:
        """Object represent a mention

        Args:
            topic_id: str topic ID
            doc_id: str document ID
            sent_id: int sentence number
            tokens_numbers: List[int] - tokens numbers
            mention_context: List[str] - list of tokens strings
            coref_chain: str
            mention_type: str one of (HUM/NON/TIM/LOC/ACT/NEG)
            predicted_coref_chain: str (should be field while evaluated)
            mention_pos: str
            mention_ner: str
            mention_index: in case order is of value (default = -1)

        """
        self.tokens_str = tokens_str
        self.mention_context = mention_context
        if not mention_head and not mention_head_lemma:
            if gen_lemma:
                self.mention_head, self.mention_head_lemma, self.mention_head_pos, self.mention_ner = (
                    SpacySyntaxAnalyzer.find_head_lemma_pos_ner(str(tokens_str))
                )
        else:
            self.mention_head = mention_head
            self.mention_head_lemma = mention_head_lemma
            self.mention_head_pos = mention_pos
            self.mention_ner = mention_ner

        self.topic_id = topic_id
        self.doc_id = doc_id
        self.sent_id = sent_id
        self.tokens_number = tokens_numbers
        self.mention_type = mention_type
        self.coref_chain = coref_chain
        self.predicted_coref_chain = predicted_coref_chain
        self.coref_link = coref_link

        if mention_id is None:
            self.mention_id = self.gen_mention_id()
        else:
            self.mention_id = str(mention_id)

        self.mention_index = mention_index

    @classmethod
    def _read_json_mention_data_line(cls, mention_line: dict) -> "MentionData":
        mention_text = mention_line["tokens_str"]
        mention_id = mention_line.get("mention_id")
        topic_id = mention_line.get("topic_id")
        coref_chain = mention_line.get("coref_chain")
        doc_id = mention_line.get("doc_id")
        sent_id = mention_line.get("sent_id")
        tokens_numbers = mention_line.get("tokens_number")
        mention_context = mention_line.get("mention_context")
        mention_type = mention_line.get("mention_type")
        predicted_coref_chain = mention_line.get("predicted_coref_chain")
        mention_index = mention_line.get("mention_index")
        coref_link = mention_line.get("coref_link")

        mention_head = mention_line.get("mention_head")
        mention_head_lemma = mention_line.get("mention_head_lemma")
        mention_pos = mention_line.get("mention_head_pos")
        mention_ner = mention_line.get("mention_ner")

        if mention_head is None or mention_head_lemma is None:
            mention_head, mention_head_lemma, mention_pos, mention_ner = SpacySyntaxAnalyzer.find_head_lemma_pos_ner(
                str(mention_text),
            )

        mention_data = cls(
            mention_id,
            topic_id,
            doc_id,
            sent_id,
            tokens_numbers,
            mention_text,
            mention_context,
            mention_head=mention_head,
            mention_head_lemma=mention_head_lemma,
            coref_chain=coref_chain,
            mention_type=mention_type,
            coref_link=coref_link,
            predicted_coref_chain=predicted_coref_chain,
            mention_pos=mention_pos,
            mention_ner=mention_ner,
            mention_index=mention_index,
        )
        return mention_data

    def gen_mention_id(self) -> str:
        if self.doc_id and self.sent_id is not None and self.tokens_number:
            tokens_ids = [str(self.doc_id), str(self.sent_id)]
            tokens_ids.extend([str(token_id) for token_id in self.tokens_number])
            return "_".join(tokens_ids)

        return "_".join(self.tokens_str.split())

    @classmethod
    def read_mentions_json_to_mentions_data_list(
        cls,
        mentions_json_file: Path,
    ) -> list["MentionData"]:
        """Args:
            mentions_json_file: the path of the mentions json file to read

        Returns:
            List[MentionData]

        """
        with mentions_json_file.open("r", encoding="utf-8") as file:
            all_mentions_only = json.load(file)

        running_index = 1
        mentions = []
        for mention_line in all_mentions_only:
            try:
                mention_data = cls._read_json_mention_data_line(mention_line)
            except Exception as e:
                raise Exception(f"Failed to read json line: {mention_line}") from e

            mention_data.mention_index = running_index
            mentions.append(mention_data)
            running_index += 1

        return mentions
