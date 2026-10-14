import json
from pathlib import Path
from typing import Literal, List

from cdcr_lexical_diversity_pairwise_scoring.utils.string_utils import SpacySyntaxAnalyzer


class MentionuCDCR:
    """
    {coref_chain	"ORGVHAd9DmrNTWrtWpoyrjitp"
    mention_id	"MvxX42fHU3oBFFwBYKSRvn"
    tokens_str	"UNHCR"
    description	"UNHCR"
    coref_type	"IDENTITY"
    mention_type	"ORG"
    mention_full_type	"ORG"
    tokens_text	[ "UNHCR" ]
    tokens_number	[ 35 ]
    mention_head	"UNHCR"
    mention_head_id	35
    mention_head_pos	"PROPN"
    mention_head_lemma	"UNHCR"
    mention_ner	"ORG"
    sent_id	5
    topic	"0_bomb_explosion_kidnap"
    topic_id	"0"
    subtopic_id	"4_tajikistan"
    subtopic	"4_Tajikistan_hostages"
    doc_id	"363541"
    doc	"363541"
    mention_context	(220)[ "U.N.", "council", "condemns", "hostage", "-", "taking", "in", "Tajikistan", ".", "UNITED", … ]
    mention_context_start_end_id	[ 12, 231 ]
    tokens_number_context	[ 103 ]
    mention_head_id_context	103
    is_singleton	false
    conll_doc_key	"0/4_tajikistan/363541"}
    """
    def __init__(self, data: dict):
        for key, value in data.items():
            setattr(self, key, value)
        self.mention_index = None

    def __repr__(self):
        return f"{self.dataset}_{self.tokens_str}"

    @classmethod
    def read_mentions(cls, mentions: List[dict]) -> list["MentionuCDCR"]:
        mentions_classes = []
        for m_i, mention_dict in enumerate(mentions):
            mention = cls(mention_dict)
            mention.mention_index = m_i
            mentions_classes.append(mention)

        return mentions_classes


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
