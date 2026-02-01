import pickle
from pathlib import Path

import torch
from transformers import RobertaModel, RobertaTokenizer

from cdcr_lexical_diversity_pairwise_scoring import logger
from cdcr_lexical_diversity_pairwise_scoring.dataobjs.mention_data import MentionuCDCR
from cdcr_lexical_diversity_pairwise_scoring.constants import LANGUAGE_MODEL


class EmbedTransformersGenerics:
    def __init__(
        self,
        max_surrounding_context: int = -1,
        finetune: bool = False,
        use_cuda: bool = True,
        bert_model_name: str = LANGUAGE_MODEL,
    ):

        self.max_surrounding_context = max_surrounding_context
        self.use_cuda = use_cuda
        self.finetune = finetune
        self.bert_model_name = bert_model_name

        self.model = RobertaModel.from_pretrained(self.bert_model_name)
        self.tokenizer = RobertaTokenizer.from_pretrained(self.bert_model_name)

        if self.use_cuda:
            self.model.cuda()

    def get_mention_full_rep(self, mention: MentionuCDCR):
        context_ids, m_start_id, m_end_id = self.encode_mention(mention)

        if self.use_cuda:
            context_ids = context_ids.cuda()

        if not self.finetune:
            with torch.no_grad():  # TODO: this should be handled outside
                last_hidden_span = self.model(context_ids).last_hidden_state
        else:
            last_hidden_span = self.model(context_ids).last_hidden_state

        mention_hidden_span = last_hidden_span.view(last_hidden_span.shape[1], -1)[m_start_id:m_end_id]
        return mention_hidden_span, mention_hidden_span[0], mention_hidden_span[-1], mention_hidden_span.shape[0]

    @staticmethod
    def extract_mention_surrounding_context(mention: MentionuCDCR):
        tokens_indexes = mention.tokens_number_context
        context = mention.mention_context
        start_mention_index = tokens_indexes[0]
        end_mention_index = tokens_indexes[-1] + 1
        assert len(tokens_indexes) == len(mention.tokens_number)

        ret_context_before = context[:start_mention_index]
        ret_mention = context[start_mention_index:end_mention_index]
        ret_context_after = context[end_mention_index:]

        assert ret_mention == mention.tokens_text
        assert ret_context_before + ret_mention + ret_context_after == mention.mention_context

        return ret_context_before, ret_mention, ret_context_after

    def encode_mention(self, mention: MentionuCDCR):
        context_before_str, mention_span_str, context_after_str = (
            EmbedTransformersGenerics.extract_mention_surrounding_context(mention)
        )

        context_before = (
            self.tokenizer.encode(" ".join(context_before_str), add_special_tokens=False)
            if len(context_before_str) > 0
            else []
        )
        context_after = (
            self.tokenizer.encode(" ".join(context_after_str), add_special_tokens=False)
            if len(context_after_str) > 0
            else []
        )

        if self.max_surrounding_context != -1:
            if len(context_before) > self.max_surrounding_context:
                context_before = context_before[-self.max_surrounding_context + 1 :]
            if len(context_after) > self.max_surrounding_context:
                context_after = context_after[: self.max_surrounding_context - 1]

        mention_span = self.tokenizer.encode(" ".join(mention_span_str), add_special_tokens=False)

        all_context_tokens = [[self.tokenizer.cls_token_id] + context_before + mention_span + context_after + [self.tokenizer.sep_token_id]]
        all_context_tokens = torch.tensor(all_context_tokens)
        mention_start_index = len(context_before) + 1
        mention_end_index = len(context_before) + len(mention_span) + 1
        return all_context_tokens, mention_start_index, mention_end_index

    @property
    def get_embed_size(self):
        return self.model.config.hidden_size


class EmbedFromFile:
    def __init__(
        self,
        file_to_load: Path,
    ):
        bert_dict = dict()

        with file_to_load.open("rb") as file:
            loaded_file = pickle.load(file)
            bert_dict.update(loaded_file)
        logger.info(f"BERT representation loaded from file: {file_to_load}")

        self.embeddings = list(bert_dict.values())
        self.embed_size = self.embeddings[0][0].shape[1]
        self.embed_key = {k: i for i, k in enumerate(bert_dict.keys())}

    def get_mentions_rep(self, mentions_list):
        embed_list = [self.embeddings[self.embed_key[mention.mention_id]] for mention in mentions_list]
        return embed_list

    def get_embed_size(self):
        return self.embed_size
