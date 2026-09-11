import pickle
from dataclasses import dataclass
from pathlib import Path

import torch
from torchtyping import TensorType
from transformers import RobertaModel, RobertaTokenizer

from cdcr_lexical_diversity_pairwise_scoring import logger
from cdcr_lexical_diversity_pairwise_scoring.dataobjs.mention_data import MentionuCDCR


@dataclass
class TensorBuildOutput:
    mention_id_to_tensor_id_mapping: dict[str, int]
    hidden_vectors_long: TensorType["N_special", "D"]
    hidden_offsets_prefix_sums: TensorType["N_plus_1"]
    first_tokens: TensorType["N", "D"]
    last_tokens: TensorType["N", "D"]
    sizes: TensorType["N"]


class EmbedTransformersGenerics:
    def __init__(
        self,
        bert_model_name: str,
        max_surrounding_context: int = -1,
        finetune: bool = False,
        use_cuda: bool = True
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
        files_to_load: Path | list[Path],
        use_cuda: bool,
    ):
        self.files_to_load = files_to_load
        self.use_cuda = use_cuda

        files_to_load = self._ensure_ok_files(files_to_load)
        bert_dict = self._load_prebuilt_bert_representation(files_to_load)
        tensors = self._build_tensors_from_dict(bert_dict)

        self.mention_id_to_tensor_id_mapping = tensors.mention_id_to_tensor_id_mapping

        device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

        self.hidden_vectors_long = tensors.hidden_vectors_long.to(device)
        self.hidden_offsets_prefix_sums = tensors.hidden_offsets_prefix_sums.to(device)
        self.first_tokens = tensors.first_tokens.to(device)
        self.last_tokens = tensors.last_tokens.to(device)
        self.sizes = tensors.sizes.to(device)
        self.embed_size = list(bert_dict.values())[0][0].shape[1]

    def get_mentions_rep(
        self,
        tensor_ids: torch.Tensor,
        target_padding_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        offsets_start = self.hidden_offsets_prefix_sums[tensor_ids]
        offsets_end = self.hidden_offsets_prefix_sums[tensor_ids + 1]

        padded = self.hidden_vectors_long.new_zeros((tensor_ids.numel(), target_padding_size, self.embed_size))
        for i, (a, b) in enumerate(zip(offsets_start.tolist(), offsets_end.tolist())):
            padded[i, : (b - a)] = self.hidden_vectors_long[a:b]

        first = self.first_tokens[tensor_ids]
        last = self.last_tokens[tensor_ids]

        if self.use_cuda:
            return padded.cuda(), first.cuda(), last.cuda()

        return padded, first, last

    def get_mentions_metadata(
        self,
        mentions_list: list[MentionuCDCR],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tensor_ids_required = torch.tensor(
            [self.mention_id_to_tensor_id_mapping[m.mention_id] for m in mentions_list],
            dtype=torch.long,
        )
        sizes = self.sizes[tensor_ids_required]
        if self.use_cuda:
            return tensor_ids_required, sizes.cuda()
        return tensor_ids_required, sizes

    @staticmethod
    def _ensure_ok_files(
        files_to_load: Path | list[Path],
    ) -> list[Path]:
        if isinstance(files_to_load, Path):
            files_to_load = [files_to_load]
        if files_to_load is None or len(files_to_load) == 0:
            raise ValueError

        return files_to_load

    @staticmethod
    def _load_prebuilt_bert_representation(
        files_to_load: list[Path],
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]:

        bert_dict: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = {}

        for single_file_path in files_to_load:
            with single_file_path.open("rb") as file:
                loaded_file = pickle.load(file)
                bert_dict.update(loaded_file)
            logger.info(f"BERT representation loaded from file: {single_file_path}")

        return bert_dict

    @staticmethod
    def _build_tensors_from_dict(
        source_bert_dict: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]],
    ) -> TensorBuildOutput:
        mention_id_to_tensor_id_mapping: dict[str, int] = {}
        hidden_vectors_long: list[torch.Tensor] = []
        hidden_offsets_prefix_sums: list[int] = [0]
        first_tokens: list[torch.Tensor] = []
        last_tokens: list[torch.Tensor] = []
        sizes: list[int] = []

        for i, (mention_id, tensors_tuple) in enumerate(source_bert_dict.items()):
            mention_id_to_tensor_id_mapping[mention_id] = i
            hidden_vectors_long.append(tensors_tuple[0])
            hidden_offsets_prefix_sums.append(hidden_offsets_prefix_sums[-1] + len(tensors_tuple[0]))
            first_tokens.append(tensors_tuple[1])
            last_tokens.append(tensors_tuple[2])
            sizes.append(tensors_tuple[3])

        return TensorBuildOutput(
            mention_id_to_tensor_id_mapping=mention_id_to_tensor_id_mapping,
            hidden_vectors_long=torch.cat(hidden_vectors_long, dim=0),
            hidden_offsets_prefix_sums=torch.tensor(hidden_offsets_prefix_sums),
            first_tokens=torch.stack(first_tokens, dim=0),
            last_tokens=torch.stack(last_tokens, dim=0),
            sizes=torch.tensor(sizes),
        )

    def get_embed_size(self):
        return self.embed_size
