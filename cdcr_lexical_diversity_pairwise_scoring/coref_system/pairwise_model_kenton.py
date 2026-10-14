import torch
from torch import nn


class PairwiseModelKenton(nn.Module):
    def __init__(self, f_in_dim, f_hidden_dim, f_out_dim, embeddings_holder, use_cuda):
        super(PairwiseModelKenton, self).__init__()
        self.W = nn.Linear(f_hidden_dim, f_out_dim)
        self.pairwise = self.get_sequential(9 * f_in_dim, f_hidden_dim)
        self.attend = self.get_sequential(embeddings_holder.get_embed_size(), f_hidden_dim)
        self.w_alpha = nn.Linear(f_hidden_dim, 1)
        self.embeddings_holder = embeddings_holder
        self.use_cuda = use_cuda
        self.__name__ = "PairwiseModelKenton"

    @staticmethod
    def get_sequential(ind, hidd):
        return nn.Sequential(
            nn.Linear(ind, hidd),
            nn.ReLU(),
            nn.Linear(hidd, hidd),
            nn.ReLU(),
        )

    def forward(self, batch_features: list, bs):
        embedded_features, gold_labels = self._get_bert_rep(batch_features, bs)
        prediction = self.W(self.pairwise(embedded_features))
        return prediction, gold_labels

    def predict(self, batch_features, bs):
        output, gold_labels = self.__call__(batch_features, bs)
        prediction = torch.sigmoid(output)
        return prediction, gold_labels

    def _get_bert_rep(self, batch_features, batch_size=32):
        # Get embeddings metadata from mentions
        mentions1, mentions2 = zip(*batch_features)
        tensor_ids_1, mention_embeddings_size_1 = self.embeddings_holder.get_mentions_metadata(mentions1)
        tensor_ids_2, mention_embeddings_size_2 = self.embeddings_holder.get_mentions_metadata(mentions2)

        max_mention_embeddings_size = max(mention_embeddings_size_1.max(), mention_embeddings_size_2.max())

        # Get embeddings themselves
        hiddens1_pad, first1_tok, last1_tok = self.embeddings_holder.get_mentions_rep(
            tensor_ids_1,
            max_mention_embeddings_size,
        )
        hiddens2_pad, first2_tok, last2_tok = self.embeddings_holder.get_mentions_rep(
            tensor_ids_2,
            max_mention_embeddings_size,
        )

        first1_tok = first1_tok.reshape(batch_size, -1)
        first2_tok = first2_tok.reshape(batch_size, -1)
        last1_tok = last1_tok.reshape(batch_size, -1)
        last2_tok = last2_tok.reshape(batch_size, -1)

        h = torch.cat([hiddens1_pad, hiddens2_pad], dim=0).contiguous()

        att = self.attend(h)
        w = self.w_alpha(att)

        att1_w, att2_w = w.chunk(2, dim=0)
        # Clean attention on padded tokens
        att1_w = att1_w.reshape(batch_size, max_mention_embeddings_size)
        att2_w = att2_w.reshape(batch_size, max_mention_embeddings_size)

        att1_w = self.mask_logits(att1_w, mention_embeddings_size_1)
        att2_w = self.mask_logits(att2_w, mention_embeddings_size_2)
        att1_soft = torch.softmax(att1_w, dim=1)
        att2_soft = torch.softmax(att2_w, dim=1)

        hidden1_reshape = hiddens1_pad.contiguous().reshape(batch_size, max_mention_embeddings_size, -1)
        hidden2_reshape = hiddens2_pad.contiguous().reshape(batch_size, max_mention_embeddings_size, -1)

        rep1 = torch.bmm(att1_soft.unsqueeze(1), hidden1_reshape).squeeze(1)
        rep2 = torch.bmm(att2_soft.unsqueeze(1), hidden2_reshape).squeeze(1)

        g1 = torch.cat((first1_tok, last1_tok, rep1), dim=1)
        g2 = torch.cat((first2_tok, last2_tok, rep2), dim=1)

        span1_span2 = g1 * g2
        concat_result = torch.cat((g1, g2, span1_span2), dim=1)

        ret_golds = self._get_gold_labels(batch_features)

        if self.use_cuda:
            ret_golds = ret_golds.cuda()

        return concat_result, ret_golds

    @staticmethod
    def mask_logits(attention: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        t = torch.arange(attention.size(1), device=attention.device).unsqueeze(0)
        mask = t < lengths.unsqueeze(1)  # Mask everything that is further than the claimed length
        return attention.masked_fill(~mask, float("-inf"))

    @staticmethod
    def _get_gold_labels(batch_features) -> torch.Tensor:
        return torch.tensor(
            [int(m1.coref_chain == m2.coref_chain) for m1, m2 in batch_features],
            dtype=torch.long,
        )

    def set_embed_utils(self, embed_utils):
        self.embeddings_holder = embed_utils
