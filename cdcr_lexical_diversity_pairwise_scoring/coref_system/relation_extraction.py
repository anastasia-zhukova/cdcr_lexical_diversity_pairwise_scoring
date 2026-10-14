from enum import Enum

import torch


class RelationTypeEnum(Enum):
    no_relation_found = 0
    exact_string = 1
    same_head_lemma = 2
    pairwise = 3


class RelationExtraction:
    def __init__(self):
        self.cache = dict()

    def predict(self, batch_features, bs):
        prediction = list()
        for pair in batch_features:
            prediction.append(self._solve(pair[0], pair[1]))

        return torch.tensor(prediction), None

    def _solve(self, mention_x, mention_y):
        raise NotImplementedError


class ExactStringRelationExtractor(RelationExtraction):
    def __init__(self):
        super(ExactStringRelationExtractor, self).__init__()

    def _solve(self, mention_x, mention_y):
        mention1_str = mention_x.tokens_str
        mention2_str = mention_y.tokens_str
        return 1 if mention1_str.lower() == mention2_str.lower() else 0


class HeadLemmaRelationExtractor(RelationExtraction):
    def __init__(self):
        super(HeadLemmaRelationExtractor, self).__init__()

    def _solve(self, mention_x, mention_y):
        return 1 if mention_x.mention_head_lemma.lower() == mention_y.mention_head_lemma.lower() else 0
