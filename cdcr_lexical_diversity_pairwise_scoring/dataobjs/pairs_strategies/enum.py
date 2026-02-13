from enum import StrEnum


class MentionPairStrategy(StrEnum):
    all = "all"
    random = "random"
    tfidf = "tfidf"
    embedding = "embedding"
    encoder = "encoder"
