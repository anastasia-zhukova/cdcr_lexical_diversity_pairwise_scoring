import pytest

from cdcr_lexical_diversity_pairwise_scoring.dataobjs.mention_data import MentionuCDCR
from cdcr_lexical_diversity_pairwise_scoring.utils.embed_utils import EmbedTransformersGenerics

MODEL_MAX_LENGTH = 16


class _WordTokenizer:

    """Stands in for the RoBERTa tokenizer: one token per whitespace-separated word, no special tokens added."""

    cls_token_id = -1
    sep_token_id = -2
    model_max_length = MODEL_MAX_LENGTH

    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [hash(word) % 1000 for word in text.split()]


def _mention(context: list[str], span: list[int], tokens_text: list[str]) -> MentionuCDCR:
    return MentionuCDCR(
        {
            "mention_id": "m1",
            "dataset": "TEST",
            "mention_context": context,
            "tokens_number_context": span,
            "tokens_number": span,
            "tokens_text": tokens_text,
        },
    )


def _embedder(max_surrounding_context: int = -1) -> EmbedTransformersGenerics:
    # bypass __init__ so that no transformer model is downloaded in the tests
    embedder = EmbedTransformersGenerics.__new__(EmbedTransformersGenerics)
    embedder.tokenizer = _WordTokenizer()
    embedder.max_surrounding_context = max_surrounding_context
    return embedder


def test_extract_matching_mention_tokens() -> None:
    mention = _mention(["a", "b", "c", "d"], [1, 2], ["b", "c"])

    before, span, after = EmbedTransformersGenerics.extract_mention_surrounding_context(mention)

    assert (before, span, after) == (["a"], ["b", "c"], ["d"])


def test_extract_accepts_span_tokenised_differently_from_mention() -> None:
    mention = _mention(["the", "EADS", "-", "led", "group"], [1, 2, 3], ["EADS-led"])

    before, span, after = EmbedTransformersGenerics.extract_mention_surrounding_context(mention)

    assert (before, span, after) == (["the"], ["EADS", "-", "led"], ["group"])


@pytest.mark.parametrize(
    ("context", "span", "tokens_text"),
    [
        (["Boeing", "787", '"', "Dreamliner", "jet"], [0, 1, 2, 3], ["Boeing", "787", "``", "Dreamliner"]),
        (["the", "AT&T.", "deal"], [1], ["AT&T"]),
        (["Ford", "Motor", "Co", "..", "said"], [0, 1, 2, 3], ["Ford", "Motor", "Co."]),
    ],
)
def test_extract_accepts_span_that_starts_with_the_mention_text(
    context: list[str], span: list[int], tokens_text: list[str],
) -> None:
    mention = _mention(context, span, tokens_text)

    _, ret_mention, _ = EmbedTransformersGenerics.extract_mention_surrounding_context(mention)

    assert ret_mention == context[span[0] : span[-1] + 1]


def test_extract_rejects_span_with_different_text() -> None:
    mention = _mention(["the", "EADS", "-", "led", "group"], [1, 2, 3], ["Boeing"])

    with pytest.raises(ValueError, match="does not match"):
        EmbedTransformersGenerics.extract_mention_surrounding_context(mention)


def test_encode_leaves_short_context_untouched() -> None:
    context = [f"w{i}" for i in range(10)]
    mention = _mention(context, [4, 5], ["w4", "w5"])

    ids, start, end = _embedder().encode_mention(mention)

    assert ids.shape == (1, 12)
    assert ids[0, 0].item() == _WordTokenizer.cls_token_id
    assert ids[0, -1].item() == _WordTokenizer.sep_token_id
    assert (start, end) == (5, 7)


def test_encode_clips_long_context_around_the_mention() -> None:
    context = [f"w{i}" for i in range(40)]
    mention = _mention(context, [20, 21], ["w20", "w21"])

    ids, start, end = _embedder().encode_mention(mention)

    assert ids.shape == (1, MODEL_MAX_LENGTH)
    assert ids[0, start:end].tolist() == _WordTokenizer().encode("w20 w21", add_special_tokens=False)
    assert ids[0, 1:start].tolist() == _WordTokenizer().encode(" ".join(context[14:20]), add_special_tokens=False)
    assert ids[0, end:-1].tolist() == _WordTokenizer().encode(" ".join(context[22:28]), add_special_tokens=False)


def test_encode_gives_unused_budget_to_the_other_side() -> None:
    context = [f"w{i}" for i in range(40)]
    mention = _mention(context, [1, 2], ["w1", "w2"])

    ids, start, end = _embedder().encode_mention(mention)

    assert ids.shape == (1, MODEL_MAX_LENGTH)
    assert (start, end) == (2, 4)
    assert ids[0, end:-1].tolist() == _WordTokenizer().encode(" ".join(context[3:14]), add_special_tokens=False)


def test_encode_rejects_mention_longer_than_the_model() -> None:
    context = [f"w{i}" for i in range(40)]
    mention = _mention(context, list(range(5, 25)), context[5:25])

    with pytest.raises(ValueError, match="more than the model can take"):
        _embedder().encode_mention(mention)
