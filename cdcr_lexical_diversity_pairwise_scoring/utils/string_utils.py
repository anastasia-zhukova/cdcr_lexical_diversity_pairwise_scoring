import spacy
from spacy.cli import download
from spacy.symbols import VERB
from spacy.util import is_package


SPACY_MODEL = "en_core_web_sm"

if not is_package(SPACY_MODEL):
    print(f"spaCy model '{SPACY_MODEL}' not found. Downloading...")
    download(SPACY_MODEL)


class SpacySyntaxAnalyzer:
    spacy_parser = spacy.load(SPACY_MODEL)

    @classmethod
    def find_head_lemma_pos_ner(
        cls,
        sentence: str,
    ) -> tuple[str, str, str, str]:
        head = "UNK"
        lemma = "UNK"
        pos = "UNK"
        ner = "UNK"

        doc = cls.spacy_parser(sentence)

        # Find root of the sentence
        for tok in doc:
            if tok.head == tok:
                head = tok.text
                lemma = tok.lemma
                pos = tok.pos_

        # Find what is the NER type of the root
        for ent in doc.ents:
            if ent.root.text == head:
                ner = ent.label_

        return head, lemma, pos, ner

    @classmethod
    def is_verb_phrase(
        cls,
        text: str,
    ) -> bool:
        doc = cls.spacy_parser(text)
        return any(tok.pos_ == VERB for tok in doc)


class SpacyTokenizer:
    spacy_parser = spacy.load(SPACY_MODEL)

    @classmethod
    def get_tokenized_string(
        cls,
        not_tokenized_str: str,
    ) -> list[tuple[str, int]]:
        tokenized_str = []
        doc = cls.spacy_parser(not_tokenized_str)
        for sentence in doc.sents:
            for token in sentence:
                tokenized_str.append((token.text, token.i))

        return tokenized_str
