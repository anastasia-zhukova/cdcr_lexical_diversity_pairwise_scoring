# utils

Helpers shared by the pipeline scripts.

- `embed_utils.py` — `EmbedTransformersGenerics` runs the language model over a mention's context and
  returns the hidden states of the mention tokens (as views into the whole context's output);
  `EmbedFromFile` loads the cached representations of one or more datasets and serves them to the
  pairwise model as padded batches.
- `encoding_cache.py` — `DatasetMentionEncoder` encodes the mentions of a dataset split that take part
  in its pairs and stores them through `EncodingCache`: one pickle per (split, dataset, language model)
  under `experiment_cache_results/`, keyed by mention id. Every cached tensor is a compact CPU copy
  (never a view into the context), each file holds only its own dataset's mentions and is written once
  per dataset; mentions already present in a file are reused and never re-encoded. Use it whenever
  something has to be embedded before training or inference.
- `io_utils.py` — the paths of the experiment cache (`experiment_cache_results/<experiment>/...`) and the
  writers of the CoNLL key/response files.
- `eval_utils.py` — confusion matrix and precision/recall/F1 for the pairwise dev evaluation.
- `string_utils.py` — spaCy-based head/lemma extraction (used by the statistics helper scripts).
- `clustering_utils.py` — legacy agglomerative clustering helper, unused by the current pipeline.
- `log_utils.py` — legacy stdlib file logger used by `train.py`.

Tests live in `tests/` (`pytest cdcr_lexical_diversity_pairwise_scoring/utils/tests`).
