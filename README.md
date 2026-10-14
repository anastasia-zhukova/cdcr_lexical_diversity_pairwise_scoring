# Transfer Learning in Pairwise Cross-Document Coreference Resolution Model
This project studies how a pairwise cross-document coreference resolution (CDCR) model trained on one dataset, or on
a mix of two, transfers to CDCR datasets with different problems, i.e., lexical diversity and lexical ambiguity. 
The model is the pairwise scorer
of [Eirew et al. 2021](https://github.com/AlonEirew/cross-doc-event-coref) (frozen RoBERTa-large mention
representations and an MLP over mention pairs, followed by agglomerative clustering). The experiments vary the training
data (single dataset vs. a mix of two) and the way training pairs are sampled (random vs. contrastive pairs chosen by
tf-idf or sentence-encoder similarity), and every model is evaluated on all 12 datasets of uCDCR.

This is a baseline to the CL-coref and MS-coref models described in "Navigating Real-World Challenges of 
Cross-Document Coreference Resolution using Contrastive and Transfer Learning". 

## uCDCR dataset
The uCDCR dataset used in this project is available on the Hugging Face Hub at
[AnZhu/uCDCR](https://huggingface.co/datasets/AnZhu/uCDCR). The code that builds it from the original
corpora is at [anastasia-zhukova/uCDCR](https://github.com/anastasia-zhukova/uCDCR); see the
[paper](https://arxiv.org/abs/2603.00621) (LREC 2026) for details.
We use all 12 English datasets of uCDCR: the 10 that have a train split are used for training, and every model is
evaluated on all 12.

| dataset (folder name) | coreference target | used for training |
|-----------------------|--------------------|-------------------|
| CD2CR                 | entity             | yes               |
| CEREC                 | entity             | yes               |
| ECBplus               | event + entity     | yes               |
| ECBplusMETAm          | event + entity     | no (no train split) |
| FCC                   | event              | yes               |
| GVC                   | event              | yes               |
| HyperCoref            | event              | yes               |
| MEANTIME              | event + entity     | yes               |
| NewsWCL50             | event + entity     | no (no train split) |
| NiDENT                | entity             | yes               |
| NP4E                  | entity             | yes               |
| WECEng                | event              | yes               |

To download the dataset into `datasets/uCDCR` (the `dataset_folder` used by all configs):
```
uv run hf download AnZhu/uCDCR --repo-type dataset --local-dir datasets/uCDCR
```
The pipeline reads `datasets/uCDCR/<dataset>/{train,val,test}/{event,entity}_mentions.json`; the folder names are
case-sensitive and must match the names in the configs.

## Setup
- Python 3.12
- [uv](https://docs.astral.sh/uv/); `uv sync` creates `.venv` from the committed `uv.lock` and installs the project
  in editable mode (PyTorch comes from the CUDA 13.0 index on Linux and Windows, CPU-only on macOS)
- A CUDA GPU for the configs as shipped (`use_cuda: true`)

```
uv sync
```
The first run also downloads the NLTK stopwords, the `roberta-large` model and, for the contrastive configs, the
sentence encoder `intfloat/multilingual-e5-large-instruct`.

### Experiment tracking (MLflow)
Runs are tracked in MLflow when a tracking server is configured. Copy `.env.example` to `.env` and set
`MLFLOW_TRACKING_URI` and, optionally, `MLFLOW_EXPERIMENT_NAME` (default `cdcr_lexical_diversity_pairwise_scoring`).
For a server behind Keycloak also set `KEYCLOAK_TOKEN_URL`, `KEYCLOAK_CLIENT_ID` and `KEYCLOAK_CLIENT_SECRET`; for a
server with basic auth set `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD` instead (not both).
Leaving `MLFLOW_TRACKING_URI` empty disables tracking; the pipeline behaves exactly as before.

One MLflow run corresponds to one experiment config and is named after it (e.g. `single-random-cd2cr`):
`train.py` opens it (params, per-epoch loss and dev metrics, training time), `inference_clustering.py` attaches
the inference outputs and `scoring.py` the MUC, B3, CEAFe, LEA and CoNLL scores per dataset and mention type.
Training a config again opens a new run with the same name. See
`cdcr_lexical_diversity_pairwise_scoring/tracking/README.md` for the details.

## Folder with experiment configs
The folder with all experiment configs for reproduction:
```
config
```
Each YAML file is one experiment and drives all five pipeline steps; the file name (without `.yaml`) is the
experiment name used for the cache and output folders and the MLflow run. Every model is tested on all 12 datasets.

| file name | experiment code | group of experiment | description |
|-----------|-----------------|---------------------|-------------|
| `single-random-<dataset>.yaml` (10 files) | `single-random` | single-dataset training | Train and dev on one dataset, random pairs (10 negatives per positive). |
| `single-contrast_tfidf-<dataset>.yaml` (10 files) | `single-contrast_tfidf` | single-dataset training | As above, but with contrastive pairs: easy/hard positives and negatives chosen by tf-idf similarity of the mentions within a topic. |
| `single-contrast_encoder-<dataset>.yaml` (10 files) | `single-contrast_encoder` | single-dataset training | As above, with similarity from the sentence encoder `intfloat/multilingual-e5-large-instruct`. |
| `mix-contrast_encoder-<base>_<dataset>.yaml` (17 files) | `mix-contrast_encoder` | mixed training | Train on a base dataset (`ecbplus` or `nident`) plus one other dataset with encoder contrastive pairs; dev on all 12 datasets (capped at 200 000 pairs). |
| `test_*.yaml` | – | smoke tests | Small runs (`roberta-base`, 8 000 train pairs, 10 epochs) for checking the pipeline; not included in the sweeps. |

`<dataset>` is one of `cd2cr`, `cerec`, `ecbplus`, `fcc`, `gvc`, `hypercoref`, `meantime`, `nident`, `np4e`, `wec`.
`config/_archive/` holds older configs that are no longer used.

The config fields:

| field | description |
|-------|-------------|
| `dataset_folder` | Path to uCDCR, relative to the project root. |
| `setting` | `single`: train on the first dataset of `train_dataset_names`; `mix`: train on all of them; `excluding_target`: train on the listed datasets minus the test datasets. |
| `type_of_pairs` | How training pairs are sampled: `random`, or contrastive `tfidf`, `embedding` (fastText) or `encoder` (sentence encoder). |
| `ratio` | Negative pairs per positive pair for training. |
| `max_pairs_train` / `max_pairs_dev` | Budget of training / dev pairs, split evenly across the datasets; `null` means no cap for random and all-pairs sampling (contrastive sampling then uses 40 000 000 train / 8 000 dev pairs). |
| `train_scope` / `dev_scope` / `test_scope` | Unit within which mention pairs are built and, for test, clustered: `subtopic`, `topic`, `dataset` (or `corpus`). |
| `train_dataset_names` / `dev_dataset_names` / `test_dataset_names` | uCDCR dataset folder names per split. |
| `dev_type_of_pairs` | Dev pair sampling; `all` takes all positive pairs and up to 20 negatives per positive (within `max_pairs_dev` if set), other values sample like `type_of_pairs`. |
| `language_model` | RoBERTa-family model used to encode the mentions. |
| `batch_size`, `learning_rate`, `weight_decay`, `training_iterations` | Training batch size, Adam learning rate and weight decay, number of epochs. |
| `hidden_size` | Hidden size of the pairwise MLP. |
| `use_cuda` | Run encoding, training and inference on the GPU. |
| `fine_tune` | Currently unused; the language model is always frozen. |

The same-head-lemma baseline is computed in the uCDCR project, see `conll_lemma_baseline` in
[create_summary.py](https://github.com/anastasia-zhukova/uCDCR/blob/main/create_summary.py).

## Running sweeps over all experiment configs
`make <step>` runs one pipeline step for every config in `config/` except the `test_*` smoke configs (`make gen-pairs`, `make embed`, `make train`,
`make inference`, `make scoring`); `make all` runs the five steps in order, each swept over all configs;
`make pipeline CONFIG=<name>` runs all steps for one config. Restrict a sweep with `CONFIGS="a b"` or to one config family with `GROUP=single-random` (also `single-contrast_tfidf`, `single-contrast_encoder`, `mix-contrast_encoder`). A failing
config does not stop the sweep — failures are listed at the end and each run's output is kept in `sweep_logs/`;
`make all` and `make pipeline` do not start the next step if any config failed.
Runs are memory-capped through `systemd-run` (`MEMORY_MAX=12G` by default, `MEMORY_MAX=` to disable). The Makefile
sets `PYTHONHASHSEED=0` so that pair sampling is reproducible. `make help` prints the full usage.

## Running the steps by hand
Every script takes the experiment config by name (without `.yaml`) through Hydra's `--config-name`; without it, the
`test_random` config is used. Single fields can be overridden on the command line (e.g. `batch_size=128`), but the
outputs still go to the folder of the config name. Set `PYTHONHASHSEED=0` as the Makefile does to get the same pairs.
Hydra writes its own log to `outputs/<date>/<time>/` in the current directory.

## 1. Preprocessing
### 1.1 Generate mentions pairs
The script creates the train, dev and test mention pairs for the given experiment config and saves them to
`experiment_cache_results/<config>/`; a split whose pairs are already there is skipped.
For example:
```
PYTHONHASHSEED=0 uv run python cdcr_lexical_diversity_pairwise_scoring/preprocess_gen_pairs.py --config-name single-random-cd2cr
```

### 1.2. Generate Embeddings
To generate the embeddings for the datasets in a given experiment, run the following script (with an examplar experiment config):
```
uv run python cdcr_lexical_diversity_pairwise_scoring/preprocess_embed.py --config-name single-random-cd2cr
```
Note that to save space, the encoded mentions are cached per split, dataset and language model in
`experiment_cache_results/cached_<split>_<dataset>_<language_model>.pickle` and shared by all experiments; only
mentions that are not in the cache yet are encoded. Delete these files if the encoding code changes.

## 2. Training
The training parameters are the fields of the experiment config (see above).
A checkpoint is saved to `checkpoints/<timestamp>/` after every epoch that improves the pairwise F1 on the dev set;
the path of the best one is recorded in `experiment_cache_results/<config>/model.json`.
```
uv run python cdcr_lexical_diversity_pairwise_scoring/train.py --config-name single-random-cd2cr
```


## 3. Inference
### 3.1. Mention pairs scoring and clustering
The script scores all test mention pairs with the best checkpoint and runs agglomerative clustering (average linkage,
distance threshold 0.5) on the pairwise predictions, separately per dataset, `test_scope` unit and mention type
(events, entities, both). Gold singletons are left out of the test pairs. The key and response CoNLL files are written
to `experiment_cache_results/<config>/predictions_conll/`.
```
uv run python cdcr_lexical_diversity_pairwise_scoring/inference_clustering.py --config-name single-random-cd2cr
```

### 3.2. Calculating evaluation metrics
The script computes MUC, B3, CEAFe, LEA and the CoNLL F1 (mean of MUC, B3 and CEAFe) per topic with a Python
implementation of the CoNLL metrics (`helper_scripts/metrics.py`) and averages them per dataset and mention type.
By default (`SCORE_ALL_EXPERIMENTS = True` in `constants.py`) it scores every experiment in
`experiment_cache_results/`, whichever config is given, and writes the summary table to
`evaluation_results/<timestamp>_conll_evaluation.csv` (per-topic scores in `..._conll_evaluation_topics.csv`).
Set the constant to `False` to score only the given experiment.

```
uv run python cdcr_lexical_diversity_pairwise_scoring/scoring.py --config-name single-random-cd2cr
```

## Acknowledgements
The pairwise model and parts of the pipeline are adapted from
[cross-doc-event-coref](https://github.com/AlonEirew/cross-doc-event-coref) by Eirew et al. 2021 (WEC-Eng).
