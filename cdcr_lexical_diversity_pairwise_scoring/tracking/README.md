# tracking

Experiment tracking for the pipeline scripts, backed by MLflow.

## What is here

- `mlflow_tracker.py`
  - `ExperimentTracker` — the contract the scripts code against (`run()`, `log_params`,
    `log_metric(s)`, `log_artifact`, `log_dict`).
  - `MLflowTracker` — the MLflow implementation. Opens/resumes a run, logs the params of a new
    run and tees everything logged during the run (both `loguru` and stdlib `logging`) into a
    `logs/run.log` artifact — also when the run fails.
  - `NoOpTracker` — records nothing, so the pipeline runs unchanged.
  - `TrackerFactory.build(tracking_uri, experiment_name)` — `MLflowTracker` for a non-empty tracking URI,
    `NoOpTracker` otherwise. The package never reads the environment itself: the scripts pass the values
    from `constants.py` in.
- `run_context.py` — `RunContext`: run name, params and tags derived from the Hydra config of
  the running job. The run is named after the experiment config (e.g. `single-random-cd2cr`).

## Configuration

Read from `.env` (see `.env.example`) by `constants.py` and handed to `TrackerFactory.build()` by the scripts:

| variable                  | meaning                                                            |
|---------------------------|--------------------------------------------------------------------|
| `MLFLOW_TRACKING_URI`     | tracking server; empty/unset disables tracking (`NoOpTracker`)     |
| `MLFLOW_TRACKING_USERNAME`/`MLFLOW_TRACKING_PASSWORD` | basic-auth credentials, read by the MLflow client |
| `MLFLOW_EXPERIMENT_NAME`  | the MLflow experiment all runs of this project are grouped under   |

## How a run flows through the pipeline

One MLflow run corresponds to one experiment config and spans three scripts:

1. `train.py` opens the run, logs the config as params, per-epoch `train/loss` and
   `dev/{accuracy,precision,recall,f1}`, the best dev F1, the training time and the best
   checkpoint path, and attaches `model.json` under `training/`. The run id is stored in
   `experiment_cache_results/<experiment>/model.json`.
2. `inference_clustering.py` resumes the run through that id and attaches the inference-time
   CSV and the predicted clusters JSON under `inference/`.
3. `scoring.py` resumes the run of every scored experiment and logs the per-dataset CoNLL
   metrics (`<metric>/<dataset>/<mention_type>`) plus the summary CSVs under `scoring/`.

If `model.json` has no run id (trained before tracking existed), the downstream scripts open a
fresh run instead so nothing is lost.
