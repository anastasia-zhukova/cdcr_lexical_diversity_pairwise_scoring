import logging
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow

from cdcr_lexical_diversity_pairwise_scoring import logger
from cdcr_lexical_diversity_pairwise_scoring.constants import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_LOG_ARTIFACT_DIR,
    MLFLOW_RUN_LOG_FILENAME,
    MLFLOW_TRACKING_URI,
)
from cdcr_lexical_diversity_pairwise_scoring.tracking.run_context import RunContext

# Both logging systems of the project (loguru in the scripts, stdlib logging in `dataobjs`)
# are captured into one file, so the format has to be applied at the stdlib handler level.
_CAPTURED_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
_LOGURU_TO_STDLIB_FORMAT = "{name}:{function}:{line} - {message}"


class ExperimentTracker(ABC):
    """Contract for run tracking used by the pipeline scripts.

    A run is opened with `run()`; everything logged inside the block is attached to it.
    Passing `run_id` resumes a run created by an earlier pipeline step (e.g. the inference
    step attaching its results to the training run).
    """

    @property
    @abstractmethod
    def run_id(self) -> str | None: ...

    @abstractmethod
    @contextmanager
    def run(
        self,
        context: RunContext,
        run_id: str | None,
    ) -> Iterator[str | None]: ...

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None: ...

    @abstractmethod
    def log_metric(
        self,
        key: str,
        value: float,
        step: int | None,
    ) -> None: ...

    @abstractmethod
    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None,
    ) -> None: ...

    @abstractmethod
    def log_artifact(
        self,
        local_path: Path,
        artifact_path: str | None,
    ) -> None: ...

    @abstractmethod
    def log_dict(
        self,
        payload: dict[str, Any],
        artifact_file: str,
    ) -> None: ...


class NoOpTracker(ExperimentTracker):
    """Tracker used when no tracking server is configured: the pipeline runs untouched."""

    @property
    def run_id(self) -> str | None:
        return None

    @contextmanager
    def run(
        self,
        context: RunContext,
        run_id: str | None,
    ) -> Iterator[str | None]:
        logger.warning(
            f"MLflow tracking is disabled (no MLFLOW_TRACKING_URI). Run '{context.run_name}' is not tracked.",
        )
        yield None

    def log_params(self, params: dict[str, Any]) -> None:
        return

    def log_metric(
        self,
        key: str,
        value: float,
        step: int | None,
    ) -> None:
        return

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None,
    ) -> None:
        return

    def log_artifact(
        self,
        local_path: Path,
        artifact_path: str | None,
    ) -> None:
        return

    def log_dict(
        self,
        payload: dict[str, Any],
        artifact_file: str,
    ) -> None:
        return


class MLflowTracker(ExperimentTracker):
    """Thin OOP wrapper over the MLflow client.

    Owns the tracking URI and the experiment name. Basic-auth credentials are read by the
    MLflow client directly from MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD.
    """

    def __init__(
        self,
        tracking_uri: str,
        experiment_name: str,
    ) -> None:
        self._experiment_name = experiment_name
        self._run_id: str | None = None
        mlflow.set_tracking_uri(tracking_uri)
        logger.debug(f"MLflow tracker configured for {tracking_uri} (experiment '{experiment_name}').")

    @property
    def run_id(self) -> str | None:
        return self._run_id

    @contextmanager
    def run(
        self,
        context: RunContext,
        run_id: str | None,
    ) -> Iterator[str | None]:
        mlflow.set_experiment(self._experiment_name)
        resumed = run_id is not None

        # A resumed run keeps its own name, tags and params: the context describes the current
        # job, which may be scoring the results of a different experiment config.
        run_arguments = {"run_id": run_id} if resumed else {"run_name": context.run_name, "tags": context.tags}

        with mlflow.start_run(**run_arguments) as active_run:
            self._run_id = active_run.info.run_id

            with self._capture_logs():
                action = "Resumed" if resumed else "Started"
                logger.info(f"{action} MLflow run '{context.run_name}' (run_id={self._run_id}).")

                if not resumed:
                    mlflow.log_params(context.params)

                try:
                    yield self._run_id
                except Exception:
                    logger.exception(f"MLflow run {self._run_id} failed.")
                    raise
                finally:
                    self._run_id = None

    def log_params(self, params: dict[str, Any]) -> None:
        mlflow.log_params(params)

    def log_metric(
        self,
        key: str,
        value: float,
        step: int | None,
    ) -> None:
        mlflow.log_metric(key, value, step=step)

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None,
    ) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(
        self,
        local_path: Path,
        artifact_path: str | None,
    ) -> None:
        mlflow.log_artifact(str(local_path), artifact_path=artifact_path)

    def log_dict(
        self,
        payload: dict[str, Any],
        artifact_file: str,
    ) -> None:
        mlflow.log_dict(payload, artifact_file)

    @contextmanager
    def _capture_logs(self) -> Iterator[None]:
        """Tee everything logged during the run into a file and attach it to the active run.

        The artifact is uploaded in `finally` so a failed or aborted run still keeps its logs.
        """
        with tempfile.TemporaryDirectory() as temporary_directory:
            log_path = Path(temporary_directory) / MLFLOW_RUN_LOG_FILENAME
            handler = logging.FileHandler(log_path, encoding="utf-8")
            handler.setFormatter(logging.Formatter(_CAPTURED_LOG_FORMAT))

            root_logger = logging.getLogger()
            root_logger.addHandler(handler)
            loguru_sink_id = logger.add(handler, format=_LOGURU_TO_STDLIB_FORMAT, level="DEBUG")

            try:
                yield
            finally:
                logger.remove(loguru_sink_id)
                root_logger.removeHandler(handler)
                handler.close()
                mlflow.log_artifact(str(log_path), artifact_path=MLFLOW_LOG_ARTIFACT_DIR)


class TrackerFactory:
    """Builds the tracker from the environment: MLflow when a tracking URI is set, no-op otherwise."""

    @staticmethod
    def build() -> ExperimentTracker:
        if MLFLOW_TRACKING_URI == "":
            return NoOpTracker()

        return MLflowTracker(tracking_uri=MLFLOW_TRACKING_URI, experiment_name=MLFLOW_EXPERIMENT_NAME)
