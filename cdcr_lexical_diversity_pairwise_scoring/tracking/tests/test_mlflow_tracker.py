import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from cdcr_lexical_diversity_pairwise_scoring import logger
from cdcr_lexical_diversity_pairwise_scoring.tracking import mlflow_tracker
from cdcr_lexical_diversity_pairwise_scoring.tracking.mlflow_tracker import MLflowTracker, NoOpTracker, TrackerFactory
from cdcr_lexical_diversity_pairwise_scoring.tracking.run_context import RunContext


@pytest.fixture
def fake_mlflow(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Patch the mlflow module so a run executes without a server; record what was sent."""
    recorded: dict[str, Any] = {"artifacts": {}, "params": [], "metrics": [], "start_run_kwargs": []}

    @contextmanager
    def fake_start_run(**kwargs: Any) -> Any:
        recorded["start_run_kwargs"].append(kwargs)
        run = MagicMock()
        run.info.run_id = kwargs.get("run_id") or "run-123"
        yield run

    def fake_log_artifact(local_path: str, artifact_path: str | None = None) -> None:
        # Read eagerly: the tracker writes into a TemporaryDirectory that is gone after the run.
        recorded["artifacts"][Path(local_path).name] = (artifact_path, Path(local_path).read_text(encoding="utf-8"))

    monkeypatch.setattr(mlflow_tracker.mlflow, "set_tracking_uri", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mlflow_tracker.mlflow, "set_experiment", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mlflow_tracker.mlflow, "start_run", fake_start_run)
    monkeypatch.setattr(mlflow_tracker.mlflow, "log_params", lambda params: recorded["params"].append(params))
    monkeypatch.setattr(mlflow_tracker.mlflow, "log_metrics", lambda metrics, step=None: recorded["metrics"].append((metrics, step)))
    monkeypatch.setattr(mlflow_tracker.mlflow, "log_artifact", fake_log_artifact)
    return recorded


@pytest.fixture
def tracker(fake_mlflow: dict[str, Any]) -> MLflowTracker:
    return MLflowTracker(tracking_uri="http://example.test", experiment_name="exp")


@pytest.fixture
def context() -> RunContext:
    return RunContext(run_name="single-random-cd2cr", params={"ratio": "10"}, tags={"setting": "single"})


def test_new_run_logs_params_and_exposes_run_id(tracker: MLflowTracker, context: RunContext, fake_mlflow: dict[str, Any]) -> None:
    with tracker.run(context, run_id=None) as run_id:
        assert run_id == "run-123"
        assert tracker.run_id == "run-123"

    assert tracker.run_id is None
    assert fake_mlflow["params"] == [{"ratio": "10"}]
    assert fake_mlflow["start_run_kwargs"][0]["run_name"] == "single-random-cd2cr"
    assert fake_mlflow["start_run_kwargs"][0]["tags"] == {"setting": "single"}


def test_resumed_run_does_not_relog_params(tracker: MLflowTracker, context: RunContext, fake_mlflow: dict[str, Any]) -> None:
    with tracker.run(context, run_id="existing-run") as run_id:
        assert run_id == "existing-run"

    assert fake_mlflow["params"] == []
    assert fake_mlflow["start_run_kwargs"] == [{"run_id": "existing-run"}]


def test_run_uploads_log_artifact_with_both_logging_systems(tracker: MLflowTracker, context: RunContext, fake_mlflow: dict[str, Any]) -> None:
    with tracker.run(context, run_id=None):
        logger.info("a loguru line emitted during the run")
        logging.getLogger("test.capture").error("a stdlib line emitted during the run")

    artifact_path, contents = fake_mlflow["artifacts"]["run.log"]
    assert artifact_path == "logs"
    assert "a loguru line emitted during the run" in contents
    assert "a stdlib line emitted during the run" in contents
    assert "\x1b[" not in contents


def test_run_uploads_logs_even_when_body_raises(tracker: MLflowTracker, context: RunContext, fake_mlflow: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="boom"):  # noqa: PT012, SIM117
        with tracker.run(context, run_id=None):
            logger.info("logged before failure")
            raise ValueError("boom")

    assert "logged before failure" in fake_mlflow["artifacts"]["run.log"][1]
    assert tracker.run_id is None


def test_metrics_are_forwarded_with_step(tracker: MLflowTracker, context: RunContext, fake_mlflow: dict[str, Any]) -> None:
    with tracker.run(context, run_id=None):
        tracker.log_metrics({"dev/f1": 0.5}, step=3)

    assert fake_mlflow["metrics"] == [({"dev/f1": 0.5}, 3)]


def test_noop_tracker_yields_no_run_id(context: RunContext) -> None:
    tracker = NoOpTracker()

    with tracker.run(context, run_id=None) as run_id:
        tracker.log_metrics({"dev/f1": 0.5}, step=None)
        assert run_id is None

    assert tracker.run_id is None


def test_factory_builds_noop_without_tracking_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mlflow_tracker, "MLFLOW_TRACKING_URI", "")

    assert isinstance(TrackerFactory.build(), NoOpTracker)


def test_factory_builds_mlflow_tracker_with_tracking_uri(monkeypatch: pytest.MonkeyPatch, fake_mlflow: dict[str, Any]) -> None:
    monkeypatch.setattr(mlflow_tracker, "MLFLOW_TRACKING_URI", "http://example.test")

    assert isinstance(TrackerFactory.build(), MLflowTracker)
