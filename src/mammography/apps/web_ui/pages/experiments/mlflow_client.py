"""MLflow client helpers for the Experiments page."""

from __future__ import annotations

from typing import Any

try:
    import mlflow
    from mlflow.tracking import MlflowClient
except Exception as exc:  # pragma: no cover - optional MLflow dependency
    mlflow = None
    MlflowClient = None
    _MLFLOW_IMPORT_ERROR = exc
else:
    _MLFLOW_IMPORT_ERROR = None


def get_mlflow_client(tracking_uri: str, session_state: Any):
    """Get or create an MLflow client for the selected tracking URI."""
    if mlflow is None or MlflowClient is None:
        raise ImportError(
            "MLflow is required for experiment tracking. Install with: pip install mlflow"
        ) from _MLFLOW_IMPORT_ERROR

    if (
        session_state.mlflow_client is not None
        and session_state.mlflow_tracking_uri == tracking_uri
    ):
        return session_state.mlflow_client

    client = MlflowClient(tracking_uri=tracking_uri)
    session_state.mlflow_client = client
    session_state.mlflow_tracking_uri = tracking_uri
    return client


def list_experiments(client: Any) -> list[Any]:
    """List non-deleted MLflow experiments."""
    experiments = client.search_experiments()
    return [exp for exp in experiments if exp.lifecycle_stage != "deleted"]


def list_runs(client: Any, experiment_id: str, max_results: int = 100) -> list[Any]:
    """List runs for a specific experiment."""
    return client.search_runs(
        experiment_ids=[experiment_id],
        max_results=max_results,
        order_by=["start_time DESC"],
    )
