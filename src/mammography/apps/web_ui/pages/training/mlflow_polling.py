"""MLflow polling helpers for the Training page."""

from __future__ import annotations

import time
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

METRIC_MAPPINGS = {
    "train/loss": "train_loss",
    "train_loss": "train_loss",
    "val/loss": "val_loss",
    "val_loss": "val_loss",
    "train/acc": "train_acc",
    "train_acc": "train_acc",
    "val/acc": "val_acc",
    "val_acc": "val_acc",
    "lr": "learning_rate",
    "learning_rate": "learning_rate",
    "epoch": "epoch",
}


def get_mlflow_client(tracking_uri: str, session_state: Any):
    """Get or create an MLflow client for the given tracking URI."""
    if mlflow is None or MlflowClient is None:
        return None

    if (
        session_state.mlflow_client is not None
        and session_state.mlflow_tracking_uri == tracking_uri
    ):
        return session_state.mlflow_client

    try:
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)
        session_state.mlflow_client = client
        session_state.mlflow_tracking_uri = tracking_uri
        return client
    except Exception:
        return None


def poll_mlflow_metrics(session_state: Any, *, min_interval: float = 2.0) -> bool:
    """Poll MLflow for metrics from the active training run."""
    if mlflow is None or session_state.mlflow_client is None:
        return False
    if session_state.active_run_id is None:
        return False

    current_time = time.time()
    if current_time - session_state.last_metrics_poll < min_interval:
        return False

    try:
        client = session_state.mlflow_client
        run_id = session_state.active_run_id
        run = client.get_run(run_id)
        if run is None:
            return False

        metrics = run.data.metrics
        updated_metrics = {}
        for mlflow_key, internal_key in METRIC_MAPPINGS.items():
            if mlflow_key in metrics:
                updated_metrics[internal_key] = metrics[mlflow_key]

        if "epoch" in metrics:
            try:
                epoch_history = client.get_metric_history(run_id, "epoch")
                if epoch_history:
                    latest_epoch_entry = max(epoch_history, key=lambda h: h.step)
                    current_epoch = int(latest_epoch_entry.value)
                    updated_metrics["epoch"] = current_epoch

                    # Fetch each metric's full history once, then index by step
                    per_metric: dict[str, dict[int, float]] = {}
                    for metric_name in ["train/loss", "val/loss", "train/acc", "val/acc", "lr"]:
                        try:
                            history = client.get_metric_history(run_id, metric_name)
                            internal_name = METRIC_MAPPINGS.get(metric_name, metric_name)
                            per_metric[internal_name] = {h.step: h.value for h in history}
                        except Exception:
                            pass

                    history_entries = [
                        {"epoch": step, **{k: v[step] for k, v in per_metric.items() if step in v}}
                        for step in range(current_epoch + 1)
                    ]

                    if history_entries:
                        session_state.training_metrics_history = history_entries
            except Exception:
                pass

        if updated_metrics:
            session_state.training_metrics.update(updated_metrics)
            session_state.last_metrics_poll = current_time
            return True
    except Exception:
        pass

    session_state.last_metrics_poll = current_time
    return False