"""Session-state helpers for the Training page."""

from __future__ import annotations

from typing import Any

DEFAULT_TRAINING_METRICS: dict[str, float | int] = {
    "epoch": 0,
    "train_loss": 0.0,
    "val_loss": 0.0,
    "train_acc": 0.0,
    "val_acc": 0.0,
    "learning_rate": 0.0,
    "samples_processed": 0,
}


def ensure_session_defaults(session_state: Any) -> None:
    """Initialize Training page session-state keys."""
    if "training_config" not in session_state:
        session_state.training_config = {}
    if "training_process" not in session_state:
        session_state.training_process = None
    if "training_output" not in session_state:
        session_state.training_output = []
    if "training_status" not in session_state:
        session_state.training_status = "idle"
    if "training_metrics" not in session_state:
        session_state.training_metrics = dict(DEFAULT_TRAINING_METRICS)
    if "training_metrics_history" not in session_state:
        session_state.training_metrics_history = []
    if "mlflow_client" not in session_state:
        session_state.mlflow_client = None
    if "mlflow_tracking_uri" not in session_state:
        session_state.mlflow_tracking_uri = None
    if "active_run_id" not in session_state:
        session_state.active_run_id = None
    if "last_metrics_poll" not in session_state:
        session_state.last_metrics_poll = 0


def sync_shared_state(session_state: Any) -> None:
    """Copy background-thread status values into Streamlit session state."""
    shared = getattr(session_state, "_training_shared", None)
    if shared is not None:
        session_state.training_status = shared["training_status"]
        if "training_output" in shared:
            session_state.training_output = list(shared["training_output"])
        if shared["active_run_id"] is not None:
            session_state.active_run_id = shared["active_run_id"]
        if shared.get("finished"):
            session_state.training_process = None
