"""Session-state helpers for the Experiments page."""

from __future__ import annotations

from typing import Any


def ensure_session_defaults(session_state: Any) -> None:
    """Initialize Experiments page session-state keys."""
    if "mlflow_client" not in session_state:
        session_state.mlflow_client = None
    if "mlflow_tracking_uri" not in session_state:
        session_state.mlflow_tracking_uri = None
    if "selected_experiment" not in session_state:
        session_state.selected_experiment = None
    if "selected_run" not in session_state:
        session_state.selected_run = None
