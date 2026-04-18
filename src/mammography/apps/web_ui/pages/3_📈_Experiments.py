#
# 3_📈_Experiments.py
# mammography-pipelines
#
# Streamlit page for viewing MLflow experiments and tracking training runs.
#
# Thales Matheus Mendonca Santos - February 2026
#
"""Experiments page for viewing and analyzing MLflow training runs."""

from __future__ import annotations

import os

from mammography.apps.web_ui.pages.experiments import mlflow_client as experiment_mlflow_client
from mammography.apps.web_ui.pages.experiments import overview as experiment_overview
from mammography.apps.web_ui.pages.experiments import state as experiment_state
from mammography.apps.web_ui.utils import ensure_shared_session_state

try:
    import streamlit as st
except Exception as exc:  # pragma: no cover - optional UI dependency
    st = None
    _STREAMLIT_IMPORT_ERROR = exc
else:
    _STREAMLIT_IMPORT_ERROR = None

try:
    import mlflow
    from mlflow.tracking import MlflowClient
except Exception as exc:  # pragma: no cover - optional MLflow dependency
    mlflow = None
    MlflowClient = None
    _MLFLOW_IMPORT_ERROR = exc
else:
    _MLFLOW_IMPORT_ERROR = None


def _require_streamlit() -> None:
    """Raise ImportError if Streamlit is not available."""
    if st is None:
        raise ImportError(
            "Streamlit is required to run the web UI dashboard."
        ) from _STREAMLIT_IMPORT_ERROR


def _require_mlflow() -> None:
    """Raise ImportError if MLflow is not available."""
    if mlflow is None:
        raise ImportError(
            "MLflow is required for experiment tracking. Install with: pip install mlflow"
        ) from _MLFLOW_IMPORT_ERROR


def _ensure_session_defaults() -> None:
    """Initialize session state with default values."""
    experiment_state.ensure_session_defaults(st.session_state)


def _get_mlflow_client(tracking_uri: str) -> MlflowClient:
    """Get or create MLflow client with the specified tracking URI."""
    return experiment_mlflow_client.get_mlflow_client(tracking_uri, st.session_state)


def _list_experiments(client: MlflowClient) -> list:
    """List all MLflow experiments."""
    try:
        return experiment_mlflow_client.list_experiments(client)
    except Exception as exc:
        st.error(f"Failed to list experiments: {exc}")
        return []

def main() -> None:
    """Render the experiments page."""
    _require_streamlit()

    st.set_page_config(
        page_title="Experiments - Mammography Pipelines",
        page_icon="📈",
        layout="wide",
    )

    # Initialize shared session state for cross-page data persistence
    try:
        ensure_shared_session_state()
        _ensure_session_defaults()
    except Exception as exc:
        st.error(f"❌ Failed to initialize session state: {exc}")
        st.stop()

    st.title("📈 Experiment Tracking")
    # Check if MLflow is available
    try:
        _require_mlflow()
    except ImportError as exc:
        st.error(f"❌ MLflow is not installed: {exc}")
        st.markdown("""
        ### Installation Instructions

        To use experiment tracking, install MLflow:

        ```bash
        pip install mlflow
        ```

        Then restart the Streamlit application.
        """)
        return

    st.header("MLflow Experiment Tracking")

    st.markdown("""
    View and analyze training experiments logged with MLflow. This page displays:

    - **Experiments**: Collections of related training runs
    - **Runs**: Individual training executions with parameters, metrics, and artifacts
    - **Metrics**: Performance indicators tracked during training (accuracy, loss, etc.)
    - **Parameters**: Configuration values used for each run (learning rate, batch size, etc.)
    - **Artifacts**: Files produced during training (checkpoints, plots, etc.)
    """)

    st.info(
        "💡 **Quick Start:** Enter your MLflow tracking URI (or use the default `./mlruns`), "
        "click 'Connect', then browse your experiments and training runs."
    )

    # MLflow tracking URI configuration
    st.subheader("MLflow Configuration")

    col1, col2 = st.columns([3, 1])

    with col1:
        # Default tracking URI
        default_uri = os.environ.get("MLFLOW_TRACKING_URI", "./mlruns")

        tracking_uri = st.text_input(
            "Tracking URI",
            value=default_uri,
            help="MLflow tracking server URI or local path (e.g., ./mlruns, http://localhost:5000)",
        )

    with col2:
        st.write("")  # Spacer
        st.write("")  # Spacer
        if st.button("🔌 Connect", type="primary"):
            with st.spinner(f"Connecting to MLflow at {tracking_uri}..."):
                try:
                    client = _get_mlflow_client(tracking_uri)
                    # Test connection
                    _list_experiments(client)
                    st.success(f"✅ Connected to MLflow at {tracking_uri}")
                except Exception as exc:
                    st.error(f"❌ Failed to connect to MLflow: {exc}")
                    st.info(
                        "💡 This may happen if:\n"
                        "- The tracking URI path doesn't exist or is inaccessible\n"
                        "- The MLflow server is not running\n"
                        "- There are network connectivity issues\n"
                        "- Permissions are insufficient to access the directory\n\n"
                        "Try using `./mlruns` as the tracking URI if you want to use local storage."
                    )
                    st.session_state.mlflow_client = None
                    st.session_state.mlflow_tracking_uri = None

    # Display experiments if connected
    if st.session_state.mlflow_client is not None:
        st.markdown("---")
        experiment_overview.display_experiment_overview(st.session_state.mlflow_client)
    else:
        st.info("Click 'Connect' to view experiments from the MLflow tracking server.")

    # Help section
    with st.expander("ℹ️ Help & Documentation"):
        st.markdown("""
        ### How to Use This Page

        1. **Configure Tracking URI**: Enter the path to your MLflow tracking directory or server URL
        2. **Connect**: Click the Connect button to establish connection
        3. **Browse Experiments**: View all experiments and select one to explore
        4. **View Runs**: See all training runs within the selected experiment
        5. **Inspect Details**: Select a run to view parameters, metrics, and artifacts

        ### MLflow Tracking URI

        The tracking URI determines where MLflow stores and retrieves experiment data:

        - **Local Directory**: `./mlruns` (default) - stores data in a local folder
        - **Local Server**: `http://localhost:5000` - connects to a local MLflow server
        - **Remote Server**: `http://your-server:5000` - connects to a remote tracking server

        ### Starting MLflow UI

        To launch the MLflow web UI locally:

        ```bash
        # Start MLflow UI on default port (5000)
        mlflow ui

        # Or specify custom port and tracking directory
        mlflow ui --port 5001 --backend-store-uri ./mlruns
        ```

        ### Training with MLflow

        To log experiments during training, use the `--tracker` flag:

        ```bash
        # Train with MLflow tracking
        python -m mammography.cli train-density \\
            --dataset mamografias \\
            --tracker mlflow \\
            --tracker-project "breast-density-classification" \\
            --tracker-run-name "resnet50-baseline"
        ```

        ### Understanding Metrics

        Common metrics logged during training:

        - **train/loss**: Training loss (lower is better)
        - **train/acc**: Training accuracy (higher is better)
        - **val/loss**: Validation loss (lower is better)
        - **val/acc**: Validation accuracy (higher is better)
        - **val/f1**: Validation F1 score (higher is better)
        - **val/kappa**: Cohen's Kappa score (higher is better)

        ### Artifacts

        Artifacts are files saved during training:

        - **Checkpoints**: Model weights (.pt, .pth files)
        - **Plots**: Confusion matrices, training curves
        - **Logs**: Text files with detailed training information
        - **Configs**: Configuration files used for the run

        ### Troubleshooting

        - **Connection Failed**: Verify the tracking URI is correct and accessible
        - **No Experiments**: Run a training command with MLflow tracking enabled
        - **Missing Metrics**: Ensure metrics were logged during training
        - **Slow Loading**: Reduce the max results or filter by status

        ### References

        - [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
        - [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
        - [MLflow Python API](https://mlflow.org/docs/latest/python_api/index.html)
        """)


if __name__ == "__main__":
    main()
