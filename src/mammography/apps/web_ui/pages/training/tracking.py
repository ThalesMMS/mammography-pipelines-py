"""Tracking and analysis sections for the Streamlit Training page."""

from __future__ import annotations

from typing import Any, Dict

try:
    import streamlit as st
except Exception as exc:  # pragma: no cover - optional UI dependency
    st = None
    _STREAMLIT_IMPORT_ERROR = exc
else:
    _STREAMLIT_IMPORT_ERROR = None


def _require_streamlit() -> None:
    if st is None:
        raise ImportError("Streamlit is required to run the web UI dashboard.") from _STREAMLIT_IMPORT_ERROR


def render_analysis_section() -> Dict[str, Any]:
    """Render analysis and outputs section and return parameters."""
    st.subheader("📊 Analysis & Outputs")

    config = {}

    col1, col2 = st.columns(2)

    with col1:
        config["gradcam"] = st.checkbox(
            "Generate GradCAM",
            value=False,
            help="Generate GradCAM visualizations during validation",
        )

        if config["gradcam"]:
            config["gradcam_limit"] = st.number_input(
                "GradCAM Limit",
                min_value=1,
                value=4,
                step=1,
                help="Maximum number of GradCAM visualizations to generate",
            )
        else:
            config["gradcam_limit"] = 4

    with col2:
        config["save_val_preds"] = st.checkbox(
            "Save Validation Predictions",
            value=False,
            help="Save validation predictions to CSV",
        )

        config["export_val_embeddings"] = st.checkbox(
            "Export Validation Embeddings",
            value=False,
            help="Export validation embeddings for analysis",
        )

    # Profiling
    with st.expander("🔍 Profiling"):
        config["profile"] = st.checkbox(
            "Enable Profiling",
            value=False,
            help="Enable PyTorch profiler in first epoch",
        )

        if config["profile"]:
            config["profile_dir"] = st.text_input(
                "Profile Output Directory",
                value="outputs/profiler",
                help="Directory for saving profiler traces",
            )
        else:
            config["profile_dir"] = "outputs/profiler"

    return config

def render_tracking_section() -> Dict[str, Any]:
    """Render experiment tracking section and return parameters."""
    st.subheader("📈 Experiment Tracking")

    config = {}

    config["tracker"] = st.selectbox(
        "Tracking Backend",
        options=["none", "mlflow", "wandb"],
        help="Experiment tracking backend (none/mlflow/wandb)",
    )

    if config["tracker"] != "none":
        col1, col2 = st.columns(2)

        with col1:
            config["tracker_project"] = st.text_input(
                "Project/Experiment Name",
                value="mammography-density",
                help="Name of the project or experiment for tracking",
            )

        with col2:
            config["tracker_run_name"] = st.text_input(
                "Run Name",
                value="",
                help="Optional name for this specific run",
            )

        if config["tracker"] == "mlflow":
            config["tracker_uri"] = st.text_input(
                "MLflow Tracking URI",
                value="./mlruns",
                help="MLflow tracking server URI or local path",
            )
        else:
            config["tracker_uri"] = ""
    else:
        config["tracker_project"] = ""
        config["tracker_run_name"] = ""
        config["tracker_uri"] = ""

    return config
