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

import io
import os
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mammography.apps.web_ui.utils import ensure_shared_session_state

try:
    from mammography.apps.web_ui.components.report_exporter import ReportExporter
except ImportError:
    ReportExporter = None

try:
    import streamlit as st
except Exception as exc:  # pragma: no cover - optional UI dependency
    st = None
    _STREAMLIT_IMPORT_ERROR = exc
else:
    _STREAMLIT_IMPORT_ERROR = None

try:
    import mlflow
    from mlflow.entities import Run, Experiment
    from mlflow.tracking import MlflowClient
except Exception as exc:  # pragma: no cover - optional MLflow dependency
    mlflow = None
    MlflowClient = None
    Run = None
    Experiment = None
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
    if "mlflow_client" not in st.session_state:
        st.session_state.mlflow_client = None
    if "mlflow_tracking_uri" not in st.session_state:
        st.session_state.mlflow_tracking_uri = None
    if "selected_experiment" not in st.session_state:
        st.session_state.selected_experiment = None
    if "selected_run" not in st.session_state:
        st.session_state.selected_run = None


def _format_timestamp(timestamp_ms: int) -> str:
    """Format Unix timestamp in milliseconds to human-readable string."""
    if timestamp_ms is None:
        return "N/A"
    try:
        dt = datetime.fromtimestamp(timestamp_ms / 1000.0)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "Invalid timestamp"


def _format_duration(start_ms: int, end_ms: Optional[int]) -> str:
    """Format duration between start and end timestamps."""
    if start_ms is None:
        return "N/A"
    if end_ms is None:
        end_ms = int(datetime.now().timestamp() * 1000)

    try:
        duration_sec = (end_ms - start_ms) / 1000.0
        hours = int(duration_sec // 3600)
        minutes = int((duration_sec % 3600) // 60)
        seconds = int(duration_sec % 60)

        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    except Exception:
        return "N/A"


def _export_plot_buttons(fig: Any, filename_prefix: str) -> None:
    """Display export buttons for a Plotly figure.

    Args:
        fig: Plotly figure object to export
        filename_prefix: Prefix for the exported filename (e.g., "accuracy_comparison")
    """
    st.markdown("**Export Plot:**")

    col1, col2, col3 = st.columns(3)

    with col1:
        try:
            # Export as PNG
            img_bytes = fig.to_image(format="png", width=1200, height=800, scale=2)
            st.download_button(
                label="📥 Download PNG",
                data=img_bytes,
                file_name=f"{filename_prefix}.png",
                mime="image/png",
                help="Download high-resolution PNG (1200x800, 2x scale)",
            )
        except Exception as exc:
            st.warning(f"PNG export requires kaleido: pip install kaleido")

    with col2:
        try:
            # Export as PDF
            pdf_bytes = fig.to_image(format="pdf", width=1200, height=800)
            st.download_button(
                label="📥 Download PDF",
                data=pdf_bytes,
                file_name=f"{filename_prefix}.pdf",
                mime="application/pdf",
                help="Download vector PDF for publication",
            )
        except Exception as exc:
            st.warning(f"PDF export requires kaleido: pip install kaleido")

    with col3:
        try:
            # Export as SVG
            svg_bytes = fig.to_image(format="svg", width=1200, height=800)
            st.download_button(
                label="📥 Download SVG",
                data=svg_bytes,
                file_name=f"{filename_prefix}.svg",
                mime="image/svg+xml",
                help="Download vector SVG for editing",
            )
        except Exception as exc:
            st.warning(f"SVG export requires kaleido: pip install kaleido")


def _render_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: str = "true",
    title: str = "Confusion Matrix",
) -> plt.Figure:
    """
    Render confusion matrix as heatmap using matplotlib.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Names for each class
        normalize: 'true', 'pred', 'all', or None
        title: Plot title

    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import confusion_matrix

    try:
        import seaborn as sns
    except ImportError:
        sns = None

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)

    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 7))

    # Determine format for annotations
    fmt = ".2f" if normalize else "d"

    # Plot heatmap
    if sns is not None:
        sns.heatmap(
            cm,
            ax=ax,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            square=True,
            linewidths=0.5,
            cbar_kws={"label": "Proportion" if normalize else "Count"},
        )
    else:
        # Fallback without seaborn
        im = ax.imshow(cm, cmap="Blues", aspect="equal")
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticks(range(len(class_names)))
        ax.set_yticklabels(class_names)

        # Add annotations
        for i in range(n_classes):
            for j in range(n_classes):
                text_value = format(cm[i, j], fmt)
                ax.text(j, i, text_value, ha="center", va="center", fontsize=10)

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Proportion" if normalize else "Count")

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    return fig


def _render_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "ROC Curves",
) -> plt.Figure:
    """
    Render ROC curves with AUC scores using matplotlib.

    Args:
        y_true: Ground truth labels (integer encoded)
        y_proba: Predicted probabilities (N, n_classes) array
        class_names: Names for each class
        title: Plot title

    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    # Determine number of classes
    n_classes = y_proba.shape[1] if len(y_proba.shape) > 1 else 2

    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Define colors for different classes
    colors = plt.cm.get_cmap("tab10").colors

    if n_classes == 2:
        # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)

        ax.plot(
            fpr,
            tpr,
            color=colors[0],
            linewidth=2,
            label=f"{class_names[1]} (AUC = {roc_auc:.3f})",
        )
    else:
        # Multi-class classification: One-vs-Rest
        y_true_binarized = label_binarize(y_true, classes=range(n_classes))

        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)

            ax.plot(
                fpr,
                tpr,
                color=colors[i % len(colors)],
                linewidth=2,
                label=f"{class_names[i]} (AUC = {roc_auc:.3f})",
            )

        # Compute micro-average ROC curve
        fpr_micro, tpr_micro, _ = roc_curve(y_true_binarized.ravel(), y_proba.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)

        ax.plot(
            fpr_micro,
            tpr_micro,
            color="deeppink",
            linewidth=2,
            linestyle="--",
            label=f"Micro-average (AUC = {roc_auc_micro:.3f})",
        )

    # Plot diagonal reference line
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random (AUC = 0.500)")

    # Formatting
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig


def _get_mlflow_client(tracking_uri: str) -> MlflowClient:
    """Get or create MLflow client with the specified tracking URI."""
    if (
        st.session_state.mlflow_client is not None
        and st.session_state.mlflow_tracking_uri == tracking_uri
    ):
        return st.session_state.mlflow_client

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    st.session_state.mlflow_client = client
    st.session_state.mlflow_tracking_uri = tracking_uri
    return client


def _list_experiments(client: MlflowClient) -> List[Experiment]:
    """List all MLflow experiments."""
    try:
        experiments = client.search_experiments()
        # Filter out deleted experiments
        return [exp for exp in experiments if exp.lifecycle_stage != "deleted"]
    except Exception as exc:
        st.error(f"❌ Failed to list experiments: {exc}")
        return []


def _list_runs(
    client: MlflowClient,
    experiment_id: str,
    max_results: int = 100,
) -> List[Run]:
    """List runs for a specific experiment."""
    try:
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            max_results=max_results,
            order_by=["start_time DESC"],
        )
        return runs
    except Exception as exc:
        st.error(f"❌ Failed to list runs: {exc}")
        return []


def _display_experiment_overview(client: MlflowClient) -> None:
    """Display overview of all experiments."""
    st.header("Experiments Overview")

    experiments = _list_experiments(client)

    if not experiments:
        st.info("No experiments found. Create an experiment by running a training command with MLflow tracking.")
        return

    # Create DataFrame for experiments
    exp_data = []
    for exp in experiments:
        # Count runs
        runs = _list_runs(client, exp.experiment_id, max_results=1000)
        num_runs = len(runs)
        active_runs = sum(1 for r in runs if r.info.status == "RUNNING")

        exp_data.append({
            "ID": exp.experiment_id,
            "Name": exp.name,
            "Total Runs": num_runs,
            "Active Runs": active_runs,
            "Artifact Location": exp.artifact_location,
            "Created": _format_timestamp(exp.creation_time),
        })

    df = pd.DataFrame(exp_data)

    # Display as interactive table
    st.dataframe(df, width="stretch", hide_index=True)

    # Experiment selector
    st.subheader("Select Experiment to View Runs")

    experiment_names = [exp.name for exp in experiments]
    selected_name = st.selectbox(
        "Experiment",
        options=experiment_names,
        help="Select an experiment to view its runs",
    )

    if selected_name:
        selected_exp = next((exp for exp in experiments if exp.name == selected_name), None)
        if selected_exp:
            st.session_state.selected_experiment = selected_exp.experiment_id
            _display_experiment_runs(client, selected_exp)


def _display_experiment_runs(client: MlflowClient, experiment: Experiment) -> None:
    """Display runs for a specific experiment."""
    st.header(f"Runs for Experiment: {experiment.name}")

    # Filters
    col1, col2 = st.columns([1, 1])

    with col1:
        max_results = st.number_input(
            "Max Results",
            min_value=10,
            max_value=1000,
            value=50,
            step=10,
            help="Maximum number of runs to display",
        )

    with col2:
        status_filter = st.multiselect(
            "Status Filter",
            options=["RUNNING", "FINISHED", "FAILED", "SCHEDULED", "KILLED"],
            default=["RUNNING", "FINISHED"],
            help="Filter runs by status",
        )

    # Get runs
    runs = _list_runs(client, experiment.experiment_id, max_results=int(max_results))

    # Apply status filter
    if status_filter:
        runs = [r for r in runs if r.info.status in status_filter]

    if not runs:
        st.info("No runs found for this experiment with the selected filters.")
        return

    st.write(f"Found {len(runs)} run(s)")

    # Create DataFrame for runs
    run_data = []
    for run in runs:
        metrics = run.data.metrics
        params = run.data.params

        # Extract key metrics
        acc = metrics.get("val/acc", metrics.get("val_acc", metrics.get("accuracy", 0.0)))
        loss = metrics.get("val/loss", metrics.get("val_loss", metrics.get("loss", 0.0)))

        run_data.append({
            "Run ID": run.info.run_id[:8],
            "Run Name": run.info.run_name or "Unnamed",
            "Status": run.info.status,
            "Start Time": _format_timestamp(run.info.start_time),
            "Duration": _format_duration(run.info.start_time, run.info.end_time),
            "Val Accuracy": f"{acc:.4f}" if acc > 0 else "N/A",
            "Val Loss": f"{loss:.4f}" if loss > 0 else "N/A",
            "Architecture": params.get("arch", params.get("model", "N/A")),
            "Epochs": params.get("epochs", "N/A"),
        })

    df = pd.DataFrame(run_data)

    # Display runs table
    st.dataframe(df, width="stretch", hide_index=True)

    # Run comparison section
    st.subheader("📊 Compare Multiple Runs")

    run_options = [f"{r.info.run_name or 'Unnamed'} ({r.info.run_id[:8]})" for r in runs]
    selected_run_options = st.multiselect(
        "Select Runs to Compare",
        options=run_options,
        default=[],
        help="Select multiple runs to compare their metrics",
    )

    if selected_run_options and len(selected_run_options) >= 2:
        # Extract selected runs
        selected_runs = []
        for option in selected_run_options:
            run_id_short = option.split("(")[-1].replace(")", "")
            run = next((r for r in runs if r.info.run_id.startswith(run_id_short)), None)
            if run:
                selected_runs.append(run)

        if selected_runs:
            _display_run_comparison(client, selected_runs)
    elif selected_run_options and len(selected_run_options) == 1:
        st.info("Select at least 2 runs to enable comparison.")

    # Run selector for details
    st.subheader("Select Run for Details")

    selected_run_option = st.selectbox(
        "Run",
        options=run_options,
        help="Select a run to view detailed information",
    )

    if selected_run_option:
        # Extract run ID from selection
        run_id_short = selected_run_option.split("(")[-1].replace(")", "")
        selected_run = next((r for r in runs if r.info.run_id.startswith(run_id_short)), None)

        if selected_run:
            _display_run_details(client, selected_run)


def _display_run_details(client: MlflowClient, run: Run) -> None:
    """Display detailed information for a specific run."""
    st.header(f"Run Details: {run.info.run_name or run.info.run_id[:8]}")

    # Run info
    st.subheader("Run Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Status", run.info.status)
        st.metric("Run ID", run.info.run_id[:8])

    with col2:
        st.metric("Start Time", _format_timestamp(run.info.start_time))
        if run.info.end_time:
            st.metric("End Time", _format_timestamp(run.info.end_time))

    with col3:
        st.metric("Duration", _format_duration(run.info.start_time, run.info.end_time))
        if run.info.user_id:
            st.metric("User", run.info.user_id)

    # Parameters
    st.subheader("Parameters")

    if run.data.params:
        params_df = pd.DataFrame(
            [{"Parameter": k, "Value": v} for k, v in sorted(run.data.params.items())]
        )
        st.dataframe(params_df, width="stretch", hide_index=True)
    else:
        st.info("No parameters logged for this run.")

    # Metrics
    st.subheader("Metrics")

    if run.data.metrics:
        metrics_df = pd.DataFrame(
            [{"Metric": k, "Value": f"{v:.6f}"} for k, v in sorted(run.data.metrics.items())]
        )
        st.dataframe(metrics_df, width="stretch", hide_index=True)

        # Plot metrics history if available
        try:
            metric_keys = list(run.data.metrics.keys())
            if metric_keys:
                st.subheader("Metrics History")

                selected_metrics = st.multiselect(
                    "Select metrics to plot",
                    options=metric_keys,
                    default=metric_keys[:min(3, len(metric_keys))],
                    help="Select one or more metrics to visualize over time",
                )

                if selected_metrics:
                    import plotly.graph_objects as go

                    fig = go.Figure()

                    for metric_key in selected_metrics:
                        # Get metric history
                        history = client.get_metric_history(run.info.run_id, metric_key)

                        if history:
                            steps = [h.step for h in history]
                            values = [h.value for h in history]

                            fig.add_trace(go.Scatter(
                                x=steps,
                                y=values,
                                mode='lines+markers',
                                name=metric_key,
                            ))

                    fig.update_layout(
                        title="Metrics Over Time",
                        xaxis_title="Step",
                        yaxis_title="Value",
                        hovermode="x unified",
                        height=400,
                    )

                    st.plotly_chart(fig, width="stretch")

                    # Export buttons
                    _export_plot_buttons(fig, f"metrics_history_{run.info.run_id[:8]}")
        except Exception as exc:
            st.warning(f"Could not plot metrics history: {exc}")
    else:
        st.info("No metrics logged for this run.")

    # Tags
    if run.data.tags:
        st.subheader("Tags")
        tags_df = pd.DataFrame(
            [{"Tag": k, "Value": v} for k, v in sorted(run.data.tags.items())]
        )
        st.dataframe(tags_df, width="stretch", hide_index=True)

    # Artifacts
    st.subheader("Artifacts")

    try:
        artifacts = client.list_artifacts(run.info.run_id)

        if artifacts:
            artifact_data = []
            for artifact in artifacts:
                artifact_data.append({
                    "Path": artifact.path,
                    "Size (bytes)": artifact.file_size if artifact.file_size else "N/A",
                    "Is Directory": "Yes" if artifact.is_dir else "No",
                })

            artifacts_df = pd.DataFrame(artifact_data)
            st.dataframe(artifacts_df, width="stretch", hide_index=True)

            # Download artifact link
            if run.info.artifact_uri:
                st.info(f"Artifact Location: {run.info.artifact_uri}")
        else:
            st.info("No artifacts logged for this run.")
    except Exception as exc:
        st.warning(f"Could not list artifacts: {exc}")

    # Confusion Matrix Visualization
    st.subheader("Confusion Matrix")

    try:
        # Try to load confusion matrix data from artifacts
        confusion_matrix_found = False

        # Look for confusion matrix artifacts (common naming patterns)
        cm_artifact_patterns = [
            "confusion_matrix.npy",
            "confusion_matrix.npz",
            "cm.npy",
            "predictions.npz",
        ]

        artifacts_list = client.list_artifacts(run.info.run_id)
        cm_artifact_path = None

        for artifact in artifacts_list:
            if any(pattern in artifact.path for pattern in cm_artifact_patterns):
                cm_artifact_path = artifact.path
                break

        if cm_artifact_path:
            try:
                # Download artifact
                local_path = client.download_artifacts(run.info.run_id, cm_artifact_path)

                # Load data
                if cm_artifact_path.endswith(".npz"):
                    data = np.load(local_path)
                    # Check for y_true and y_pred keys
                    if "y_true" in data and "y_pred" in data:
                        y_true = data["y_true"]
                        y_pred = data["y_pred"]
                        class_names = data.get("class_names", None)
                        if class_names is not None:
                            class_names = class_names.tolist()
                        confusion_matrix_found = True
                elif cm_artifact_path.endswith(".npy"):
                    # Assume it's the confusion matrix itself
                    cm_data = np.load(local_path)
                    # Display pre-computed confusion matrix
                    st.info("Displaying pre-computed confusion matrix from artifacts")
                    fig, ax = plt.subplots(figsize=(8, 7))

                    try:
                        import seaborn as sns
                        sns.heatmap(
                            cm_data,
                            ax=ax,
                            annot=True,
                            fmt=".2f" if cm_data.dtype == np.float64 else "d",
                            cmap="Blues",
                            square=True,
                            linewidths=0.5,
                            cbar_kws={"label": "Count"},
                        )
                    except ImportError:
                        im = ax.imshow(cm_data, cmap="Blues", aspect="equal")
                        for i in range(cm_data.shape[0]):
                            for j in range(cm_data.shape[1]):
                                ax.text(j, i, f"{cm_data[i, j]}", ha="center", va="center")
                        fig.colorbar(im, ax=ax)

                    ax.set_xlabel("Predicted Label", fontsize=12)
                    ax.set_ylabel("True Label", fontsize=12)
                    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
                    fig.tight_layout()

                    st.pyplot(fig)
                    plt.close(fig)
                    confusion_matrix_found = True

            except Exception as load_exc:
                st.warning(f"Could not load confusion matrix artifact: {load_exc}")

        # If we found y_true and y_pred, render the confusion matrix
        if confusion_matrix_found and 'y_true' in locals() and 'y_pred' in locals():
            # Add normalization options
            col1, col2 = st.columns([1, 3])

            with col1:
                normalize_option = st.selectbox(
                    "Normalization",
                    options=["None", "True Labels", "Predictions", "All"],
                    help="Choose how to normalize the confusion matrix",
                )

                normalize_map = {
                    "None": None,
                    "True Labels": "true",
                    "Predictions": "pred",
                    "All": "all",
                }
                normalize = normalize_map[normalize_option]

            # Render confusion matrix
            fig = _render_confusion_matrix(
                y_true,
                y_pred,
                class_names=class_names,
                normalize=normalize,
                title=f"Confusion Matrix ({normalize_option})",
            )

            st.pyplot(fig)
            plt.close(fig)

            # Add download button for the plot
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)

            st.download_button(
                label="📥 Download Confusion Matrix (PNG)",
                data=buf,
                file_name=f"confusion_matrix_{run.info.run_id[:8]}.png",
                mime="image/png",
            )

        if not confusion_matrix_found:
            st.info(
                "No confusion matrix data found. To visualize confusion matrices, "
                "save prediction results as artifacts during training:\n"
                "```python\n"
                "import numpy as np\n"
                "np.savez('predictions.npz', y_true=y_true, y_pred=y_pred, class_names=class_names)\n"
                "mlflow.log_artifact('predictions.npz')\n"
                "```"
            )

    except Exception as exc:
        st.warning(f"Could not display confusion matrix: {exc}")

    # ROC Curve Visualization
    st.subheader("ROC Curve")

    try:
        # Try to load prediction data with probabilities from artifacts
        roc_curve_found = False

        # Look for prediction artifacts with probability scores
        pred_artifact_patterns = [
            "predictions.npz",
            "predictions_proba.npz",
            "pred.npz",
            "proba.npz",
        ]

        artifacts_list = client.list_artifacts(run.info.run_id)
        pred_artifact_path = None

        for artifact in artifacts_list:
            if any(pattern in artifact.path for pattern in pred_artifact_patterns):
                pred_artifact_path = artifact.path
                break

        if pred_artifact_path:
            try:
                # Download artifact
                local_path = client.download_artifacts(run.info.run_id, pred_artifact_path)

                # Load data
                if pred_artifact_path.endswith(".npz"):
                    data = np.load(local_path)

                    # Check for required keys: y_true and y_proba
                    if "y_true" in data and "y_proba" in data:
                        y_true_roc = data["y_true"]
                        y_proba = data["y_proba"]
                        class_names_roc = data.get("class_names", None)
                        if class_names_roc is not None:
                            class_names_roc = class_names_roc.tolist()

                        # Validate shapes
                        if len(y_proba.shape) == 2 and y_proba.shape[0] == y_true_roc.shape[0]:
                            roc_curve_found = True

                            # Render ROC curve
                            fig_roc = _render_roc_curve(
                                y_true_roc,
                                y_proba,
                                class_names=class_names_roc,
                                title="ROC Curves with AUC Scores",
                            )

                            st.pyplot(fig_roc)
                            plt.close(fig_roc)

                            # Add download button for the plot
                            buf_roc = io.BytesIO()
                            fig_roc.savefig(buf_roc, format="png", dpi=150, bbox_inches="tight")
                            buf_roc.seek(0)

                            st.download_button(
                                label="📥 Download ROC Curve (PNG)",
                                data=buf_roc,
                                file_name=f"roc_curve_{run.info.run_id[:8]}.png",
                                mime="image/png",
                            )
                        else:
                            st.warning(
                                f"Invalid prediction data shape: y_proba shape {y_proba.shape} "
                                f"does not match y_true shape {y_true_roc.shape}"
                            )

            except Exception as load_exc:
                st.warning(f"Could not load prediction artifact for ROC curve: {load_exc}")

        if not roc_curve_found:
            st.info(
                "No prediction probability data found. To visualize ROC curves, "
                "save prediction probabilities as artifacts during training:\n"
                "```python\n"
                "import numpy as np\n"
                "# y_proba should be shape (N, n_classes) with probability scores\n"
                "np.savez('predictions.npz', y_true=y_true, y_pred=y_pred, y_proba=y_proba, class_names=class_names)\n"
                "mlflow.log_artifact('predictions.npz')\n"
                "```"
            )

    except Exception as exc:
        st.warning(f"Could not display ROC curve: {exc}")

    # Report Export Section
    st.subheader("📦 Export Report Package")

    try:
        if ReportExporter is None:
            st.info(
                "Report export functionality is available. "
                "The ReportExporter component should be accessible."
            )
        else:
            st.markdown(
                "Generate a publication-ready report package with training curves, "
                "confusion matrices, and metrics in multiple formats."
            )

            # Export configuration in an expander to keep UI clean
            with st.expander("🔧 Export Configuration", expanded=False):
                col1, col2 = st.columns(2)

                with col1:
                    export_formats = st.multiselect(
                        "Export Formats",
                        options=["png", "pdf", "svg"],
                        default=["png", "pdf"],
                        help="Select image formats for exported figures",
                        key=f"export_formats_{run.info.run_id[:8]}",
                    )

                with col2:
                    include_checkpoint = st.checkbox(
                        "Include Model Checkpoint",
                        value=False,
                        help="Include trained model weights in export (increases package size)",
                        key=f"include_checkpoint_{run.info.run_id[:8]}",
                    )

                # Output directory configuration
                default_output = f"reports/run_{run.info.run_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                output_dir = st.text_input(
                    "Output Directory",
                    value=default_output,
                    help="Directory where exported files will be saved",
                    key=f"output_dir_{run.info.run_id[:8]}",
                )

            # Export button
            col1, col2, col3 = st.columns([1, 1, 2])

            with col1:
                if st.button(
                    "🚀 Generate Report",
                    type="primary",
                    help="Generate report package with all visualizations and metrics",
                    key=f"generate_report_{run.info.run_id[:8]}",
                ):
                    with st.spinner("Generating report package..."):
                        try:
                            # Create exporter instance
                            exporter = ReportExporter()

                            # Export from MLflow
                            manifest = exporter.export_from_mlflow(
                                run_id=run.info.run_id,
                                output_dir=output_dir,
                                formats=export_formats if export_formats else ["png"],
                                include_checkpoint=include_checkpoint,
                            )

                            st.success(f"✅ Report generated in `{output_dir}`")

                            # Display summary
                            st.markdown("**Export Summary:**")
                            summary_col1, summary_col2 = st.columns(2)
                            with summary_col1:
                                st.metric("Exported Files", len(manifest.exported_files))
                            with summary_col2:
                                st.metric("Missing Files", len(manifest.missing_files))

                            # Show file list
                            if manifest.exported_files:
                                with st.expander("📄 Exported Files", expanded=False):
                                    for file in sorted(manifest.exported_files):
                                        st.text(f"✓ {file}")

                            if manifest.missing_files:
                                with st.expander("⚠️ Missing Files"):
                                    st.warning(
                                        "Some expected files were not found in the MLflow artifacts. "
                                        "This may be normal if certain metrics or artifacts were not logged during training."
                                    )
                                    for file in sorted(manifest.missing_files):
                                        st.text(f"✗ {file}")

                            # Store output directory in session state for download button
                            st.session_state[f"last_export_dir_{run.info.run_id[:8]}"] = output_dir

                        except Exception as e:
                            st.error(f"❌ Export failed: {e}")
                            import logging
                            logging.exception("Export failed")

            with col2:
                # Download ZIP button (enabled if report was generated)
                export_dir_key = f"last_export_dir_{run.info.run_id[:8]}"
                if export_dir_key in st.session_state:
                    last_export_dir = st.session_state[export_dir_key]

                    if st.button(
                        "📦 Download ZIP",
                        help="Create and download ZIP archive of the report",
                        key=f"download_zip_{run.info.run_id[:8]}",
                    ):
                        try:
                            with st.spinner("Creating ZIP archive..."):
                                # Create exporter instance
                                exporter = ReportExporter()

                                # Create temporary ZIP file
                                with tempfile.NamedTemporaryFile(
                                    mode="wb",
                                    suffix=".zip",
                                    delete=False,
                                ) as tmp_zip:
                                    tmp_zip_path = tmp_zip.name

                                # Create ZIP archive
                                zip_file = exporter.create_zip_archive(
                                    last_export_dir,
                                    tmp_zip_path,
                                )

                                # Read ZIP file for download
                                with open(zip_file, "rb") as f:
                                    zip_bytes = f.read()

                                # Provide download button
                                zip_filename = f"report_{run.info.run_id[:8]}.zip"
                                st.download_button(
                                    label="⬇️ Download Report Archive",
                                    data=zip_bytes,
                                    file_name=zip_filename,
                                    mime="application/zip",
                                    key=f"download_button_{run.info.run_id[:8]}",
                                )

                                st.success(f"✅ Archive ready: {zip_filename}")

                                # Clean up temporary file
                                try:
                                    Path(tmp_zip_path).unlink()
                                except Exception:
                                    pass

                        except Exception as e:
                            st.error(f"❌ Failed to create ZIP archive: {e}")
                else:
                    st.button(
                        "📦 Download ZIP",
                        disabled=True,
                        help="Generate report first to enable download",
                        key=f"download_zip_disabled_{run.info.run_id[:8]}",
                    )

            with col3:
                st.markdown(
                    "<div style='font-size: 0.85em; color: #666; padding-top: 8px;'>"
                    "💡 <b>Tip:</b> Generate report to create publication-ready figures, "
                    "then download as ZIP for easy sharing."
                    "</div>",
                    unsafe_allow_html=True,
                )

    except Exception as exc:
        st.warning(f"Could not display export section: {exc}")


def _display_run_comparison(client: MlflowClient, runs: List[Run]) -> None:
    """Display comparison charts for multiple runs."""
    st.header(f"Comparing {len(runs)} Runs")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        st.error("Plotly is required for comparison charts. Install with: pip install plotly")
        return

    # Get all available metrics from selected runs
    all_metrics = set()
    for run in runs:
        all_metrics.update(run.data.metrics.keys())

    # Common training/validation metrics
    accuracy_metrics = [m for m in all_metrics if 'acc' in m.lower()]
    loss_metrics = [m for m in all_metrics if 'loss' in m.lower()]

    # Create tabs for different comparison views
    tab1, tab2, tab3 = st.tabs(["📈 Accuracy Curves", "📉 Loss Curves", "📊 Final Metrics"])

    with tab1:
        if accuracy_metrics:
            st.subheader("Accuracy Comparison")

            # Let user select which accuracy metrics to plot
            selected_acc_metrics = st.multiselect(
                "Select accuracy metrics",
                options=accuracy_metrics,
                default=accuracy_metrics[:min(2, len(accuracy_metrics))],
                key="acc_metrics",
            )

            if selected_acc_metrics:
                fig = go.Figure()

                for run in runs:
                    run_name = run.info.run_name or run.info.run_id[:8]

                    for metric_key in selected_acc_metrics:
                        if metric_key in run.data.metrics:
                            try:
                                # Get metric history
                                history = client.get_metric_history(run.info.run_id, metric_key)

                                if history:
                                    steps = [h.step for h in history]
                                    values = [h.value for h in history]

                                    fig.add_trace(go.Scatter(
                                        x=steps,
                                        y=values,
                                        mode='lines+markers',
                                        name=f"{run_name} - {metric_key}",
                                        hovertemplate='<b>%{fullData.name}</b><br>Step: %{x}<br>Accuracy: %{y:.4f}<extra></extra>',
                                    ))
                            except Exception as exc:
                                st.warning(f"Could not fetch {metric_key} for {run_name}: {exc}")

                fig.update_layout(
                    title="Accuracy Comparison Across Runs",
                    xaxis_title="Step/Epoch",
                    yaxis_title="Accuracy",
                    hovermode="x unified",
                    height=500,
                    legend=dict(
                        orientation="v",
                        yanchor="bottom",
                        y=0.01,
                        xanchor="right",
                        x=0.99
                    )
                )

                st.plotly_chart(fig, width="stretch")

                # Export buttons
                _export_plot_buttons(fig, "accuracy_comparison")
            else:
                st.info("Select at least one accuracy metric to visualize.")
        else:
            st.info("No accuracy metrics found in the selected runs.")

    with tab2:
        if loss_metrics:
            st.subheader("Loss Comparison")

            # Let user select which loss metrics to plot
            selected_loss_metrics = st.multiselect(
                "Select loss metrics",
                options=loss_metrics,
                default=loss_metrics[:min(2, len(loss_metrics))],
                key="loss_metrics",
            )

            if selected_loss_metrics:
                fig = go.Figure()

                for run in runs:
                    run_name = run.info.run_name or run.info.run_id[:8]

                    for metric_key in selected_loss_metrics:
                        if metric_key in run.data.metrics:
                            try:
                                # Get metric history
                                history = client.get_metric_history(run.info.run_id, metric_key)

                                if history:
                                    steps = [h.step for h in history]
                                    values = [h.value for h in history]

                                    fig.add_trace(go.Scatter(
                                        x=steps,
                                        y=values,
                                        mode='lines+markers',
                                        name=f"{run_name} - {metric_key}",
                                        hovertemplate='<b>%{fullData.name}</b><br>Step: %{x}<br>Loss: %{y:.4f}<extra></extra>',
                                    ))
                            except Exception as exc:
                                st.warning(f"Could not fetch {metric_key} for {run_name}: {exc}")

                fig.update_layout(
                    title="Loss Comparison Across Runs",
                    xaxis_title="Step/Epoch",
                    yaxis_title="Loss",
                    hovermode="x unified",
                    height=500,
                    legend=dict(
                        orientation="v",
                        yanchor="top",
                        y=0.99,
                        xanchor="right",
                        x=0.99
                    )
                )

                st.plotly_chart(fig, width="stretch")

                # Export buttons
                _export_plot_buttons(fig, "loss_comparison")
            else:
                st.info("Select at least one loss metric to visualize.")
        else:
            st.info("No loss metrics found in the selected runs.")

    with tab3:
        st.subheader("Final Metrics Comparison")

        # Get all metrics for comparison
        comparison_metrics = st.multiselect(
            "Select metrics to compare",
            options=sorted(all_metrics),
            default=sorted(all_metrics)[:min(4, len(all_metrics))],
            help="Select metrics to compare final values across runs",
        )

        if comparison_metrics:
            # Create bar chart for each metric
            for metric in comparison_metrics:
                run_names = []
                metric_values = []

                for run in runs:
                    if metric in run.data.metrics:
                        run_names.append(run.info.run_name or run.info.run_id[:8])
                        metric_values.append(run.data.metrics[metric])

                if metric_values:
                    fig = go.Figure()

                    fig.add_trace(go.Bar(
                        x=run_names,
                        y=metric_values,
                        text=[f"{v:.4f}" for v in metric_values],
                        textposition='auto',
                        hovertemplate='<b>%{x}</b><br>Value: %{y:.6f}<extra></extra>',
                    ))

                    fig.update_layout(
                        title=f"{metric.replace('_', ' ').title()} Comparison",
                        xaxis_title="Run",
                        yaxis_title="Value",
                        height=400,
                    )

                    st.plotly_chart(fig, width="stretch")

                    # Export buttons
                    _export_plot_buttons(fig, f"final_metrics_{metric}")
        else:
            st.info("Select metrics to display comparison charts.")

        # Summary table
        st.subheader("Summary Table")

        summary_data = []
        for run in runs:
            row_data = {
                "Run Name": run.info.run_name or run.info.run_id[:8],
                "Status": run.info.status,
                "Duration": _format_duration(run.info.start_time, run.info.end_time),
            }

            # Add selected metrics
            for metric in comparison_metrics:
                if metric in run.data.metrics:
                    row_data[metric] = f"{run.data.metrics[metric]:.6f}"
                else:
                    row_data[metric] = "N/A"

            # Add key parameters
            params_to_show = ['arch', 'model', 'epochs', 'batch_size', 'lr']
            for param in params_to_show:
                if param in run.data.params:
                    row_data[param] = run.data.params[param]

            summary_data.append(row_data)

        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, width="stretch", hide_index=True)

    # Export Comparison Report Section
    st.subheader("📦 Export Comparison Report")

    try:
        if ReportExporter is None:
            st.info(
                "Report export functionality requires the ReportExporter component."
            )
        else:
            st.markdown(
                "Export individual run reports or batch export all selected runs for comparison."
            )

            with st.expander("🔧 Batch Export Configuration", expanded=False):
                col1, col2 = st.columns(2)

                with col1:
                    batch_export_formats = st.multiselect(
                        "Export Formats",
                        options=["png", "pdf", "svg"],
                        default=["png", "pdf"],
                        help="Select image formats for exported figures",
                        key="batch_export_formats",
                    )

                with col2:
                    batch_include_checkpoint = st.checkbox(
                        "Include Model Checkpoints",
                        value=False,
                        help="Include trained model weights in export (increases package size significantly)",
                        key="batch_include_checkpoint",
                    )

                batch_output_base = st.text_input(
                    "Output Base Directory",
                    value=f"reports/comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    help="Base directory for batch export (subdirectories will be created for each run)",
                    key="batch_output_base",
                )

            # Export buttons
            col1, col2, col3 = st.columns([1, 1, 2])

            with col1:
                if st.button(
                    "🚀 Batch Export All Runs",
                    type="primary",
                    help=f"Export all {len(runs)} selected runs to separate directories",
                    key="batch_export_button",
                ):
                    with st.spinner(f"Exporting {len(runs)} runs..."):
                        try:
                            exporter = ReportExporter()
                            export_results = []
                            failed_exports = []

                            # Progress tracking
                            progress_bar = st.progress(0)
                            status_text = st.empty()

                            for idx, run in enumerate(runs):
                                run_name = run.info.run_name or run.info.run_id[:8]
                                status_text.text(f"Exporting {idx + 1}/{len(runs)}: {run_name}")

                                try:
                                    # Create subdirectory for each run
                                    run_output_dir = Path(batch_output_base) / f"run_{run.info.run_id[:8]}"

                                    manifest = exporter.export_from_mlflow(
                                        run_id=run.info.run_id,
                                        output_dir=str(run_output_dir),
                                        formats=batch_export_formats if batch_export_formats else ["png"],
                                        include_checkpoint=batch_include_checkpoint,
                                    )

                                    export_results.append({
                                        "run_id": run.info.run_id[:8],
                                        "run_name": run_name,
                                        "output_dir": str(run_output_dir),
                                        "exported_files": len(manifest.exported_files),
                                        "missing_files": len(manifest.missing_files),
                                        "status": "success",
                                    })

                                except Exception as e:
                                    failed_exports.append({
                                        "run_id": run.info.run_id[:8],
                                        "run_name": run_name,
                                        "error": str(e),
                                    })
                                    export_results.append({
                                        "run_id": run.info.run_id[:8],
                                        "run_name": run_name,
                                        "output_dir": "N/A",
                                        "exported_files": 0,
                                        "missing_files": 0,
                                        "status": "failed",
                                    })

                                # Update progress
                                progress_bar.progress((idx + 1) / len(runs))

                            status_text.empty()
                            progress_bar.empty()

                            # Show results
                            success_count = sum(1 for r in export_results if r["status"] == "success")
                            st.success(
                                f"✅ Batch export completed: {success_count}/{len(runs)} runs exported successfully"
                            )

                            # Results summary
                            results_df = pd.DataFrame(export_results)
                            st.dataframe(results_df, width="stretch", hide_index=True)

                            if failed_exports:
                                with st.expander("⚠️ Failed Exports", expanded=True):
                                    for fail in failed_exports:
                                        st.error(
                                            f"**{fail['run_name']}** ({fail['run_id']}): {fail['error']}"
                                        )

                            # Store output directory in session state
                            st.session_state["last_batch_export_dir"] = batch_output_base

                        except Exception as e:
                            st.error(f"❌ Batch export failed: {e}")
                            import logging
                            logging.exception("Batch export failed")

            with col2:
                # Download ZIP button for batch export
                if "last_batch_export_dir" in st.session_state:
                    if st.button(
                        "📦 Download All as ZIP",
                        help="Create and download ZIP archive of all exported runs",
                        key="batch_download_zip_button",
                    ):
                        try:
                            with st.spinner("Creating ZIP archive of all runs..."):
                                exporter = ReportExporter()

                                # Create temporary ZIP file
                                with tempfile.NamedTemporaryFile(
                                    mode="wb",
                                    suffix=".zip",
                                    delete=False,
                                ) as tmp_zip:
                                    tmp_zip_path = tmp_zip.name

                                # Create ZIP archive
                                batch_export_dir = st.session_state["last_batch_export_dir"]
                                zip_file = exporter.create_zip_archive(
                                    batch_export_dir,
                                    tmp_zip_path,
                                )

                                # Read ZIP file for download
                                with open(zip_file, "rb") as f:
                                    zip_bytes = f.read()

                                # Provide download button
                                zip_filename = f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
                                st.download_button(
                                    label="⬇️ Download Comparison Archive",
                                    data=zip_bytes,
                                    file_name=zip_filename,
                                    mime="application/zip",
                                    key="batch_download_button",
                                )

                                st.success(f"✅ Archive ready: {zip_filename}")

                                # Clean up temporary file
                                try:
                                    Path(tmp_zip_path).unlink()
                                except Exception:
                                    pass

                        except Exception as e:
                            st.error(f"❌ Failed to create ZIP archive: {e}")
                else:
                    st.button(
                        "📦 Download All as ZIP",
                        disabled=True,
                        help="Run batch export first to enable download",
                        key="batch_download_zip_disabled",
                    )

            with col3:
                st.markdown(
                    f"<div style='font-size: 0.85em; color: #666; padding-top: 8px;'>"
                    f"💡 <b>Tip:</b> Batch export creates separate directories for all {len(runs)} runs. "
                    f"Download as ZIP for convenient sharing or archiving."
                    f"</div>",
                    unsafe_allow_html=True,
                )

    except Exception as exc:
        st.warning(f"Could not display export section: {exc}")


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
        _display_experiment_overview(st.session_state.mlflow_client)
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
