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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

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
    st.dataframe(df, use_container_width=True, hide_index=True)

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
    st.dataframe(df, use_container_width=True, hide_index=True)

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
        st.dataframe(params_df, use_container_width=True, hide_index=True)
    else:
        st.info("No parameters logged for this run.")

    # Metrics
    st.subheader("Metrics")

    if run.data.metrics:
        metrics_df = pd.DataFrame(
            [{"Metric": k, "Value": f"{v:.6f}"} for k, v in sorted(run.data.metrics.items())]
        )
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

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

                    st.plotly_chart(fig, use_container_width=True)

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
        st.dataframe(tags_df, use_container_width=True, hide_index=True)

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
            st.dataframe(artifacts_df, use_container_width=True, hide_index=True)

            # Download artifact link
            if run.info.artifact_uri:
                st.info(f"Artifact Location: {run.info.artifact_uri}")
        else:
            st.info("No artifacts logged for this run.")
    except Exception as exc:
        st.warning(f"Could not list artifacts: {exc}")


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

                st.plotly_chart(fig, use_container_width=True)

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

                st.plotly_chart(fig, use_container_width=True)

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

                    st.plotly_chart(fig, use_container_width=True)

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
        st.dataframe(summary_df, use_container_width=True, hide_index=True)


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

    st.markdown("""
    <div style="background-color: #fff3cd; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #ffc107; margin-bottom: 1rem;">
    <h3 style="color: #856404; margin-top: 0;">⚠️ EDUCATIONAL RESEARCH USE ONLY</h3>
    <p style="color: #856404; margin-bottom: 0;">
    This tool is for <strong>educational and research purposes only</strong>. It is <strong>NOT</strong>
    intended for clinical diagnosis or treatment. All results should be validated by qualified
    medical professionals before any clinical decision-making.
    </p>
    </div>
    """, unsafe_allow_html=True)

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
