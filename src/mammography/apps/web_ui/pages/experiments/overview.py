"""Experiment overview and run table sections for the Streamlit Experiments page."""

from __future__ import annotations

from typing import Any

import pandas as pd

from mammography.apps.web_ui.pages.experiments import comparison as experiment_comparison
from mammography.apps.web_ui.pages.experiments import run_details as experiment_run_details
from mammography.apps.web_ui.pages.experiments import mlflow_client as experiment_mlflow_client
from mammography.apps.web_ui.pages.experiments.formatters import (
    format_duration as _format_duration,
    format_timestamp as _format_timestamp,
)

try:
    import streamlit as st
except Exception as exc:  # pragma: no cover - optional UI dependency
    st = None
    _STREAMLIT_IMPORT_ERROR = exc
else:
    _STREAMLIT_IMPORT_ERROR = None


def display_experiment_overview(client: Any) -> None:
    """Display overview of all experiments."""
    st.header("Experiments Overview")

    try:
        experiments = experiment_mlflow_client.list_experiments(client)
    except Exception as exc:
        st.error(f"Failed to list experiments: {exc}")
        return

    if not experiments:
        st.info("No experiments found. Create an experiment by running a training command with MLflow tracking.")
        return

    # Create DataFrame for experiments
    exp_data = []
    for exp in experiments:
        # Count runs
        try:
            runs = experiment_mlflow_client.list_runs(client, exp.experiment_id, max_results=1000)
        except Exception:
            runs = []
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
            display_experiment_runs(client, selected_exp)

def display_experiment_runs(client: Any, experiment: Any) -> None:
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
    try:
        runs = experiment_mlflow_client.list_runs(
            client, experiment.experiment_id, max_results=int(max_results)
        )
    except Exception as exc:
        st.error(f"Failed to list runs: {exc}")
        return

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
            experiment_comparison.display_run_comparison(client, selected_runs)
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
            experiment_run_details.display_run_details(client, selected_run)