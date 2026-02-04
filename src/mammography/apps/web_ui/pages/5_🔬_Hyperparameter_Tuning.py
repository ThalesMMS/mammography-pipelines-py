#
# 5_🔬_Hyperparameter_Tuning.py
# mammography-pipelines
#
# Streamlit page for viewing and analyzing Optuna hyperparameter tuning studies.
#
# Thales Matheus Mendonca Santos - February 2026
#
"""Hyperparameter Tuning page for viewing and analyzing Optuna optimization studies."""

from __future__ import annotations

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
    import optuna
    from optuna.study import Study
    from optuna.trial import FrozenTrial, TrialState
except Exception as exc:  # pragma: no cover - optional Optuna dependency
    optuna = None
    Study = None
    FrozenTrial = None
    TrialState = None
    _OPTUNA_IMPORT_ERROR = exc
else:
    _OPTUNA_IMPORT_ERROR = None


def _require_streamlit() -> None:
    """Raise ImportError if Streamlit is not available."""
    if st is None:
        raise ImportError(
            "Streamlit is required to run the web UI dashboard."
        ) from _STREAMLIT_IMPORT_ERROR


def _require_optuna() -> None:
    """Raise ImportError if Optuna is not available."""
    if optuna is None:
        raise ImportError(
            "Optuna is required for hyperparameter tuning. Install with: pip install optuna"
        ) from _OPTUNA_IMPORT_ERROR


def _ensure_session_defaults() -> None:
    """Initialize session state with default values."""
    if "optuna_storage_uri" not in st.session_state:
        st.session_state.optuna_storage_uri = None
    if "selected_study_name" not in st.session_state:
        st.session_state.selected_study_name = None


def _format_duration(duration_seconds: Optional[float]) -> str:
    """Format duration in seconds to human-readable string."""
    if duration_seconds is None:
        return "N/A"
    try:
        hours = int(duration_seconds // 3600)
        minutes = int((duration_seconds % 3600) // 60)
        seconds = int(duration_seconds % 60)

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
        filename_prefix: Prefix for the exported filename (e.g., "optimization_history")
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


def _list_studies(storage_uri: str) -> List[str]:
    """List all studies in the specified storage.

    Args:
        storage_uri: Optuna storage URI (e.g., "sqlite:///optuna.db")

    Returns:
        List of study names
    """
    try:
        study_summaries = optuna.study.get_all_study_summaries(storage=storage_uri)
        return [summary.study_name for summary in study_summaries]
    except Exception as exc:
        st.error(f"❌ Failed to list studies: {exc}")
        return []


def _load_study(study_name: str, storage_uri: str) -> Optional[Study]:
    """Load an Optuna study from storage.

    Args:
        study_name: Name of the study to load
        storage_uri: Optuna storage URI

    Returns:
        Loaded Study object or None if failed
    """
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_uri)
        return study
    except Exception as exc:
        st.error(f"❌ Failed to load study '{study_name}': {exc}")
        return None


def _display_studies_overview(storage_uri: str) -> None:
    """Display overview of all studies in the storage."""
    st.header("Studies Overview")

    study_names = _list_studies(storage_uri)

    if not study_names:
        st.info(
            "No Optuna studies found. Create a study by running the hyperparameter tuning command:\n\n"
            "```bash\n"
            "python -m mammography.cli tune --dataset mamografias --n-trials 50 --storage sqlite:///optuna.db\n"
            "```"
        )
        return

    st.write(f"Found {len(study_names)} study/studies")

    # Load study summaries
    study_data = []
    for study_name in study_names:
        try:
            summaries = optuna.study.get_all_study_summaries(storage=storage_uri)
            summary = next((s for s in summaries if s.study_name == study_name), None)
            if summary:
                study_data.append({
                    "Study Name": study_name,
                    "Direction": summary.direction.name,
                    "Total Trials": summary.n_trials,
                    "Best Value": f"{summary.best_trial.value:.4f}" if summary.best_trial else "N/A",
                    "Created": summary.datetime_start.strftime("%Y-%m-%d %H:%M:%S") if summary.datetime_start else "N/A",
                })
        except Exception as exc:
            st.warning(f"Could not load summary for study '{study_name}': {exc}")

    if study_data:
        df = pd.DataFrame(study_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

    # Study selector
    st.subheader("Select Study to View Details")

    selected_study = st.selectbox(
        "Study",
        options=study_names,
        help="Select a study to view detailed optimization results",
    )

    if selected_study:
        st.session_state.selected_study_name = selected_study
        study = _load_study(selected_study, storage_uri)
        if study:
            _display_study_details(study)


def _display_study_details(study: Study) -> None:
    """Display detailed information for a specific study."""
    st.header(f"Study: {study.study_name}")

    # Get all trials
    trials = study.trials
    completed_trials = [t for t in trials if t.state == TrialState.COMPLETE]
    pruned_trials = [t for t in trials if t.state == TrialState.PRUNED]
    failed_trials = [t for t in trials if t.state == TrialState.FAIL]

    # Study statistics
    st.subheader("Study Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Trials", len(trials))
        st.metric("Completed", len(completed_trials))

    with col2:
        st.metric("Pruned", len(pruned_trials))
        st.metric("Failed", len(failed_trials))

    with col3:
        if completed_trials:
            best_value = study.best_value
            st.metric("Best Value", f"{best_value:.6f}")
        else:
            st.metric("Best Value", "N/A")

    with col4:
        if completed_trials:
            best_trial_num = study.best_trial.number
            st.metric("Best Trial #", best_trial_num)
        else:
            st.metric("Best Trial #", "N/A")

    # Best trial parameters
    if completed_trials:
        st.subheader("Best Hyperparameters")

        best_params = study.best_params
        params_df = pd.DataFrame(
            [{"Parameter": k, "Value": v} for k, v in sorted(best_params.items())]
        )
        st.dataframe(params_df, use_container_width=True, hide_index=True)

    # Trials table
    st.subheader("All Trials")

    # Trial state filter
    state_filter = st.multiselect(
        "Filter by State",
        options=["COMPLETE", "PRUNED", "FAIL", "RUNNING"],
        default=["COMPLETE", "PRUNED"],
        help="Filter trials by their completion state",
    )

    # Apply filter
    filtered_trials = [
        t for t in trials
        if t.state.name in state_filter
    ]

    if not filtered_trials:
        st.info("No trials match the selected filters.")
    else:
        # Create trials DataFrame
        trials_data = []
        for trial in filtered_trials:
            row = {
                "Trial #": trial.number,
                "State": trial.state.name,
                "Value": f"{trial.value:.6f}" if trial.value is not None else "N/A",
                "Duration": _format_duration(trial.duration.total_seconds() if trial.duration else None),
            }

            # Add key parameters (up to 4)
            if trial.params:
                for idx, (param_name, param_value) in enumerate(sorted(trial.params.items())[:4]):
                    row[param_name] = param_value

            trials_data.append(row)

        trials_df = pd.DataFrame(trials_data)
        st.dataframe(trials_df, use_container_width=True, hide_index=True)

    # Visualization tabs
    if completed_trials:
        st.subheader("Optimization Analysis")

        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Optimization History",
            "🎯 Parameter Importance",
            "🔀 Parallel Coordinates",
            "📊 Trial Distributions"
        ])

        with tab1:
            _display_optimization_history(study)

        with tab2:
            _display_parameter_importance(study)

        with tab3:
            _display_parallel_coordinates(study)

        with tab4:
            _display_parameter_distributions(study)


def _display_optimization_history(study: Study) -> None:
    """Display optimization history plot."""
    st.markdown("### Optimization History")
    st.markdown(
        "Shows how the objective value improved over trials. "
        "The blue line represents each trial's value, and the red line shows the best value so far."
    )

    try:
        fig = optuna.visualization.plot_optimization_history(study)
        st.plotly_chart(fig, use_container_width=True)
        _export_plot_buttons(fig, f"optimization_history_{study.study_name}")
    except Exception as exc:
        st.error(f"Failed to generate optimization history plot: {exc}")


def _display_parameter_importance(study: Study) -> None:
    """Display parameter importance plot."""
    st.markdown("### Parameter Importance")
    st.markdown(
        "Shows which hyperparameters had the most impact on the objective value. "
        "Higher importance means the parameter had a stronger effect on performance."
    )

    try:
        # Check if we have enough completed trials for importance analysis
        completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if len(completed_trials) < 2:
            st.info("Need at least 2 completed trials to compute parameter importance.")
            return

        fig = optuna.visualization.plot_param_importances(study)
        st.plotly_chart(fig, use_container_width=True)
        _export_plot_buttons(fig, f"param_importance_{study.study_name}")
    except Exception as exc:
        st.error(f"Failed to generate parameter importance plot: {exc}")


def _display_parallel_coordinates(study: Study) -> None:
    """Display parallel coordinates plot."""
    st.markdown("### Parallel Coordinates Plot")
    st.markdown(
        "Visualizes the relationship between all hyperparameters and the objective value. "
        "Each line represents one trial. Better trials are shown in warmer colors (red/orange)."
    )

    try:
        fig = optuna.visualization.plot_parallel_coordinate(study)
        st.plotly_chart(fig, use_container_width=True)
        _export_plot_buttons(fig, f"parallel_coords_{study.study_name}")
    except Exception as exc:
        st.error(f"Failed to generate parallel coordinates plot: {exc}")


def _display_parameter_distributions(study: Study) -> None:
    """Display parameter slice plots showing distributions."""
    st.markdown("### Parameter Slice Plots")
    st.markdown(
        "Shows the distribution of objective values for each parameter value. "
        "Helps identify which parameter ranges lead to better performance."
    )

    try:
        fig = optuna.visualization.plot_slice(study)
        st.plotly_chart(fig, use_container_width=True)
        _export_plot_buttons(fig, f"param_slices_{study.study_name}")
    except Exception as exc:
        st.error(f"Failed to generate parameter slice plots: {exc}")


def main() -> None:
    """Render the hyperparameter tuning page."""
    _require_streamlit()

    st.set_page_config(
        page_title="Hyperparameter Tuning - Mammography Pipelines",
        page_icon="🔬",
        layout="wide",
    )

    # Initialize shared session state for cross-page data persistence
    try:
        ensure_shared_session_state()
        _ensure_session_defaults()
    except Exception as exc:
        st.error(f"❌ Failed to initialize session state: {exc}")
        st.stop()

    st.title("🔬 Hyperparameter Tuning")

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

    # Check if Optuna is available
    try:
        _require_optuna()
    except ImportError as exc:
        st.error(f"❌ Optuna is not installed: {exc}")
        st.markdown("""
        ### Installation Instructions

        To use hyperparameter tuning visualization, install Optuna:

        ```bash
        pip install optuna
        ```

        Then restart the Streamlit application.
        """)
        return

    st.header("Optuna Hyperparameter Optimization")

    st.markdown("""
    View and analyze hyperparameter optimization studies created with Optuna. This page displays:

    - **Studies**: Collections of optimization trials exploring different hyperparameter combinations
    - **Trials**: Individual training runs with specific hyperparameter values
    - **Optimization History**: How performance improved over trials
    - **Parameter Importance**: Which hyperparameters had the most impact
    - **Parallel Coordinates**: Multi-dimensional visualization of hyperparameter relationships
    - **Distributions**: How different parameter values affect performance
    """)

    st.info(
        "💡 **Quick Start:** Enter your Optuna storage URI (e.g., `sqlite:///optuna.db` or local directory path), "
        "click 'Connect', then browse your optimization studies."
    )

    # Optuna storage configuration
    st.subheader("Optuna Storage Configuration")

    col1, col2 = st.columns([3, 1])

    with col1:
        # Default storage URI - check common locations
        default_uri = os.environ.get("OPTUNA_STORAGE", "sqlite:///./outputs/tune/optuna.db")

        storage_uri = st.text_input(
            "Storage URI",
            value=default_uri,
            help=(
                "Optuna storage URI or directory path.\n"
                "Examples:\n"
                "- sqlite:///./outputs/tune/optuna.db (SQLite database)\n"
                "- ./outputs/tune (directory - will scan for .db files)\n"
                "- postgresql://user:pass@localhost/optuna (PostgreSQL)"
            ),
        )

    with col2:
        st.write("")  # Spacer
        st.write("")  # Spacer
        if st.button("🔌 Connect", type="primary"):
            with st.spinner(f"Connecting to Optuna storage at {storage_uri}..."):
                try:
                    # Normalize storage URI
                    normalized_uri = storage_uri

                    # If it looks like a directory path, try to find a database file
                    if not storage_uri.startswith(("sqlite://", "postgresql://", "mysql://")):
                        path = Path(storage_uri)
                        if path.is_dir():
                            # Look for .db files in directory
                            db_files = list(path.glob("*.db"))
                            if db_files:
                                normalized_uri = f"sqlite:///{db_files[0]}"
                                st.info(f"Found database: {db_files[0].name}")
                            else:
                                st.warning(f"No .db files found in {storage_uri}")
                                normalized_uri = f"sqlite:///{path}/optuna.db"
                        elif path.suffix == ".db" or not path.exists():
                            # Treat as SQLite database path
                            normalized_uri = f"sqlite:///{path}"
                        else:
                            st.error(f"Invalid storage path: {storage_uri}")
                            st.stop()

                    # Test connection by listing studies
                    study_names = _list_studies(normalized_uri)

                    st.session_state.optuna_storage_uri = normalized_uri
                    st.success(f"✅ Connected to Optuna storage (found {len(study_names)} studies)")
                except Exception as exc:
                    st.error(f"❌ Failed to connect to Optuna storage: {exc}")
                    st.info(
                        "💡 This may happen if:\n"
                        "- The storage URI is invalid or inaccessible\n"
                        "- The database file doesn't exist yet\n"
                        "- The database format is incompatible\n"
                        "- Permissions are insufficient to access the file\n\n"
                        "Try running a tuning command first to create the database:\n"
                        "```bash\n"
                        "python -m mammography.cli tune --dataset mamografias --n-trials 10 --storage sqlite:///optuna.db\n"
                        "```"
                    )
                    st.session_state.optuna_storage_uri = None

    # Display studies if connected
    if st.session_state.optuna_storage_uri is not None:
        st.markdown("---")
        _display_studies_overview(st.session_state.optuna_storage_uri)
    else:
        st.info("Click 'Connect' to view studies from the Optuna storage.")

    # Help section
    with st.expander("ℹ️ Help & Documentation"):
        st.markdown("""
        ### How to Use This Page

        1. **Configure Storage URI**: Enter the path to your Optuna database or directory
        2. **Connect**: Click the Connect button to establish connection
        3. **Browse Studies**: View all optimization studies and select one to explore
        4. **Analyze Results**: Examine optimization history, parameter importance, and distributions
        5. **Export Plots**: Download visualizations as PNG, PDF, or SVG for reports/papers

        ### Optuna Storage URI

        Optuna supports multiple storage backends:

        - **SQLite (Local)**: `sqlite:///./outputs/tune/optuna.db` - stores data in a local database file
        - **Directory Path**: `./outputs/tune` - automatically finds .db files in the directory
        - **PostgreSQL**: `postgresql://user:pass@localhost/optuna` - remote database
        - **MySQL**: `mysql://user:pass@localhost/optuna` - remote database

        ### Running Hyperparameter Tuning

        To create a new optimization study:

        ```bash
        # Basic tuning with 50 trials
        python -m mammography.cli tune \\
            --dataset mamografias \\
            --n-trials 50 \\
            --storage sqlite:///optuna.db

        # Tuning with custom search space
        python -m mammography.cli tune \\
            --dataset mamografias \\
            --n-trials 100 \\
            --tune-config configs/tune.yaml \\
            --storage sqlite:///optuna.db \\
            --study-name "resnet50_density_v1"

        # Quick tuning for testing (fewer epochs per trial)
        python -m mammography.cli tune \\
            --dataset mamografias \\
            --n-trials 20 \\
            --epochs 5 \\
            --subset 100 \\
            --storage sqlite:///optuna.db
        ```

        ### Understanding the Visualizations

        **Optimization History:**
        - Shows how the objective value (validation accuracy) improved over trials
        - Blue points: individual trial results
        - Red line: best value achieved so far
        - Look for convergence or plateaus

        **Parameter Importance:**
        - Ranks hyperparameters by their impact on performance
        - Higher bars = more important parameters
        - Focus tuning efforts on the most important parameters
        - Requires multiple completed trials (minimum 2)

        **Parallel Coordinates:**
        - Multi-dimensional view of all hyperparameters and objective value
        - Each line = one trial
        - Color intensity = performance (red/orange = better, blue = worse)
        - Helps identify patterns across multiple parameters

        **Parameter Distributions (Slice Plots):**
        - Shows how each parameter value affects the objective
        - Helps identify optimal parameter ranges
        - Look for clear trends or sweet spots

        ### Hyperparameter Search Space

        The search space is defined in `configs/tune.yaml` and typically includes:

        - **Learning Rate (lr)**: Head/classifier learning rate
        - **Backbone LR**: Learning rate for pretrained backbone
        - **Batch Size**: Number of samples per batch
        - **Weight Decay**: L2 regularization strength
        - **Warmup Epochs**: Learning rate warmup duration
        - **Early Stop Patience**: Epochs to wait before early stopping

        ### Pruning and Efficiency

        Optuna uses **MedianPruner** to terminate unpromising trials early:

        - Trials that perform worse than the median are pruned
        - Saves computation time by focusing on promising regions
        - Pruner only activates after warmup steps and startup trials
        - Pruned trials appear in the statistics but don't complete

        ### Best Practices

        1. **Start Small**: Run 10-20 trials first to understand the search space
        2. **Increase Trials**: Scale up to 50-100 trials for thorough exploration
        3. **Monitor Progress**: Use this dashboard to track optimization in real-time
        4. **Iterate**: Refine search space based on parameter importance
        5. **Validate**: Re-train with best parameters on full dataset to confirm results

        ### Troubleshooting

        - **No Studies Found**: Run a tuning command to create studies first
        - **Connection Failed**: Verify the storage URI is correct and accessible
        - **Empty Plots**: Ensure you have completed trials (not just pruned/failed)
        - **Import Errors**: Install required packages: `pip install optuna plotly kaleido`

        ### References

        - [Optuna Documentation](https://optuna.readthedocs.io/)
        - [Optuna Visualization](https://optuna.readthedocs.io/en/stable/reference/visualization/index.html)
        - [Hyperparameter Optimization Best Practices](https://optuna.readthedocs.io/en/stable/tutorial/index.html)
        """)


if __name__ == "__main__":
    main()
