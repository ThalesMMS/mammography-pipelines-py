"""Run details section for the Streamlit Experiments page."""

from __future__ import annotations

import io
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mammography.apps.web_ui.components.export_buttons import export_plot_buttons
from mammography.apps.web_ui.components.results_visualizer import (
    plot_confusion_matrix,
    plot_roc_curves,
)
from mammography.apps.web_ui.pages.experiments.formatters import (
    format_duration as _format_duration,
    format_timestamp as _format_timestamp,
)

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


def display_run_details(client: Any, run: Any) -> None:
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
                    export_plot_buttons(fig, f"metrics_history_{run.info.run_id[:8]}")
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
        artifacts_list = client.list_artifacts(run.info.run_id)

        if artifacts_list:
            artifact_data = [
                {
                    "Path": artifact.path,
                    "Size (bytes)": artifact.file_size if artifact.file_size else "N/A",
                    "Is Directory": "Yes" if artifact.is_dir else "No",
                }
                for artifact in artifacts_list
            ]

            artifacts_df = pd.DataFrame(artifact_data)
            st.dataframe(artifacts_df, width="stretch", hide_index=True)

            # Download artifact link
            if run.info.artifact_uri:
                st.info(f"Artifact Location: {run.info.artifact_uri}")
        else:
            st.info("No artifacts logged for this run.")
    except Exception as exc:
        st.warning(f"Could not list artifacts: {exc}")
        artifacts_list = []

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

        cm_artifact_path = None

        for artifact in artifacts_list:
            if any(pattern in artifact.path for pattern in cm_artifact_patterns):
                cm_artifact_path = artifact.path
                break

        y_true = None
        y_pred = None
        class_names = None

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
                        class_names_arr = data.get("class_names", None)
                        if class_names_arr is not None:
                            class_names = class_names_arr.tolist()
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
        if confusion_matrix_found and y_true is not None and y_pred is not None:
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
            fig = plot_confusion_matrix(
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
                            fig_roc = plot_roc_curves(
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