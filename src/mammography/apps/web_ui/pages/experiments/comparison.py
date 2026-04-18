"""Run comparison section for the Streamlit Experiments page."""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from mammography.apps.web_ui.components.export_buttons import export_plot_buttons
from mammography.apps.web_ui.pages.experiments.formatters import format_duration as _format_duration

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


def _render_metric_curves_tab(
    client: Any,
    runs: Any,
    available_metrics: list,
    metric_type: str,
    multiselect_key: str,
    export_filename: str,
) -> None:
    """Render a metric-curves comparison tab (shared by accuracy and loss tabs)."""
    import plotly.graph_objects as go

    label = metric_type.title()
    if available_metrics:
        st.subheader(f"{label} Comparison")
        selected_metrics = st.multiselect(
            f"Select {metric_type} metrics",
            options=available_metrics,
            default=available_metrics[:min(2, len(available_metrics))],
            key=multiselect_key,
        )
        if selected_metrics:
            fig = go.Figure()
            for run in runs:
                run_name = run.info.run_name or run.info.run_id[:8]
                for metric_key in selected_metrics:
                    if metric_key in run.data.metrics:
                        try:
                            history = client.get_metric_history(run.info.run_id, metric_key)
                            if history:
                                fig.add_trace(go.Scatter(
                                    x=[h.step for h in history],
                                    y=[h.value for h in history],
                                    mode="lines+markers",
                                    name=f"{run_name} - {metric_key}",
                                    hovertemplate=(
                                        f"<b>%{{fullData.name}}</b><br>Step: %{{x}}<br>"
                                        f"{label}: %{{y:.4f}}<extra></extra>"
                                    ),
                                ))
                        except Exception as exc:
                            st.warning(f"Could not fetch {metric_key} for {run_name}: {exc}")

            # Legend anchor differs: accuracy anchors bottom, loss anchors top
            legend_yanchor = "bottom" if metric_type == "accuracy" else "top"
            legend_y = 0.01 if metric_type == "accuracy" else 0.99

            fig.update_layout(
                title=f"{label} Comparison Across Runs",
                xaxis_title="Step/Epoch",
                yaxis_title=label,
                hovermode="x unified",
                height=500,
                legend=dict(
                    orientation="v",
                    yanchor=legend_yanchor,
                    y=legend_y,
                    xanchor="right",
                    x=0.99,
                ),
            )
            st.plotly_chart(fig, width="stretch")
            export_plot_buttons(fig, export_filename)
        else:
            st.info(f"Select at least one {metric_type} metric to visualize.")
    else:
        st.info(f"No {metric_type} metrics found in the selected runs.")


def display_run_comparison(client: Any, runs: Any) -> None:
    """Display comparison charts for multiple runs."""
    st.header(f"Comparing {len(runs)} Runs")

    try:
        import plotly.graph_objects as go
    except ImportError:
        st.error("Plotly is required for comparison charts. Install with: pip install plotly")
        return

    # Get all available metrics from selected runs
    all_metrics = set()
    for run in runs:
        all_metrics.update(run.data.metrics.keys())

    # Common training/validation metrics
    accuracy_metrics = [m for m in all_metrics if "acc" in m.lower()]
    loss_metrics = [m for m in all_metrics if "loss" in m.lower()]

    # Create tabs for different comparison views
    tab1, tab2, tab3 = st.tabs(["📈 Accuracy Curves", "📉 Loss Curves", "📊 Final Metrics"])

    with tab1:
        _render_metric_curves_tab(
            client, runs, accuracy_metrics, "accuracy", "acc_metrics", "accuracy_comparison"
        )

    with tab2:
        _render_metric_curves_tab(
            client, runs, loss_metrics, "loss", "loss_metrics", "loss_comparison"
        )

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
                    export_plot_buttons(fig, f"final_metrics_{metric}")
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