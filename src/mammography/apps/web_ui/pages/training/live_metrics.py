"""Live metrics section for the Streamlit Training page."""

from __future__ import annotations

import time
from datetime import datetime

from mammography.apps.web_ui.pages.training.mlflow_polling import poll_mlflow_metrics

try:
    import streamlit as st
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception as exc:  # pragma: no cover - optional UI dependency
    st = None
    go = None
    make_subplots = None
    _STREAMLIT_IMPORT_ERROR = exc
else:
    _STREAMLIT_IMPORT_ERROR = None


def _require_streamlit() -> None:
    if st is None:
        raise ImportError("Streamlit is required to run the web UI dashboard.") from _STREAMLIT_IMPORT_ERROR


def _build_training_figure(
    epochs: list,
    train_losses: list,
    val_losses: list,
    train_accs: list,
    val_accs: list,
    learning_rates: list,
    samples_processed: list,
):
    """Build the 2x2 training dashboard subplot figure."""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Training & Validation Loss",
            "Training & Validation Accuracy",
            "Learning Rate Schedule",
            "Samples Processed",
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
        ],
    )

    traces = [
        (epochs, train_losses, "Train Loss", "blue", 1, 1),
        (epochs, val_losses, "Val Loss", "red", 1, 1),
        (epochs, train_accs, "Train Acc", "green", 1, 2),
        (epochs, val_accs, "Val Acc", "orange", 1, 2),
        (epochs, learning_rates, "Learning Rate", "purple", 2, 1),
        (epochs, samples_processed, "Samples", "brown", 2, 2),
    ]
    for x, y, name, color, row, col in traces:
        fig.add_trace(go.Scatter(x=x, y=y, name=name, line=dict(color=color)), row=row, col=col)

    fig.update_layout(height=600, showlegend=True, template="plotly_white")
    for row, col in [(1, 1), (1, 2), (2, 1), (2, 2)]:
        fig.update_xaxes(title_text="Epoch", row=row, col=col)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", row=1, col=2)
    fig.update_yaxes(title_text="LR", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=2)

    return fig


def render_live_metrics_section() -> None:
    """Render live training metrics monitoring section."""
    st.subheader("📊 Live Training Metrics")

    # Poll MLflow metrics if training is running
    if st.session_state.training_status == "running":
        poll_mlflow_metrics(st.session_state)

        # Show MLflow tracking status
        if st.session_state.mlflow_client is not None and st.session_state.active_run_id is not None:
            st.info(
                f"📈 MLflow tracking active - Run ID: `{st.session_state.active_run_id[:8]}...` "
                f"(polling every 2 seconds)"
            )
        elif st.session_state.training_config.get("tracker") == "mlflow":
            st.warning("⏳ Waiting for MLflow run to start...")

    if st.session_state.training_status == "idle":
        st.info("💡 Metrics will appear here when training starts. Configure your training parameters above and launch training to monitor real-time metrics.")

        # Show placeholder charts
        st.markdown("**Preview: Sample Metrics Dashboard**")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label="Current Epoch", value="0 / 0", delta=None)
        with col2:
            st.metric(label="Training Loss", value="0.000", delta=None)
        with col3:
            st.metric(label="Validation Accuracy", value="0.00%", delta=None)
        with col4:
            st.metric(label="Learning Rate", value="0.0e+00", delta=None)

        if go is not None and make_subplots is not None:
            zeros = [0]
            fig = _build_training_figure(zeros, zeros, zeros, zeros, zeros, zeros, zeros)
            st.plotly_chart(fig, width="stretch")
        else:
            st.warning("📊 Install plotly to see interactive metrics charts: `pip install plotly`")

    else:
        # Display current metrics
        metrics = st.session_state.training_metrics

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            epoch_val = metrics.get("epoch", 0)
            if st.session_state.training_config:
                total_epochs = st.session_state.training_config.get("epochs", 100)
                epoch_display = f"{epoch_val} / {total_epochs}"
            else:
                epoch_display = str(epoch_val)
            st.metric(label="Current Epoch", value=epoch_display, delta=None)

        with col2:
            st.metric(label="Training Loss", value=f"{metrics.get('train_loss', 0):.4f}", delta=None)
        with col3:
            st.metric(label="Validation Accuracy", value=f"{metrics.get('val_acc', 0) * 100:.2f}%", delta=None)
        with col4:
            st.metric(label="Learning Rate", value=f"{metrics.get('learning_rate', 0):.2e}", delta=None)

        # Display metrics charts if history is available
        if st.session_state.training_metrics_history and go is not None and make_subplots is not None:
            history = st.session_state.training_metrics_history

            fig = _build_training_figure(
                epochs=[m.get("epoch", 0) for m in history],
                train_losses=[m.get("train_loss", 0) for m in history],
                val_losses=[m.get("val_loss", 0) for m in history],
                train_accs=[m.get("train_acc", 0) for m in history],
                val_accs=[m.get("val_acc", 0) for m in history],
                learning_rates=[m.get("learning_rate", 0) for m in history],
                samples_processed=[m.get("samples_processed", 0) for m in history],
            )
            st.plotly_chart(fig, width="stretch")
            st.caption(f"📅 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.info("📊 Metrics history will be displayed here as training progresses.")

        # Auto-refresh when training is running
        if st.session_state.training_status == "running":
            st.caption("🔄 Page auto-refreshing every 3 seconds to update metrics...")
            time.sleep(3)
            st.rerun()