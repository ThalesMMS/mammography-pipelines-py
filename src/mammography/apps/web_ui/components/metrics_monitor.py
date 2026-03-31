#
# metrics_monitor.py
# mammography-pipelines
#
# Metrics monitor component for displaying real-time training metrics in the web UI.
#
# Thales Matheus Mendonça Santos - February 2026
#
"""Metrics monitor component for real-time training metrics display."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    import streamlit as st
except Exception as exc:  # pragma: no cover - optional UI dependency
    st = None
    _STREAMLIT_IMPORT_ERROR = exc
else:
    _STREAMLIT_IMPORT_ERROR = None


LOGGER = logging.getLogger("mammography")


def _require_streamlit() -> None:
    """Raise ImportError if Streamlit is not available."""
    if st is None:
        raise ImportError(
            "Streamlit is required to run the web UI dashboard."
        ) from _STREAMLIT_IMPORT_ERROR


class MetricsMonitor:
    """Component for monitoring and displaying real-time training metrics.

    This component provides functionality to:
    - Track training metrics (loss, accuracy, learning rate, etc.)
    - Display current metrics in real-time
    - Show metrics history as time series charts
    - Calculate and display training statistics
    - Monitor system resources (memory, GPU usage)
    - Provide progress indicators for training

    DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
    It must NOT be used for clinical or medical diagnostic purposes.
    No medical decision should be based on these results.

    Example:
        >>> monitor = MetricsMonitor()
        >>> monitor.update_metrics({
        ...     "epoch": 1,
        ...     "train_loss": 0.45,
        ...     "val_loss": 0.52,
        ...     "train_acc": 0.82,
        ...     "val_acc": 0.78,
        ... })
        >>> monitor.render()
    """

    def __init__(self) -> None:
        """Initialize the metrics monitor component."""
        _require_streamlit()
        self.metrics: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []
        self.start_time = time.time()
        self._last_update_time = time.time()

    def update_metrics(self, new_metrics: Dict[str, Any]) -> None:
        """Update metrics with new values.

        Args:
            new_metrics: Dictionary of metric name -> value pairs
        """
        # Update current metrics
        self.metrics.update(new_metrics)
        self.metrics["timestamp"] = datetime.now()
        self.metrics["elapsed_time"] = time.time() - self.start_time
        self._last_update_time = time.time()

        # Append to history
        self.history.append(self.metrics.copy())

        # Keep only last 1000 entries to prevent memory issues
        if len(self.history) > 1000:
            self.history = self.history[-1000:]

        LOGGER.debug("Metrics updated: %s", new_metrics)

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics.

        Returns:
            Dictionary of current metric values
        """
        return self.metrics.copy()

    def get_metrics_history(self) -> pd.DataFrame:
        """Get metrics history as DataFrame.

        Returns:
            DataFrame with metrics history, or empty DataFrame if no history
        """
        if not self.history:
            return pd.DataFrame()
        return pd.DataFrame(self.history)

    def get_elapsed_time(self) -> float:
        """Get elapsed time since monitoring started.

        Returns:
            Elapsed time in seconds
        """
        return time.time() - self.start_time

    def format_time(self, seconds: float) -> str:
        """Format seconds as human-readable time string.

        Args:
            seconds: Time in seconds

        Returns:
            Formatted time string (HH:MM:SS)
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def render(
        self,
        show_charts: bool = True,
        show_system_metrics: bool = True,
        chart_height: int = 300,
    ) -> None:
        """Render the metrics monitor display.

        Args:
            show_charts: Whether to show metric history charts
            show_system_metrics: Whether to show system resource metrics
            chart_height: Height of charts in pixels
        """
        _require_streamlit()

        if not self.metrics:
            st.info("⏳ Waiting for training metrics...")
            return

        # Display current metrics in columns
        st.subheader("📊 Current Metrics")

        # Create metric columns
        cols = st.columns(4)

        # Training metrics
        with cols[0]:
            if "epoch" in self.metrics:
                st.metric(
                    "Epoch",
                    f"{self.metrics['epoch']}",
                    delta=None,
                )

        with cols[1]:
            if "train_loss" in self.metrics:
                # Calculate delta from previous epoch if available
                delta = None
                if len(self.history) > 1:
                    prev_loss = self.history[-2].get("train_loss")
                    if prev_loss is not None:
                        delta = self.metrics["train_loss"] - prev_loss

                st.metric(
                    "Train Loss",
                    f"{self.metrics['train_loss']:.4f}",
                    delta=f"{delta:.4f}" if delta is not None else None,
                    delta_color="inverse",  # Lower is better
                )

        with cols[2]:
            if "val_loss" in self.metrics:
                delta = None
                if len(self.history) > 1:
                    prev_loss = self.history[-2].get("val_loss")
                    if prev_loss is not None:
                        delta = self.metrics["val_loss"] - prev_loss

                st.metric(
                    "Val Loss",
                    f"{self.metrics['val_loss']:.4f}",
                    delta=f"{delta:.4f}" if delta is not None else None,
                    delta_color="inverse",
                )

        with cols[3]:
            if "train_acc" in self.metrics:
                delta = None
                if len(self.history) > 1:
                    prev_acc = self.history[-2].get("train_acc")
                    if prev_acc is not None:
                        delta = self.metrics["train_acc"] - prev_acc

                st.metric(
                    "Train Acc",
                    f"{self.metrics['train_acc']:.2%}",
                    delta=f"{delta:.2%}" if delta is not None else None,
                )

        # Additional metrics row
        cols2 = st.columns(4)

        with cols2[0]:
            if "val_acc" in self.metrics:
                delta = None
                if len(self.history) > 1:
                    prev_acc = self.history[-2].get("val_acc")
                    if prev_acc is not None:
                        delta = self.metrics["val_acc"] - prev_acc

                st.metric(
                    "Val Acc",
                    f"{self.metrics['val_acc']:.2%}",
                    delta=f"{delta:.2%}" if delta is not None else None,
                )

        with cols2[1]:
            if "learning_rate" in self.metrics:
                st.metric(
                    "Learning Rate",
                    f"{self.metrics['learning_rate']:.2e}",
                )

        with cols2[2]:
            # Elapsed time
            elapsed = self.get_elapsed_time()
            st.metric(
                "Elapsed Time",
                self.format_time(elapsed),
            )

        with cols2[3]:
            # Estimated time remaining (if total epochs available)
            if "epoch" in self.metrics and "total_epochs" in self.metrics:
                current_epoch = self.metrics["epoch"]
                total_epochs = self.metrics["total_epochs"]
                if current_epoch > 0:
                    time_per_epoch = elapsed / current_epoch
                    remaining_epochs = total_epochs - current_epoch
                    eta_seconds = time_per_epoch * remaining_epochs
                    st.metric(
                        "ETA",
                        self.format_time(eta_seconds),
                    )

        # System metrics
        if show_system_metrics:
            st.divider()
            st.subheader("🖥️ System Metrics")

            sys_cols = st.columns(3)

            with sys_cols[0]:
                if "memory_usage_mb" in self.metrics:
                    st.metric(
                        "Memory Usage",
                        f"{self.metrics['memory_usage_mb']:.1f} MB",
                    )

            with sys_cols[1]:
                if "gpu_memory_mb" in self.metrics:
                    st.metric(
                        "GPU Memory",
                        f"{self.metrics['gpu_memory_mb']:.1f} MB",
                    )

            with sys_cols[2]:
                if "gpu_utilization" in self.metrics:
                    st.metric(
                        "GPU Utilization",
                        f"{self.metrics['gpu_utilization']:.1f}%",
                    )

        # Metrics history charts
        if show_charts and len(self.history) > 1:
            st.divider()
            st.subheader("📈 Training History")

            df = self.get_metrics_history()

            # Loss chart
            if "train_loss" in df.columns or "val_loss" in df.columns:
                st.markdown("**Loss over Time**")
                loss_data = {}
                if "train_loss" in df.columns:
                    loss_data["Train Loss"] = df["train_loss"]
                if "val_loss" in df.columns:
                    loss_data["Val Loss"] = df["val_loss"]

                if loss_data:
                    loss_df = pd.DataFrame(loss_data)
                    st.line_chart(loss_df, height=chart_height)

            # Accuracy chart
            if "train_acc" in df.columns or "val_acc" in df.columns:
                st.markdown("**Accuracy over Time**")
                acc_data = {}
                if "train_acc" in df.columns:
                    acc_data["Train Accuracy"] = df["train_acc"]
                if "val_acc" in df.columns:
                    acc_data["Val Accuracy"] = df["val_acc"]

                if acc_data:
                    acc_df = pd.DataFrame(acc_data)
                    st.line_chart(acc_df, height=chart_height)

            # Learning rate chart
            if "learning_rate" in df.columns:
                st.markdown("**Learning Rate over Time**")
                st.line_chart(df["learning_rate"], height=chart_height)

    def render_progress_bar(
        self,
        current: Optional[int] = None,
        total: Optional[int] = None,
    ) -> None:
        """Render a progress bar for training.

        Args:
            current: Current step/epoch (uses metrics if None)
            total: Total steps/epochs (uses metrics if None)
        """
        _require_streamlit()

        # Use provided values or fall back to metrics
        if current is None and "epoch" in self.metrics:
            current = self.metrics["epoch"]
        if total is None and "total_epochs" in self.metrics:
            total = self.metrics["total_epochs"]

        if current is not None and total is not None and total > 0:
            progress = current / total
            st.progress(
                progress,
                text=f"Epoch {current}/{total} ({progress:.1%})",
            )
        else:
            st.info("⏳ Training in progress...")

    def clear_history(self) -> None:
        """Clear metrics history to free memory."""
        self.history.clear()
        LOGGER.info("Metrics history cleared")

    def reset(self) -> None:
        """Reset the monitor to initial state."""
        self.metrics.clear()
        self.history.clear()
        self.start_time = time.time()
        self._last_update_time = time.time()
        LOGGER.info("Metrics monitor reset")
