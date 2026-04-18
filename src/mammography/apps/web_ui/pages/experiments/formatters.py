"""Formatting helpers for the Experiments page."""

from __future__ import annotations

from datetime import datetime


def format_timestamp(timestamp_ms: int | None) -> str:
    """Format a Unix timestamp in milliseconds."""
    if timestamp_ms is None:
        return "N/A"
    try:
        dt = datetime.fromtimestamp(timestamp_ms / 1000.0)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "Invalid timestamp"


def format_duration(start_ms: int | None, end_ms: int | None) -> str:
    """Format elapsed time between start and end timestamps."""
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
        if minutes > 0:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"
    except Exception:
        return "N/A"
