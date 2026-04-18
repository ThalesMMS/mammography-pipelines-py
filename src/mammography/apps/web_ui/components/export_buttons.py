"""Reusable export buttons for Plotly figures in the Streamlit UI."""

from __future__ import annotations

from typing import Any

try:
    import streamlit as st
except Exception as exc:  # pragma: no cover - optional UI dependency
    st = None
    _STREAMLIT_IMPORT_ERROR = exc
else:
    _STREAMLIT_IMPORT_ERROR = None


def _require_streamlit() -> None:
    if st is None:
        raise ImportError(
            "Streamlit is required to render export buttons."
        ) from _STREAMLIT_IMPORT_ERROR


def export_plot_buttons(fig: Any, filename_prefix: str) -> None:
    """Display PNG, PDF, and SVG download buttons for a Plotly figure."""
    _require_streamlit()
    st.markdown("**Export Plot:**")

    col1, col2, col3 = st.columns(3)

    with col1:
        try:
            img_bytes = fig.to_image(format="png", width=1200, height=800, scale=2)
            st.download_button(
                label="📥 Download PNG",
                data=img_bytes,
                file_name=f"{filename_prefix}.png",
                mime="image/png",
                help="Download high-resolution PNG (1200x800, 2x scale)",
            )
        except Exception:
            st.warning("PNG export requires kaleido: pip install kaleido")

    with col2:
        try:
            pdf_bytes = fig.to_image(format="pdf", width=1200, height=800)
            st.download_button(
                label="📥 Download PDF",
                data=pdf_bytes,
                file_name=f"{filename_prefix}.pdf",
                mime="application/pdf",
                help="Download vector PDF for publication",
            )
        except Exception:
            st.warning("PDF export requires kaleido: pip install kaleido")

    with col3:
        try:
            svg_bytes = fig.to_image(format="svg", width=1200, height=800)
            st.download_button(
                label="📥 Download SVG",
                data=svg_bytes,
                file_name=f"{filename_prefix}.svg",
                mime="image/svg+xml",
                help="Download vector SVG for editing",
            )
        except Exception:
            st.warning("SVG export requires kaleido: pip install kaleido")
