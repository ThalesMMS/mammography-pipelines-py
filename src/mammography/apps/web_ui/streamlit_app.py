#
# streamlit_app.py
# mammography-pipelines
#
# Main landing page for the web-based UI dashboard with navigation to inference,
# explainability, experiments, and training configuration pages.
#
# Thales Matheus Mendonca Santos - February 2026
#
"""Streamlit web UI dashboard for model inference, visualization, and training configuration."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

try:
    import streamlit as st
except Exception as exc:  # pragma: no cover - optional UI dependency
    st = None
    _STREAMLIT_IMPORT_ERROR = exc
else:
    _STREAMLIT_IMPORT_ERROR = None

from .utils import ensure_shared_session_state


def _require_streamlit() -> None:
    """Raise ImportError if Streamlit is not available."""
    if st is None:
        raise ImportError(
            "Streamlit is required to run the web UI dashboard."
        ) from _STREAMLIT_IMPORT_ERROR


def main() -> None:
    """Render the main landing page of the web UI dashboard."""
    _require_streamlit()

    st.set_page_config(
        page_title="Mammography Pipelines Dashboard",
        page_icon="🩻",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize shared session state for cross-page data persistence
    try:
        ensure_shared_session_state()
    except Exception as exc:
        st.error(f"❌ Failed to initialize session state: {exc}")
        st.stop()

    st.title("🩻 Mammography Pipelines Dashboard")

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

    st.header("Welcome")

    st.markdown("""
    This web interface provides access to the core functionality of the Mammography
    Pipelines project for breast density classification (BI-RADS A–D) using deep learning.

    ### Available Features

    Use the sidebar navigation to access the following pages:

    - **📊 Inference**: Upload DICOM or PNG mammography images and get BI-RADS density
      predictions from trained models

    - **🔍 Explainability**: Visualize GradCAM heatmaps to understand which regions
      of the image influenced the model's predictions

    - **📈 Experiments**: View and compare training experiment results tracked with MLflow,
      including metrics and hyperparameters

    - **⚙️ Training**: Configure and launch new training jobs with custom hyperparameters

    ### Getting Started

    1. Select a page from the sidebar navigation
    2. Follow the instructions on each page
    3. For inference, you'll need a trained model checkpoint
    4. For training, ensure your dataset is configured correctly

    ### Technical Details

    This dashboard is built with Streamlit and integrates with the existing command-line
    tools in the Mammography Pipelines project. It provides a user-friendly interface
    for researchers and clinicians who prefer visual tools over terminal commands.

    For more information about the project, see the [repository documentation](https://github.com/yourusername/mammography-pipelines).
    """)

    st.header("System Status")

    col1, col2, col3 = st.columns(3)

    with col1:
        try:
            st.metric("Python Version", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        except Exception:
            st.metric("Python Version", "Unknown")

    with col2:
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            device_name = "GPU (CUDA)" if cuda_available else "CPU"
            st.metric("Compute Device", device_name)
            if not cuda_available:
                st.caption("⚠️ GPU not detected, inference will be slower")
        except ImportError:
            st.metric("Compute Device", "❌ PyTorch Missing")
            st.caption("Install PyTorch to enable model features")
        except Exception as exc:
            st.metric("Compute Device", "❌ Error")
            st.caption(f"Failed to detect device: {exc}")

    with col3:
        try:
            import mlflow
            st.metric("MLflow Available", "✅ Yes")
        except ImportError:
            st.metric("MLflow Available", "❌ No")
            st.caption("Install MLflow to track experiments")

    st.markdown("""
    ---
    **Need Help?**

    - Run `mammography --help` in your terminal for CLI documentation
    - Check the project README for setup instructions
    - Report issues on GitHub
    """)


def run(argv: Sequence[str] | None = None) -> int:
    """
    Launch the Streamlit web UI dashboard.

    Args:
        argv: Command-line arguments (unused, for CLI compatibility)

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    _require_streamlit()
    script_path = Path(__file__).resolve()
    args = list(argv) if argv else []

    try:
        from streamlit.web import cli as stcli
    except Exception:
        try:
            from streamlit.web import bootstrap
        except Exception as exc:  # pragma: no cover - optional UI dependency
            raise ImportError(
                "Streamlit CLI is required to launch the web UI dashboard."
            ) from exc
        try:
            bootstrap.run(str(script_path), "", args, {})
        except SystemExit as exc:
            return int(exc.code) if exc.code else 0
        return 0

    saved_argv = sys.argv[:]
    sys.argv = ["streamlit", "run", str(script_path), *args]
    try:
        stcli.main()
    except SystemExit as exc:
        return int(exc.code) if exc.code else 0
    finally:
        sys.argv = saved_argv
    return 0


if __name__ == "__main__":
    main()
