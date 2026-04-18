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
import types
from typing import Sequence

try:
    import streamlit as st
except Exception as exc:  # pragma: no cover - optional UI dependency
    st = None
    _STREAMLIT_IMPORT_ERROR = exc
else:
    _STREAMLIT_IMPORT_ERROR = None
    if getattr(st, "web", None) is None:
        st.web = types.SimpleNamespace(cli=types.SimpleNamespace())

from mammography.apps.web_ui.utils import ensure_shared_session_state


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
    st.header("Welcome")

    st.markdown("""
    This web interface provides access to the core functionality of the Mammography
    Pipelines project for breast density classification (BI-RADS A–D) using deep learning.

    ### Available Features

    Use the sidebar navigation to access the following pages:

    #### 📁 Dataset Browser
    Explore your mammography datasets with an interactive interface.
    - **View images**: Browse images in grid or table layout with configurable columns
    - **Filter data**: Filter by BI-RADS density class (A, B, C, D) and view position (CC, MLO)
    - **View metadata**: Inspect AccessionNumber, patient ID, acquisition details, and labels
    - **Analyze statistics**: View density class distribution with interactive bar charts
    - **Export data**: Download filtered metadata as CSV for further analysis

    **Usage Example:**
    1. Navigate to Dataset Browser from the sidebar
    2. Configure your dataset path or use auto-detection
    3. Use filters to find specific density classes or view positions
    4. Switch between grid and table views for different perspectives
    5. Export filtered results for offline analysis

    #### 📊 Inference
    Upload mammography images and get instant BI-RADS density predictions.
    - **Upload files**: Support for DICOM (.dcm) and PNG/JPEG images
    - **Model selection**: Choose from available trained model checkpoints
    - **Instant predictions**: Get class probabilities for all BI-RADS categories
    - **View confidence**: See model confidence scores for each prediction

    **Usage Example:**
    1. Navigate to Inference page
    2. Select a trained model checkpoint from the dropdown
    3. Upload a mammography image (DICOM or PNG)
    4. View prediction results with confidence scores
    5. Upload additional images for batch processing

    #### 🔍 Explainability
    Understand model decisions through visual explanations.
    - **GradCAM heatmaps**: Visualize which regions influenced the prediction
    - **Attention visualization**: See where the model focuses during classification
    - **Layer selection**: Choose different model layers for analysis
    - **Overlay options**: View heatmaps overlaid on original images

    **Usage Example:**
    1. Navigate to Explainability page
    2. Load a model checkpoint and upload an image
    3. Generate GradCAM visualization for the prediction
    4. Experiment with different layers to understand feature hierarchies
    5. Download visualizations for presentations or publications

    #### 📈 Experiments
    Track, compare, and analyze training experiments with MLflow integration.
    - **View runs**: Browse all training experiments with sortable metrics
    - **Compare models**: Side-by-side comparison of hyperparameters and results
    - **Visualize results**: Confusion matrices and ROC curves with AUC scores
    - **Download artifacts**: Access model checkpoints and saved predictions
    - **Export reports**: Generate publication-ready report packages (PNG, PDF, SVG)

    **Usage Example:**
    1. Navigate to Experiments page
    2. Review list of all training runs with metrics (accuracy, loss, F1-score)
    3. Select a run to view detailed results and visualizations
    4. Compare multiple runs to identify best performing configurations
    5. Export results as a ZIP package for sharing or publication
    6. Use batch export to generate reports for multiple experiments

    **New Visualization Features:**
    - **Confusion Matrix**: View classification errors with normalization options
    - **ROC Curves**: Multi-class ROC curves with micro/macro-averaged AUC
    - **Report Export**: One-click generation of publication-ready figures

    #### ⚙️ Training
    Configure and launch model training jobs with live monitoring.
    - **Configure hyperparameters**: Set learning rate, batch size, epochs, optimizer
    - **Select architecture**: Choose from ResNet50, EfficientNetB0, or Vision Transformer
    - **Monitor progress**: Real-time metrics updates during training
    - **Track resources**: View memory usage and GPU utilization
    - **Auto-logging**: Automatic experiment tracking with MLflow

    **Usage Example:**
    1. Navigate to Training page
    2. Configure dataset path and training parameters
    3. Select model architecture and hyperparameters
    4. Click "Start Training" to launch the job
    5. Monitor live metrics: loss, accuracy, learning rate
    6. Track training progress with real-time charts and metrics
    7. View system resource usage (memory, GPU)
    8. After training completes, view results in Experiments page

    **Live Metrics Features:**
    - **Real-time updates**: Metrics refresh every 3 seconds during training
    - **Interactive charts**: Loss curves, accuracy trends, learning rate schedule
    - **Progress tracking**: Current epoch, samples processed, time remaining
    - **System monitoring**: Memory usage, GPU utilization (if available)

    ### Getting Started

    #### Quick Start Workflow

    **Option 1: Explore Existing Datasets**
    1. Go to **Dataset Browser** to explore your mammography data
    2. Use filters to understand class distribution and data quality
    3. Export metadata for statistical analysis

    **Option 2: Run Inference on New Images**
    1. Ensure you have a trained model checkpoint (or train a new one)
    2. Go to **Inference** page
    3. Upload a mammography image and get instant predictions
    4. Use **Explainability** page to understand the prediction

    **Option 3: Train a New Model**
    1. Prepare your dataset with proper directory structure
    2. Go to **Training** page
    3. Configure hyperparameters and dataset path
    4. Launch training and monitor live metrics
    5. Review results in **Experiments** page
    6. Export publication-ready reports

    **Option 4: Analyze Experiment Results**
    1. Go to **Experiments** page
    2. Browse previous training runs
    3. View confusion matrices and ROC curves
    4. Compare multiple experiments side-by-side
    5. Export report packages for sharing

    ### Tips and Best Practices

    **For Dataset Exploration:**
    - Use auto-detection for quick dataset setup
    - Filter by density class to check class balance
    - Export filtered metadata for statistical analysis
    - Check for missing labels or corrupted images

    **For Training:**
    - Start with a small subset to verify configuration
    - Monitor live metrics to detect overfitting early
    - Use learning rate warmup for stable training
    - Enable data augmentation to improve generalization
    - Track experiments with descriptive names

    **For Inference:**
    - Use models trained on similar data distribution
    - Check prediction confidence scores
    - Generate GradCAM visualizations for low-confidence predictions
    - Validate results with domain experts before clinical use

    **For Reporting:**
    - Export reports in SVG format for publications
    - Include confusion matrices to show per-class performance
    - Use ROC curves to demonstrate discrimination ability
    - Batch export for comparing multiple model versions

    ### Troubleshooting

    **"Failed to load dataset"**
    - Verify dataset path points to correct directory
    - Check that metadata files (classificacao.csv or featureS.txt) exist
    - Try using auto-detection feature
    - Ensure image files are accessible and not corrupted

    **"Model checkpoint not found"**
    - Check that model file exists at specified path
    - Verify checkpoint was saved correctly during training
    - Look in MLflow artifacts directory for checkpoints

    **"Live metrics not updating"**
    - Ensure training is actually running (check process status)
    - Verify MLflow tracking is enabled
    - Check that run ID was captured from training output
    - Try refreshing the page

    **"Out of memory during training"**
    - Reduce batch size in training configuration
    - Use gradient accumulation for effective larger batches
    - Enable mixed precision training (FP16)
    - Close other GPU-intensive applications

    ### Technical Details

    This dashboard is built with Streamlit and integrates with the existing command-line
    tools in the Mammography Pipelines project. It provides a user-friendly interface
    for researchers and clinicians who prefer visual tools over terminal commands.

    **Architecture:**
    - **Frontend**: Streamlit for interactive UI components
    - **Backend**: PyTorch for deep learning, MLflow for experiment tracking
    - **Visualization**: Plotly, Matplotlib, and Seaborn for charts
    - **Data Processing**: DICOM support with lazy loading and caching

    **Data Flow:**
    1. Dataset Browser → Explore raw data
    2. Training → Train models with live monitoring
    3. Experiments → Analyze results and visualizations
    4. Inference → Deploy models for predictions
    5. Explainability → Understand model decisions

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
