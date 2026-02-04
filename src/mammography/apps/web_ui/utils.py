#
# utils.py
# mammography-pipelines
#
# Utility functions for the web-based UI dashboard.
#
# Thales Matheus Mendonça Santos - November 2025
#
"""Utility functions for web UI operations and data handling."""

from __future__ import annotations

try:
    import streamlit as st
except Exception as exc:  # pragma: no cover - optional UI dependency
    st = None
    _STREAMLIT_IMPORT_ERROR = exc
else:
    _STREAMLIT_IMPORT_ERROR = None


def ensure_shared_session_state() -> None:
    """Initialize shared session state variables across all pages.

    This function should be called at the start of every page to ensure
    that shared state variables are initialized. This allows data like
    uploaded images, loaded models, and inference results to persist
    across page navigation.

    Shared state variables:
    - shared_uploaded_image: PIL Image object of the current image
    - shared_uploaded_image_path: Path to the uploaded image file
    - shared_uploaded_image_name: Name of the uploaded image file
    - shared_model: Currently loaded PyTorch model
    - shared_checkpoint_path: Path to the loaded checkpoint
    - shared_model_arch: Architecture of the loaded model
    - shared_model_num_classes: Number of classes for the loaded model
    - shared_inference_results: DataFrame of inference results
    - shared_prediction_data: Dictionary with prediction info (class, probs)
    """
    if st is None:
        raise ImportError(
            "Streamlit is required for shared session state."
        ) from _STREAMLIT_IMPORT_ERROR

    # Uploaded image data
    if "shared_uploaded_image" not in st.session_state:
        st.session_state.shared_uploaded_image = None

    if "shared_uploaded_image_path" not in st.session_state:
        st.session_state.shared_uploaded_image_path = None

    if "shared_uploaded_image_name" not in st.session_state:
        st.session_state.shared_uploaded_image_name = None

    # Model data
    if "shared_model" not in st.session_state:
        st.session_state.shared_model = None

    if "shared_checkpoint_path" not in st.session_state:
        st.session_state.shared_checkpoint_path = None

    if "shared_model_arch" not in st.session_state:
        st.session_state.shared_model_arch = None

    if "shared_model_num_classes" not in st.session_state:
        st.session_state.shared_model_num_classes = None

    # Inference results
    if "shared_inference_results" not in st.session_state:
        st.session_state.shared_inference_results = None

    # Prediction data (for single image predictions)
    if "shared_prediction_data" not in st.session_state:
        st.session_state.shared_prediction_data = None
