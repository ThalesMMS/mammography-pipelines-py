"""Data loading sections for the Streamlit Training page."""

from __future__ import annotations

from typing import Any, Dict

try:
    import streamlit as st
except Exception as exc:  # pragma: no cover - optional UI dependency
    st = None
    _STREAMLIT_IMPORT_ERROR = exc
else:
    _STREAMLIT_IMPORT_ERROR = None


def _require_streamlit() -> None:
    if st is None:
        raise ImportError("Streamlit is required to run the web UI dashboard.") from _STREAMLIT_IMPORT_ERROR


def render_data_loading_section() -> Dict[str, Any]:
    """Render data loading settings section and return parameters."""
    st.subheader("💾 Data Loading")

    config = {}

    col1, col2, col3 = st.columns(3)

    with col1:
        config["num_workers"] = st.number_input(
            "Number of Workers",
            min_value=0,
            value=4,
            step=1,
            help="Number of worker processes for data loading (0 = main process)",
        )

    with col2:
        config["prefetch_factor"] = st.number_input(
            "Prefetch Factor",
            min_value=0,
            value=4,
            step=1,
            help="Number of batches to prefetch per worker",
        )

    with col3:
        config["persistent_workers"] = st.checkbox(
            "Persistent Workers",
            value=True,
            help="Keep workers alive between epochs",
        )

    config["loader_heuristics"] = st.checkbox(
        "Loader Heuristics",
        value=True,
        help="Apply automatic heuristics for optimal data loading",
    )

    return config

def render_class_balancing_section() -> Dict[str, Any]:
    """Render class balancing settings section and return parameters."""
    st.subheader("⚖️ Class Balancing")

    config = {}

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Class Weights**")

        config["class_weights"] = st.selectbox(
            "Class Weights Mode",
            options=["none", "auto"],
            help="How to weight classes in loss function (none/auto)",
        )

        config["class_weights_alpha"] = st.number_input(
            "Class Weights Alpha",
            min_value=0.01,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="Exponent for auto class weights calculation",
        )

    with col2:
        st.markdown("**Weighted Sampling**")

        config["sampler_weighted"] = st.checkbox(
            "Weighted Sampler",
            value=False,
            help="Use weighted random sampling to balance classes",
        )

        config["sampler_alpha"] = st.number_input(
            "Sampler Alpha",
            min_value=0.01,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="Exponent for weighted sampler calculation",
        )

    return config

def render_augmentation_section() -> Dict[str, Any]:
    """Render augmentation settings section and return parameters."""
    st.subheader("🎨 Data Augmentation")

    config = {}

    config["augment"] = st.checkbox(
        "Enable Augmentation",
        value=True,
        help="Enable data augmentation during training",
    )

    if config["augment"]:
        col1, col2, col3 = st.columns(3)

        with col1:
            config["augment_vertical"] = st.checkbox(
                "Vertical Flip",
                value=False,
                help="Enable random vertical flips",
            )

        with col2:
            config["augment_color"] = st.checkbox(
                "Color Jitter",
                value=False,
                help="Enable random brightness/contrast adjustment",
            )

        with col3:
            config["augment_rotation_deg"] = st.number_input(
                "Rotation (degrees)",
                min_value=0.0,
                max_value=45.0,
                value=5.0,
                step=1.0,
                help="Maximum rotation angle for random rotation",
            )
    else:
        config["augment_vertical"] = False
        config["augment_color"] = False
        config["augment_rotation_deg"] = 0.0

    return config
