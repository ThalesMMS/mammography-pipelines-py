#
# 0_📁_Dataset_Browser.py
# mammography-pipelines
#
# Streamlit page for browsing mammography datasets with image grid and metadata table.
#
# Thales Matheus Mendonca Santos - February 2026
#
"""Dataset browser page for exploring mammography images and metadata."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import pandas as pd

from mammography.apps.web_ui.utils import ensure_shared_session_state
from mammography.apps.web_ui.components.dataset_viewer import DatasetViewer

try:
    import streamlit as st
except Exception as exc:  # pragma: no cover - optional UI dependency
    st = None
    _STREAMLIT_IMPORT_ERROR = exc
else:
    _STREAMLIT_IMPORT_ERROR = None


def _require_streamlit() -> None:
    """Raise ImportError if Streamlit is not available."""
    if st is None:
        raise ImportError(
            "Streamlit is required to run the web UI dashboard."
        ) from _STREAMLIT_IMPORT_ERROR


def _ensure_session_defaults() -> None:
    """Initialize session state with default values."""
    if "dataset_metadata" not in st.session_state:
        st.session_state.dataset_metadata = None
    if "dataset_viewer" not in st.session_state:
        st.session_state.dataset_viewer = None
    if "dataset_csv_path" not in st.session_state:
        st.session_state.dataset_csv_path = None
    if "dataset_image_root" not in st.session_state:
        st.session_state.dataset_image_root = None


def _load_dataset(csv_path: str, image_root: Optional[str]) -> Optional[pd.DataFrame]:
    """Load dataset metadata from CSV file.

    Args:
        csv_path: Path to the CSV metadata file
        image_root: Optional root directory for images

    Returns:
        DataFrame with metadata, or None if loading fails
    """
    try:
        # Create or retrieve dataset viewer
        if st.session_state.dataset_viewer is None:
            st.session_state.dataset_viewer = DatasetViewer()

        viewer = st.session_state.dataset_viewer

        # Load metadata
        metadata = viewer.load_metadata(csv_path=csv_path, image_root=image_root)

        # Store in session state
        st.session_state.dataset_metadata = metadata
        st.session_state.dataset_csv_path = csv_path
        st.session_state.dataset_image_root = image_root

        return metadata

    except Exception as exc:
        st.error(f"❌ Failed to load dataset: {exc}")
        return None


def _render_dataset_stats(viewer: DatasetViewer, metadata: pd.DataFrame) -> None:
    """Render dataset statistics in an expandable section.

    Args:
        viewer: DatasetViewer instance
        metadata: Dataset metadata DataFrame
    """
    with st.expander("📊 Dataset Statistics", expanded=False):
        stats = viewer.get_dataset_stats(metadata)

        if not stats:
            st.warning("No statistics available")
            return

        # Overall statistics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Images", stats.get("total_images", 0))

        with col2:
            if "density_distribution" in stats:
                num_classes = len(stats["density_distribution"])
                st.metric("Density Classes", num_classes)

        with col3:
            if "view_distribution" in stats:
                num_views = len(stats["view_distribution"])
                st.metric("View Types", num_views)

        # Density distribution
        if "density_distribution" in stats:
            st.subheader("Density Class Distribution")

            density_data = stats["density_distribution"]
            density_labels = {
                1: "Fatty (A)",
                2: "Scattered (B)",
                3: "Heterogeneous (C)",
                4: "Dense (D)",
            }

            # Create bar chart data
            chart_data = pd.DataFrame({
                "Density Class": [density_labels.get(k, f"Class {k}") for k in density_data.keys()],
                "Count": list(density_data.values()),
            })

            st.bar_chart(chart_data.set_index("Density Class"))

        # View distribution
        if "view_distribution" in stats:
            st.subheader("View Position Distribution")

            view_data = stats["view_distribution"]
            view_df = pd.DataFrame({
                "View": list(view_data.keys()),
                "Count": list(view_data.values()),
            })

            st.bar_chart(view_df.set_index("View"))

        # Laterality distribution
        if "laterality_distribution" in stats:
            st.subheader("Laterality Distribution")

            lat_data = stats["laterality_distribution"]
            lat_df = pd.DataFrame({
                "Side": list(lat_data.keys()),
                "Count": list(lat_data.values()),
            })

            st.bar_chart(lat_df.set_index("Side"))


def _render_filter_controls() -> tuple[Optional[List[int]], Optional[str], int, int, int]:
    """Render filter controls in the sidebar.

    Returns:
        Tuple of (density_filter, view_filter, columns, image_size, max_images)
    """
    st.sidebar.header("🔍 Display Controls")

    # Density filter
    st.sidebar.subheader("Filter by Density")
    density_options = {
        "Fatty (A)": 1,
        "Scattered (B)": 2,
        "Heterogeneous (C)": 3,
        "Dense (D)": 4,
    }

    selected_densities = st.sidebar.multiselect(
        "Select density classes",
        options=list(density_options.keys()),
        default=list(density_options.keys()),
        help="Filter images by BI-RADS density classification",
    )

    density_filter = [density_options[d] for d in selected_densities] if selected_densities else None

    # View filter
    st.sidebar.subheader("Filter by View")
    view_filter = st.sidebar.selectbox(
        "Select view position",
        options=["All", "CC", "MLO"],
        index=0,
        help="Filter images by mammographic view position",
    )

    if view_filter == "All":
        view_filter = None

    # Grid layout controls
    st.sidebar.subheader("Grid Layout")

    columns = st.sidebar.slider(
        "Columns",
        min_value=1,
        max_value=6,
        value=3,
        step=1,
        help="Number of columns in the image grid",
    )

    image_size = st.sidebar.slider(
        "Image Size (px)",
        min_value=128,
        max_value=512,
        value=256,
        step=64,
        help="Target size for displayed images",
    )

    max_images = st.sidebar.slider(
        "Images per Page",
        min_value=6,
        max_value=48,
        value=12,
        step=6,
        help="Maximum number of images to display per page",
    )

    return density_filter, view_filter, columns, image_size, max_images


def main() -> None:
    """Render the dataset browser page."""
    _require_streamlit()

    st.set_page_config(
        page_title="Dataset Browser - Mammography Pipelines",
        page_icon="📁",
        layout="wide",
    )

    # Initialize shared session state for cross-page data persistence
    try:
        ensure_shared_session_state()
        _ensure_session_defaults()
    except Exception as exc:
        st.error(f"❌ Failed to initialize session state: {exc}")
        st.stop()

    st.title("📁 Dataset Browser")
    st.header("Explore Mammography Datasets")

    st.markdown("""
    Browse and explore mammography datasets with interactive image viewing and metadata filtering.
    This tool helps you:

    - **Visualize** mammography images in a grid layout
    - **Filter** images by density class, view position, and other attributes
    - **Analyze** dataset statistics and class distributions
    - **Export** metadata tables for further analysis
    """)

    st.info(
        "💡 **Quick Start:** (1) Enter the path to your dataset CSV file, "
        "(2) Optionally specify the image root directory, "
        "(3) Click 'Load Dataset', and (4) Use the sidebar filters to explore."
    )

    # Dataset loading section
    st.subheader("1. Load Dataset")

    col1, col2 = st.columns([2, 1])

    with col1:
        csv_path = st.text_input(
            "Dataset CSV Path",
            value=st.session_state.dataset_csv_path or "classificacao.csv",
            placeholder="path/to/classificacao.csv",
            help="Path to the CSV file containing dataset metadata",
        )

    with col2:
        image_root = st.text_input(
            "Image Root Directory (optional)",
            value=st.session_state.dataset_image_root or "archive",
            placeholder="path/to/images/",
            help="Root directory for image paths (if not absolute in CSV)",
        )

    load_button = st.button(
        "🔄 Load Dataset",
        type="primary",
        disabled=not csv_path,
        help="Load dataset metadata from the specified CSV file",
    )

    # Load dataset when button clicked or if already loaded
    if load_button:
        if not os.path.exists(csv_path):
            st.error(f"❌ CSV file not found: {csv_path}")
        else:
            with st.spinner("Loading dataset metadata..."):
                metadata = _load_dataset(csv_path, image_root if image_root else None)

                if metadata is not None:
                    st.success(f"✅ Successfully loaded {len(metadata)} records from dataset")

    # Check if dataset is loaded
    if st.session_state.dataset_metadata is None:
        st.warning("⚠️ No dataset loaded. Please load a dataset to continue.")
        st.stop()

    metadata = st.session_state.dataset_metadata
    viewer = st.session_state.dataset_viewer

    # Render dataset statistics
    _render_dataset_stats(viewer, metadata)

    # Render filter controls in sidebar
    density_filter, view_filter, columns, image_size, max_images = _render_filter_controls()

    # Clear cache button in sidebar
    st.sidebar.divider()
    if st.sidebar.button("🗑️ Clear Image Cache"):
        viewer.clear_cache()
        st.sidebar.success("Cache cleared!")

    # Display mode tabs
    st.header("2. Browse Dataset")

    tab1, tab2 = st.tabs(["🖼️ Image Grid", "📋 Metadata Table"])

    with tab1:
        st.subheader("Image Grid View")

        # Render image grid
        viewer.render(
            metadata=metadata,
            columns=columns,
            image_size=image_size,
            max_images=max_images,
            show_metadata=True,
            density_filter=density_filter,
            view_filter=view_filter,
        )

    with tab2:
        st.subheader("Metadata Table View")

        # Apply same filters to table view
        filtered_df = metadata.copy()

        if density_filter is not None and len(density_filter) > 0:
            # Filter by density class
            if "professional_label" in filtered_df.columns:
                filtered_df = filtered_df[
                    filtered_df["professional_label"].isin(density_filter)
                ]
            elif "density_class" in filtered_df.columns:
                filtered_df = filtered_df[
                    filtered_df["density_class"].isin(density_filter)
                ]
            elif "Classification" in filtered_df.columns:
                filtered_df = filtered_df[
                    filtered_df["Classification"].isin(density_filter)
                ]

        if view_filter is not None:
            # Filter by view
            if "view" in filtered_df.columns:
                filtered_df = filtered_df[filtered_df["view"] == view_filter]
            elif "ViewPosition" in filtered_df.columns:
                filtered_df = filtered_df[filtered_df["ViewPosition"] == view_filter]

        # Render metadata table
        viewer.render_metadata_table(metadata=filtered_df, max_rows=200)

        # Download button for filtered data
        if len(filtered_df) > 0:
            csv_data = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Metadata as CSV",
                data=csv_data,
                file_name="filtered_metadata.csv",
                mime="text/csv",
                help="Download the currently filtered metadata as a CSV file",
            )


if __name__ == "__main__":
    main()
