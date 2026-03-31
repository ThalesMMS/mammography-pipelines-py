#
# dataset_viewer.py
# mammography-pipelines
#
# Dataset viewer component for displaying mammography images in a grid layout with metadata.
#
# Thales Matheus Mendonça Santos - February 2026
#
"""Dataset viewer component for browsing mammography images with metadata."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from mammography.io.dicom import dicom_to_pil_rgb, is_dicom_path
from mammography.data.csv_loader import load_dataset_dataframe

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


class DatasetViewer:
    """Component for displaying mammography dataset images in a grid layout.

    This component provides functionality to:
    - Load dataset metadata from CSV files
    - Display images in a configurable grid layout
    - Show metadata for each image
    - Support filtering by density class, view, and other attributes
    - Handle both DICOM and PNG image formats
    - Provide pagination for large datasets

    Example:
        >>> viewer = DatasetViewer()
        >>> viewer.render(
        ...     csv_path="classificacao.csv",
        ...     image_root="archive/",
        ...     columns=3,
        ...     max_images=12
        ... )
    """

    def __init__(self) -> None:
        """Initialize the dataset viewer component."""
        _require_streamlit()
        self._metadata: Optional[pd.DataFrame] = None
        self._image_cache: Dict[str, Image.Image] = {}

    def load_metadata(
        self,
        csv_path: str,
        image_root: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load dataset metadata from a CSV file.

        Args:
            csv_path: Path to the metadata CSV file
            image_root: Optional root directory for image paths (if not absolute)

        Returns:
            DataFrame with dataset metadata

        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If required columns are missing
        """
        csv_file = Path(csv_path)
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        try:
            # Load metadata using the dataset loader
            metadata = load_dataset_dataframe(
                csv_path=str(csv_file),
                dicom_root=image_root,
                exclude_class_5=True,
            )

            # Store metadata for future use
            self._metadata = metadata
            LOGGER.info(
                "Loaded %d records from %s",
                len(metadata),
                csv_path,
            )

            return metadata

        except Exception as exc:
            raise ValueError(f"Failed to load metadata from {csv_path}: {exc}") from exc

    def _load_image(
        self,
        image_path: str,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> Optional[Image.Image]:
        """Load a single image from disk.

        Args:
            image_path: Path to the image file
            target_size: Optional (width, height) to resize image

        Returns:
            PIL Image object, or None if loading fails
        """
        # Check cache first
        cache_key = f"{image_path}_{target_size}"
        if cache_key in self._image_cache:
            return self._image_cache[cache_key]

        try:
            path = Path(image_path)
            if not path.exists():
                LOGGER.warning("Image file not found: %s", image_path)
                return None

            # Load based on file type
            if is_dicom_path(str(path)):
                img = dicom_to_pil_rgb(str(path))
            else:
                img = Image.open(path).convert("RGB")

            # Resize if requested
            if target_size is not None:
                img = img.resize(target_size, Image.Resampling.LANCZOS)

            # Cache the loaded image
            self._image_cache[cache_key] = img

            return img

        except Exception as exc:
            LOGGER.error("Failed to load image %s: %s", image_path, exc)
            return None

    def _get_density_label(self, density_class: Any) -> str:
        """Convert density class to readable label.

        Args:
            density_class: Density class (1-4 or string label)

        Returns:
            Human-readable density label
        """
        # Map numeric classes to BI-RADS labels
        density_map = {
            1: "Fatty (A)",
            2: "Scattered (B)",
            3: "Heterogeneous (C)",
            4: "Dense (D)",
            "1": "Fatty (A)",
            "2": "Scattered (B)",
            "3": "Heterogeneous (C)",
            "4": "Dense (D)",
            "A": "Fatty (A)",
            "B": "Scattered (B)",
            "C": "Heterogeneous (C)",
            "D": "Dense (D)",
        }

        return density_map.get(density_class, f"Class {density_class}")

    def render(
        self,
        metadata: Optional[pd.DataFrame] = None,
        columns: int = 3,
        image_size: int = 256,
        max_images: int = 12,
        show_metadata: bool = True,
        density_filter: Optional[List[int]] = None,
        view_filter: Optional[str] = None,
    ) -> None:
        """Render the dataset viewer grid.

        Args:
            metadata: DataFrame with image metadata (uses loaded metadata if None)
            columns: Number of columns in the grid layout
            image_size: Target size for displayed images (width/height in pixels)
            max_images: Maximum number of images to display per page
            show_metadata: Whether to show metadata below each image
            density_filter: Optional list of density classes to filter (1-4)
            view_filter: Optional view filter ('CC', 'MLO', or None for all)
        """
        _require_streamlit()

        # Use provided metadata or fall back to loaded metadata
        df = metadata if metadata is not None else self._metadata

        if df is None or len(df) == 0:
            st.warning("⚠️ No metadata loaded. Please load a dataset first.")
            return

        # Apply filters
        filtered_df = df.copy()

        if density_filter is not None and len(density_filter) > 0:
            # Filter by density class (professional_label is the standard column)
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

        if len(filtered_df) == 0:
            st.warning("⚠️ No images match the selected filters.")
            return

        # Pagination
        total_images = len(filtered_df)
        total_pages = (total_images + max_images - 1) // max_images

        if total_pages > 1:
            st.info(f"📊 Showing {min(max_images, total_images)} of {total_images} images")
            page = st.number_input(
                "Page",
                min_value=1,
                max_value=total_pages,
                value=1,
                step=1,
                help=f"Navigate through {total_pages} pages of images",
            )
        else:
            page = 1

        # Calculate slice for current page
        start_idx = (page - 1) * max_images
        end_idx = min(start_idx + max_images, total_images)
        page_df = filtered_df.iloc[start_idx:end_idx]

        # Render image grid
        rows = (len(page_df) + columns - 1) // columns

        for row_idx in range(rows):
            cols = st.columns(columns)

            for col_idx in range(columns):
                img_idx = row_idx * columns + col_idx
                if img_idx >= len(page_df):
                    break

                row_data = page_df.iloc[img_idx]

                with cols[col_idx]:
                    # Get image path
                    image_path = None
                    if "image_path" in row_data:
                        image_path = row_data["image_path"]
                    elif "path" in row_data:
                        image_path = row_data["path"]

                    if image_path is None:
                        st.error("❌ No image path in metadata")
                        continue

                    # Load and display image
                    img = self._load_image(
                        str(image_path),
                        target_size=(image_size, image_size),
                    )

                    if img is not None:
                        st.image(img, width="stretch")
                    else:
                        st.error(f"❌ Failed to load image")

                    # Show metadata if requested
                    if show_metadata:
                        metadata_items = []

                        # Accession number (standard column from load_dataset_dataframe)
                        if "accession" in row_data and pd.notna(row_data['accession']):
                            metadata_items.append(f"**ID:** {row_data['accession']}")
                        elif "AccessionNumber" in row_data and pd.notna(row_data['AccessionNumber']):
                            metadata_items.append(f"**ID:** {row_data['AccessionNumber']}")

                        # Density class (professional_label is the standard column)
                        if "professional_label" in row_data and pd.notna(row_data["professional_label"]):
                            label = self._get_density_label(row_data["professional_label"])
                            metadata_items.append(f"**Density:** {label}")
                        elif "density_class" in row_data and pd.notna(row_data["density_class"]):
                            label = self._get_density_label(row_data["density_class"])
                            metadata_items.append(f"**Density:** {label}")
                        elif "Classification" in row_data and pd.notna(row_data["Classification"]):
                            label = self._get_density_label(row_data["Classification"])
                            metadata_items.append(f"**Density:** {label}")

                        # View position (standard column from load_dataset_dataframe)
                        if "view" in row_data and pd.notna(row_data['view']):
                            metadata_items.append(f"**View:** {row_data['view']}")
                        elif "ViewPosition" in row_data and pd.notna(row_data['ViewPosition']):
                            metadata_items.append(f"**View:** {row_data['ViewPosition']}")

                        # Laterality
                        if "laterality" in row_data and pd.notna(row_data['laterality']):
                            metadata_items.append(f"**Side:** {row_data['laterality']}")
                        elif "Laterality" in row_data and pd.notna(row_data['Laterality']):
                            metadata_items.append(f"**Side:** {row_data['Laterality']}")

                        # Display metadata in a compact format
                        if metadata_items:
                            st.markdown(
                                "<br>".join(metadata_items),
                                unsafe_allow_html=True,
                            )

    def render_metadata_table(
        self,
        metadata: Optional[pd.DataFrame] = None,
        max_rows: int = 100,
    ) -> None:
        """Render metadata as a searchable table.

        Args:
            metadata: DataFrame with image metadata (uses loaded metadata if None)
            max_rows: Maximum number of rows to display
        """
        _require_streamlit()

        # Use provided metadata or fall back to loaded metadata
        df = metadata if metadata is not None else self._metadata

        if df is None or len(df) == 0:
            st.warning("⚠️ No metadata loaded. Please load a dataset first.")
            return

        st.subheader("📋 Metadata Table")

        # Show record count
        st.info(f"📊 Total records: {len(df)}")

        # Display table (Streamlit automatically makes it searchable/sortable)
        display_df = df.head(max_rows)

        # Select relevant columns for display
        # Standard columns from load_dataset_dataframe: image_path, professional_label, accession, view
        display_columns = []
        for col in ["accession", "professional_label", "view", "image_path",
                    "AccessionNumber", "density_class", "Classification",
                    "ViewPosition", "laterality", "Laterality", "path"]:
            if col in display_df.columns:
                display_columns.append(col)

        if display_columns:
            st.dataframe(
                display_df[display_columns],
                width="stretch",
                height=400,
            )
        else:
            # Show all columns if no standard columns found
            st.dataframe(
                display_df,
                width="stretch",
                height=400,
            )

        if len(df) > max_rows:
            st.caption(f"Showing first {max_rows} of {len(df)} records")

    def get_dataset_stats(
        self,
        metadata: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Get statistics about the dataset.

        Args:
            metadata: DataFrame with image metadata (uses loaded metadata if None)

        Returns:
            Dictionary with dataset statistics
        """
        df = metadata if metadata is not None else self._metadata

        if df is None or len(df) == 0:
            return {}

        stats: Dict[str, Any] = {
            "total_images": len(df),
        }

        # Density class distribution (professional_label is the standard column)
        if "professional_label" in df.columns:
            stats["density_distribution"] = df["professional_label"].value_counts().to_dict()
        elif "density_class" in df.columns:
            stats["density_distribution"] = df["density_class"].value_counts().to_dict()
        elif "Classification" in df.columns:
            stats["density_distribution"] = df["Classification"].value_counts().to_dict()

        # View distribution (view is the standard column)
        if "view" in df.columns:
            stats["view_distribution"] = df["view"].value_counts().to_dict()
        elif "ViewPosition" in df.columns:
            stats["view_distribution"] = df["ViewPosition"].value_counts().to_dict()

        # Laterality distribution
        if "laterality" in df.columns:
            stats["laterality_distribution"] = df["laterality"].value_counts().to_dict()
        elif "Laterality" in df.columns:
            stats["laterality_distribution"] = df["Laterality"].value_counts().to_dict()

        return stats

    def clear_cache(self) -> None:
        """Clear the internal image cache to free memory."""
        self._image_cache.clear()
        LOGGER.info("Image cache cleared")
