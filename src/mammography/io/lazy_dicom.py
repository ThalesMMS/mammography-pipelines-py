"""
Lazy loading wrapper for DICOM datasets.

Defers loading of pixel data until explicitly accessed to reduce memory usage
and improve loading times for large mammography datasets.

WARNING: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
No medical decision should be based on these results.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError

logger = logging.getLogger(__name__)


class LazyDicomDataset:
    """
    Lazy loading wrapper for DICOM datasets.

    Loads DICOM metadata immediately but defers pixel_array loading until
    the property is explicitly accessed. This dramatically reduces memory
    usage when working with large datasets where pixel data may not be
    needed immediately.

    Example:
        >>> ds = LazyDicomDataset("path/to/file.dcm")
        >>> # Metadata is available immediately
        >>> patient_id = ds.PatientID
        >>> # Pixel data is only loaded when accessed
        >>> pixels = ds.pixel_array  # Loading happens here

    Attributes:
        filepath: Path to the DICOM file
        _dataset: The underlying pydicom dataset (metadata only)
        _pixel_array: Cached pixel data array (None until accessed)
    """

    def __init__(
        self,
        filepath: Union[str, Path],
        stop_before_pixels: bool = True
    ):
        """
        Initialize lazy DICOM dataset.

        Args:
            filepath: Path to DICOM file
            stop_before_pixels: If True, stops reading before pixel data.
                This is the default behavior for lazy loading. Set to False
                to load everything immediately (defeats lazy loading).

        Raises:
            FileNotFoundError: If the file does not exist
            InvalidDicomError: If the file is not a valid DICOM file
        """
        self.filepath = Path(filepath)

        if not self.filepath.exists():
            raise FileNotFoundError(f"DICOM file not found: {self.filepath}")

        try:
            # Load metadata only, defer pixel data
            self._dataset = pydicom.dcmread(
                str(self.filepath),
                stop_before_pixels=stop_before_pixels,
                force=True
            )
        except Exception as exc:
            raise InvalidDicomError(
                f"Failed to read DICOM metadata from {self.filepath}: {exc!r}"
            ) from exc

        # Cache for pixel array (loaded on demand)
        self._pixel_array: Optional[np.ndarray] = None
        self._stop_before_pixels = stop_before_pixels

        logger.debug(f"Loaded DICOM metadata from {self.filepath}")

    @property
    def pixel_array(self) -> np.ndarray:
        """
        Load and return pixel data array (lazy loading).

        The pixel data is loaded from disk only on first access and then
        cached for subsequent accesses. This property provides transparent
        lazy loading of pixel data.

        Returns:
            Pixel data as numpy array

        Raises:
            RuntimeError: If pixel data cannot be loaded
        """
        if self._pixel_array is None:
            try:
                # If we initially skipped pixels, reload with pixel data
                if self._stop_before_pixels:
                    logger.debug(f"Loading pixel data from {self.filepath}")
                    full_dataset = pydicom.dcmread(str(self.filepath), force=True)
                    self._pixel_array = full_dataset.pixel_array
                else:
                    # Pixel data was already loaded
                    self._pixel_array = self._dataset.pixel_array
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load pixel data from {self.filepath}: {exc!r}"
                ) from exc

        return self._pixel_array

    def clear_pixel_cache(self) -> None:
        """
        Clear cached pixel data to free memory.

        Metadata remains in memory, but pixel data will be reloaded
        from disk on next access to pixel_array property.
        """
        if self._pixel_array is not None:
            logger.debug(f"Clearing pixel cache for {self.filepath}")
            self._pixel_array = None

    @property
    def is_pixel_data_loaded(self) -> bool:
        """Check if pixel data has been loaded into memory."""
        return self._pixel_array is not None

    def __getattr__(self, name: str) -> Any:
        """
        Delegate attribute access to underlying pydicom dataset.

        This allows transparent access to DICOM metadata attributes
        like PatientID, StudyInstanceUID, etc.

        Args:
            name: Attribute name

        Returns:
            Attribute value from DICOM dataset

        Raises:
            AttributeError: If attribute doesn't exist in dataset
        """
        try:
            return getattr(self._dataset, name)
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

    def __contains__(self, name: str) -> bool:
        """Check if DICOM tag exists in dataset."""
        return name in self._dataset

    def __repr__(self) -> str:
        """String representation of lazy DICOM dataset."""
        pixel_status = "loaded" if self.is_pixel_data_loaded else "not loaded"
        return (
            f"LazyDicomDataset(filepath={self.filepath}, "
            f"pixel_data={pixel_status})"
        )
