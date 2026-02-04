"""
Mock data generators for DICOM files and datasets used in testing.

Provides reusable utilities for creating synthetic DICOM datasets and metadata
files for testing purposes. These mock data generators follow the same structure
as real mammography datasets but use synthetic data.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import pydicom
    from pydicom.dataset import FileMetaDataset
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False


def generate_mock_dicom(
    rows: int = 128,
    columns: int = 128,
    photometric_interpretation: str = "MONOCHROME2",
    bits_stored: int = 16,
    view_position: Optional[str] = "CC",
    image_laterality: Optional[str] = "L",
    patient_id: str = "TEST_PATIENT_001",
    accession_number: Optional[str] = None,
    rescale_slope: Optional[float] = None,
    rescale_intercept: Optional[float] = None,
    window_center: Optional[float] = None,
    window_width: Optional[float] = None,
    manufacturer: str = "SIEMENS",
    pixel_spacing: Optional[List[float]] = None,
    seed: Optional[int] = None,
):
    """
    Generate a mock DICOM dataset for testing.

    Args:
        rows: Image height in pixels
        columns: Image width in pixels
        photometric_interpretation: MONOCHROME1 or MONOCHROME2
        bits_stored: Bits per pixel (8, 12, or 16)
        view_position: Mammography view (CC, MLO, etc.)
        image_laterality: L or R
        patient_id: Patient identifier
        accession_number: Optional accession number
        rescale_slope: Optional RescaleSlope value
        rescale_intercept: Optional RescaleIntercept value
        window_center: Optional WindowCenter value
        window_width: Optional WindowWidth value
        manufacturer: Device manufacturer
        pixel_spacing: Optional [row_spacing, col_spacing] in mm
        seed: Random seed for reproducible pixel data

    Returns:
        pydicom.Dataset: Mock DICOM dataset

    Raises:
        ImportError: If pydicom is not available
    """
    if not PYDICOM_AVAILABLE:
        raise ImportError("pydicom is required to generate mock DICOM datasets")

    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random

    dataset = pydicom.Dataset()

    # Required DICOM fields
    dataset.PatientID = patient_id
    dataset.StudyInstanceUID = "1.2.840.12345.123456789"
    dataset.SeriesInstanceUID = "1.2.840.12345.987654321"
    dataset.SOPInstanceUID = "1.2.840.12345.456789123"
    dataset.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"  # Secondary Capture

    # File meta information
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = dataset.SOPClassUID
    file_meta.MediaStorageSOPInstanceUID = dataset.SOPInstanceUID
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    dataset.file_meta = file_meta

    # Image attributes
    dataset.Manufacturer = manufacturer
    dataset.BitsStored = bits_stored
    dataset.BitsAllocated = 16 if bits_stored > 8 else 8
    dataset.HighBit = bits_stored - 1
    dataset.PixelRepresentation = 0
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = photometric_interpretation
    dataset.Rows = rows
    dataset.Columns = columns

    # Optional attributes
    if pixel_spacing is not None:
        dataset.PixelSpacing = pixel_spacing
    else:
        dataset.PixelSpacing = [0.1, 0.1]

    if view_position is not None:
        dataset.ViewPosition = view_position

    if image_laterality is not None:
        dataset.ImageLaterality = image_laterality

    if accession_number is not None:
        dataset.AccessionNumber = accession_number

    if rescale_slope is not None:
        dataset.RescaleSlope = rescale_slope

    if rescale_intercept is not None:
        dataset.RescaleIntercept = rescale_intercept

    if window_center is not None:
        dataset.WindowCenter = window_center

    if window_width is not None:
        dataset.WindowWidth = window_width

    # Generate pixel data
    max_value = (2 ** bits_stored) - 1
    dtype = np.uint16 if bits_stored > 8 else np.uint8
    pixel_array = rng.randint(0, max_value + 1, (rows, columns), dtype=dtype)
    dataset.PixelData = pixel_array.tobytes()

    return dataset


def save_mock_dicom(
    dataset,
    path: Path,
) -> None:
    """
    Save a mock DICOM dataset to a file.

    Args:
        dataset: pydicom.Dataset to save
        path: Output file path

    Raises:
        ImportError: If pydicom is not available
    """
    if not PYDICOM_AVAILABLE:
        raise ImportError("pydicom is required to save DICOM datasets")

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        dataset.save_as(str(path), enforce_file_format=True)
    except TypeError:
        dataset.save_as(str(path), write_like_original=False)


def generate_mock_dataset(
    num_samples: int = 10,
    density_classes: Optional[List[str]] = None,
    views: Optional[List[str]] = None,
    lateralities: Optional[List[str]] = None,
    format: Literal["archive", "mamografias", "patches_completo"] = "archive",
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Generate a mock dataset with metadata for testing.

    Args:
        num_samples: Number of samples to generate
        density_classes: List of density classes (defaults to A, B, C, D)
        views: List of view positions (defaults to CC, MLO)
        lateralities: List of lateralities (defaults to L, R)
        format: Dataset format preset
        seed: Random seed for reproducibility

    Returns:
        Tuple of (DataFrame with metadata, dict with dataset info)
    """
    rng = np.random.RandomState(seed)

    if density_classes is None:
        density_classes = ["A", "B", "C", "D"]

    if views is None:
        views = ["CC", "MLO"]

    if lateralities is None:
        lateralities = ["L", "R"]

    records = []

    for i in range(num_samples):
        record = {
            "sample_id": i,
            "density_label": rng.choice(density_classes),
            "view": rng.choice(views),
            "laterality": rng.choice(lateralities),
        }

        if format == "archive":
            # Archive format uses AccessionNumber and classificacao.csv
            accession = f"ACC{i:06d}"
            record["AccessionNumber"] = accession
            record["Classification"] = record["density_label"]
            record["image_path"] = f"archive/{accession}/image.dcm"

        elif format == "mamografias":
            # Mamografias format uses subdirectories with featureS.txt
            subdir = f"patient_{i:04d}"
            record["subdirectory"] = subdir
            record["image_path"] = f"mamografias/{subdir}/image.png"
            record["features_file"] = f"mamografias/{subdir}/featureS.txt"

        elif format == "patches_completo":
            # Patches format uses root-level featureS.txt
            record["image_path"] = f"patches_completo/patch_{i:06d}.png"
            record["features_file"] = "patches_completo/featureS.txt"

        records.append(record)

    df = pd.DataFrame(records)

    info = {
        "num_samples": num_samples,
        "density_classes": density_classes,
        "views": views,
        "lateralities": lateralities,
        "format": format,
        "seed": seed,
    }

    return df, info


def create_mock_dataset_files(
    output_dir: Path,
    num_samples: int = 10,
    format: Literal["archive", "mamografias", "patches_completo"] = "archive",
    create_images: bool = True,
    image_size: Tuple[int, int] = (128, 128),
    seed: int = 42,
) -> Tuple[Path, pd.DataFrame]:
    """
    Create a complete mock dataset with files on disk.

    Args:
        output_dir: Directory to create dataset in
        num_samples: Number of samples to generate
        format: Dataset format preset
        create_images: Whether to create actual DICOM/PNG files
        image_size: (height, width) for generated images
        seed: Random seed for reproducibility

    Returns:
        Tuple of (dataset root path, metadata DataFrame)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, info = generate_mock_dataset(
        num_samples=num_samples,
        format=format,
        seed=seed,
    )

    if format == "archive":
        # Create classificacao.csv
        csv_path = output_dir / "classificacao.csv"
        df[["AccessionNumber", "Classification"]].to_csv(csv_path, index=False)

        if create_images and PYDICOM_AVAILABLE:
            # Create DICOM files
            for _, row in df.iterrows():
                dicom_path = output_dir / row["image_path"]
                dicom_dataset = generate_mock_dicom(
                    rows=image_size[0],
                    columns=image_size[1],
                    accession_number=row["AccessionNumber"],
                    view_position=row["view"],
                    image_laterality=row["laterality"],
                    seed=seed + row["sample_id"],
                )
                save_mock_dicom(dicom_dataset, dicom_path)

    elif format == "mamografias":
        # Create per-directory featureS.txt files
        if create_images:
            for subdir in df["subdirectory"].unique():
                subdir_path = output_dir / subdir
                subdir_path.mkdir(parents=True, exist_ok=True)

                # Create featureS.txt
                subdir_df = df[df["subdirectory"] == subdir]
                features_path = subdir_path / "featureS.txt"
                with open(features_path, "w") as f:
                    for _, row in subdir_df.iterrows():
                        f.write(f"Density: {row['density_label']}\n")
                        f.write(f"View: {row['view']}\n")
                        f.write(f"Laterality: {row['laterality']}\n")

                # Create PNG files
                for _, row in subdir_df.iterrows():
                    img_path = output_dir / row["image_path"]
                    _create_mock_png(img_path, image_size, seed + row["sample_id"])

    elif format == "patches_completo":
        # Create root-level featureS.txt
        features_path = output_dir / "featureS.txt"
        with open(features_path, "w") as f:
            for _, row in df.iterrows():
                f.write(f"{Path(row['image_path']).name}\t")
                f.write(f"{row['density_label']}\t")
                f.write(f"{row['view']}\t")
                f.write(f"{row['laterality']}\n")

        if create_images:
            # Create PNG files
            for _, row in df.iterrows():
                img_path = output_dir / row["image_path"]
                _create_mock_png(img_path, image_size, seed + row["sample_id"])

    return output_dir, df


def _create_mock_png(
    path: Path,
    size: Tuple[int, int],
    seed: int,
) -> None:
    """Create a mock PNG file with random pixel data."""
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("PIL is required to create PNG files")

    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    arr = rng.randint(0, 256, (*size, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    img.save(path)


def generate_mock_embeddings(
    num_samples: int = 10,
    feature_dim: int = 2048,
    seed: int = 42,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Generate mock embeddings and metadata for testing.

    Args:
        num_samples: Number of embedding vectors to generate
        feature_dim: Dimensionality of embeddings
        seed: Random seed for reproducibility

    Returns:
        Tuple of (embeddings array, metadata DataFrame)
    """
    rng = np.random.RandomState(seed)

    # Generate random embeddings
    embeddings = rng.randn(num_samples, feature_dim).astype(np.float32)

    # Generate metadata
    metadata = pd.DataFrame({
        "accession": [f"ACC{i:06d}" for i in range(num_samples)],
        "image_path": [f"path/to/image_{i:06d}.dcm" for i in range(num_samples)],
        "density_label": rng.choice(["A", "B", "C", "D"], num_samples),
        "view": rng.choice(["CC", "MLO"], num_samples),
        "laterality": rng.choice(["L", "R"], num_samples),
    })

    return embeddings, metadata


def save_mock_embeddings(
    output_dir: Path,
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
) -> Tuple[Path, Path]:
    """
    Save mock embeddings to disk in the expected format.

    Args:
        output_dir: Directory to save embeddings
        embeddings: Embeddings array (N, feature_dim)
        metadata: Metadata DataFrame

    Returns:
        Tuple of (features.npy path, metadata.csv path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features_path = output_dir / "features.npy"
    metadata_path = output_dir / "metadata.csv"

    np.save(features_path, embeddings)
    metadata.to_csv(metadata_path, index=False)

    return features_path, metadata_path
