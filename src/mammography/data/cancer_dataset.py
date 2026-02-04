#
# cancer_dataset.py
# mammography-pipelines
#
# Dataset classes for RSNA Breast Cancer Detection pipeline.
#
# Thales Matheus Mendonça Santos - November 2025
#
"""Dataset classes for RSNA Breast Cancer Detection.

WARNING: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
No medical decision should be based on these results.

This module provides PyTorch Dataset implementations for loading and preprocessing
mammography DICOM images for cancer detection tasks. It includes:

- SampleInfo: Dataclass for sample metadata
- MammoDicomDataset: Dataset that scans directories and loads DICOM files
- MammographyDataset: PyTorch dataset that provides balanced (RGB image, label) pairs
- dataset_summary: Count samples per class in a dataset
- split_dataset: Split dataset into train/validation subsets
- make_dataloader: Create a standardized DataLoader

These classes are extracted from the original eda_cancer.py for better modularity
and reusability across different pipelines.

Example usage:
    >>> from mammography.data.cancer_dataset import MammoDicomDataset, split_dataset
    >>> from mammography.commands.cancer_config import load_labels_dict
    >>>
    >>> # Load labels from CSV
    >>> labels = load_labels_dict("path/to/labels.csv")
    >>>
    >>> # Create dataset from DICOM directory
    >>> dataset = MammoDicomDataset(
    ...     data_dir="path/to/dicoms",
    ...     labels_by_accession=labels,
    ...     exclude_class_5=True
    ... )
    >>>
    >>> # Split into train/validation
    >>> train_ds, val_ds = split_dataset(dataset, val_fraction=0.2, seed=42)
    >>>
    >>> # Create dataloaders
    >>> train_loader = make_dataloader(
    ...     train_ds,
    ...     batch_size=16,
    ...     shuffle=True,
    ...     num_workers=2,
    ...     device=torch.device("cpu")
    ... )
"""

from dataclasses import dataclass
import os
from typing import Dict, List, Optional, Tuple, Union
import warnings

from PIL import Image
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from ..io.dicom import dicom_to_pil_rgb


@dataclass
class SampleInfo:
    """Lightweight container with the metadata needed to reconstruct each sample."""

    accession: str
    classification: Optional[int]
    path: str
    idx: int


class MammoDicomDataset(Dataset):
    """Dataset that scans data_dir subfolders and returns preprocessed samples."""

    def __init__(
        self,
        data_dir: str,
        labels_by_accession: Dict[str, int],
        exclude_class_5: bool = True,
        include_unlabeled: bool = False,
        transform: Optional[torch.nn.Module] = None,
        exts: Tuple[str, ...] = (".dcm", ".dicom", ".DCM", ".DICOM"),
    ):
        self.data_dir = data_dir
        self.labels_by_accession = labels_by_accession
        self.exclude_class_5 = exclude_class_5
        self.include_unlabeled = include_unlabeled
        self.transform = transform
        self.exts = exts

        self.samples: List[SampleInfo] = []
        self._build_index()

    def _list_dirs(self, root: str) -> List[str]:
        return [d for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]

    def _find_first_dicom(self, folder: str) -> Optional[str]:
        dicoms = []
        for curr, _, files in os.walk(folder):
            for f in files:
                fp = os.path.join(curr, f)
                name = f.lower()
                if fp.endswith(self.exts) or name.endswith(".dcm") or name.endswith(".dicom"):
                    dicoms.append(fp)
        dicoms.sort()
        return dicoms[0] if dicoms else None

    def _build_index(self):
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"data_dir '{self.data_dir}' não existe. Verifique o caminho.")

        idx = 0
        for sub in self._list_dirs(self.data_dir):
            accession = str(sub).strip()
            label = self.labels_by_accession.get(accession)

            if label == 5 and self.exclude_class_5:
                continue
            if (label is None) and (not self.include_unlabeled):
                continue

            folder = os.path.join(self.data_dir, sub)
            dcm_path = self._find_first_dicom(folder)
            if dcm_path is None:
                continue

            self.samples.append(SampleInfo(accession=accession, classification=label, path=dcm_path, idx=idx))
            idx += 1

        if len(self.samples) == 0:
            warnings.warn("Nenhuma amostra encontrada. Verifique diretórios e CSV.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        from torchvision import transforms as T

        info = self.samples[i]
        img = dicom_to_pil_rgb(info.path)

        # Convert RGB to grayscale (model expects 1 channel)
        img = img.convert("L")

        # Apply transform if provided, otherwise convert to tensor
        if self.transform is not None:
            img = self.transform(img)
        else:
            # Default: convert PIL Image to tensor
            img = T.ToTensor()(img)

        label = info.classification if (info.classification is not None) else -1
        return {
            "image": img,
            "label": label,
            "accession": info.accession,
            "path": info.path,
            "idx": info.idx,
        }


class MammographyDataset(Dataset):
    """PyTorch dataset that provides balanced (RGB image, label) pairs."""

    def __init__(self, meta_df: pl.DataFrame = None, img_dir: str = None, transform=None, dataframe=None):
        """Store balanced samples and the root directory of RGB images.

        Parameters
        ----------
        meta_df : pl.DataFrame
            Table with helper columns (including `fname`) to locate images.
        img_dir : str
            Base directory (train or valid) containing the PNGs generated from DICOMs.
        dataframe : pl.DataFrame or pd.DataFrame
            Alias for meta_df for backward compatibility.
        transform : callable, opcional
            Torchvision transform pipeline (ToTensor, aug, etc.).
        """
        # Handle backward compatibility: dataframe parameter
        if dataframe is not None and meta_df is None:
            import pandas as pd
            if isinstance(dataframe, pd.DataFrame):
                # Convert pandas to polars
                meta_df = pl.from_pandas(dataframe)
            else:
                meta_df = dataframe

        self.df = meta_df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        """Return how many samples are available after balancing."""
        return len(self.df)

    def __getitem__(self, idx):
        """Load RGB image derived from the DICOM, apply transforms, and return (image, label)."""

        # Retrieve the binary label directly from the Polars DataFrame.
        label = self.df.get_column("cancer")
        label = torch.tensor(label[idx], dtype=torch.float32)

        # Build the absolute path to the PNG
        # Support both fname+img_dir and image_path formats
        if "image_path" in self.df.columns:
            img_path = self.df.get_column("image_path")[idx]
        elif "fname" in self.df.columns:
            img_fname = self.df.get_column("fname")[idx]
            img_path = f"{self.img_dir}/{img_fname}.png"
        else:
            raise ValueError("DataFrame must have either 'image_path' or 'fname' column")

        # Open the RGB image and convert to grayscale (model expects 1 channel)
        img = Image.open(img_path)
        if img.mode != "L":
            img = img.convert("L")

        # Apply transform if provided, otherwise convert to tensor
        if self.transform:
            img = self.transform(img)
        else:
            # Default: convert PIL Image to tensor
            from torchvision import transforms as T
            img = T.ToTensor()(img)

        return img, label


# ------------------------------- Utilidades -------------------------------


def dataset_summary(dataset: Dataset) -> Dict[int, int]:
    """Count how many samples exist per class 1–4 in the provided dataset."""

    counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for sample in dataset.samples:  # type: ignore[attr-defined]
        label = sample.classification
        if label in counts:
            counts[label] += 1
    return counts


def split_dataset(
    dataset: Dataset,
    val_fraction_or_lengths: Union[float, List[int]],
    seed: int = 42,
) -> Tuple[Dataset, Optional[Dataset]]:
    """Split the dataset into train/validation while preserving original indices.

    Args:
        dataset: Dataset to split
        val_fraction_or_lengths: Either a float (validation fraction) or list of ints (split sizes)
        seed: Random seed for reproducibility (default: 42)

    Returns:
        Tuple of (train_dataset, val_dataset). val_dataset is None if no validation split.
    """
    from torch.utils.data import random_split

    n_total = len(dataset)
    if n_total == 0:
        raise RuntimeError("Dataset vazio após aplicar filtros. Nada a treinar.")

    # Handle both list-of-sizes and fraction APIs
    if isinstance(val_fraction_or_lengths, list):
        # List of sizes - use PyTorch's random_split directly
        generator = torch.Generator().manual_seed(seed)
        splits = random_split(dataset, val_fraction_or_lengths, generator=generator)
        if len(splits) == 2:
            return splits[0], splits[1]
        elif len(splits) == 1:
            return splits[0], None
        else:
            # More than 2 splits - return first two
            return splits[0], splits[1]

    # Fraction-based split
    val_fraction = val_fraction_or_lengths
    if val_fraction <= 0.0:
        return dataset, None

    val_size = int(round(n_total * val_fraction))
    if val_size <= 0:
        val_size = 1
    if val_size >= n_total:
        val_size = max(1, n_total - 1)

    train_size = n_total - val_size
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(dataset, [train_size, val_size], generator=generator)
    return train_subset, val_subset


def make_dataloader(
    subset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: Optional[torch.device] = None,
) -> DataLoader:
    """Standardize DataLoader creation for train/validation."""

    # Default to CPU if no device specified
    if device is None:
        device = torch.device("cpu")

    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
