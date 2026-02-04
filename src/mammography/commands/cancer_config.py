#
# cancer_config.py
# mammography-pipelines
#
# Centralizes hyperparameters for the RSNA Breast Cancer Detection educational pipeline.
#
# Thales Matheus Mendonça Santos - November 2025
#
"""Configuration and hyperparameters for breast cancer detection pipeline.

WARNING: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
No medical decision should be based on these results.

This module centralizes all hyperparameters, preprocessing configurations, and
utility functions for the RSNA Breast Cancer Detection pipeline. It provides
reproducibility controls and data loading utilities.

Components:
    - HP: Dataclass containing all pipeline hyperparameters
    - seed_everything: Set random seeds for reproducibility
    - find_best_data_dir: Resolve data directory with typo correction
    - load_labels_dict: Load classification labels from CSV

Hyperparameter categories:
    - Image preprocessing: resize, crop, windowing percentiles
    - Training: train/val split, workers, random seed
    - Visualization: grid layout parameters
    - Data balancing: negative sampling strategy
    - Feature engineering: age imputation, categorical encoding

Example usage:
    >>> from mammography.commands.cancer_config import HP, seed_everything, load_labels_dict
    >>>
    >>> # Initialize hyperparameters
    >>> hp = HP()
    >>> print(f"Image size: {hp.IMG_CROP}x{hp.IMG_CROP}")
    >>> print(f"Train split: {hp.TRAIN_SPLIT}")
    >>>
    >>> # Set random seeds for reproducibility
    >>> seed_everything(hp.RANDOM_SEED)
    >>>
    >>> # Load labels from CSV
    >>> labels = load_labels_dict("data/classification.csv")
    >>> print(f"Loaded {len(labels)} labeled samples")
    >>>
    >>> # Custom hyperparameters
    >>> custom_hp = HP(
    ...     IMG_CROP=256,
    ...     TRAIN_SPLIT=0.85,
    ...     DOWNSAMPLE_RATIO=2.0
    ... )
"""

from dataclasses import dataclass, field
import os
import random
from typing import Dict, List

import numpy as np
import pandas as pd
import torch


@dataclass
class HP:
    """Core hyperparameters and preprocessing decisions for cancer detection pipeline.

    Image preprocessing:
    - IMG_RESIZE/IMG_CROP: resize then center-crop to the model's expected input.
    - WINDOW_P_LOW/HIGH: percentile windowing for robust DICOM contrast.

    Training:
    - TRAIN_SPLIT: patient-level split fraction (rest is validation).
    - DATALOADER_WORKERS: helper workers for data loading.
    - RANDOM_SEED: reproducibility anchor for splits and initialization.

    Visualization:
    - VIS_N_PATIENTS: number of patients to display in grids.
    - VIS_TILE: tile size for visualization.
    - VIS_GAP: gap between tiles in visualization.

    Data balancing:
    - DOWNSAMPLE_NEGATIVE: whether to downsample negative samples.
    - DOWNSAMPLE_RATIO: negatives ~ ratio * positives.

    Age preprocessing:
    - AGE_IMPUTE: imputation strategy ('mean' | 'median').
    - AGE_NORM: normalization strategy ('minmax' | 'zscore').

    Feature encoding:
    - CAT_COLS: categorical columns for one-hot encoding.
    """

    IMG_RESIZE: int = 256
    IMG_CROP: int = 224
    WINDOW_P_LOW: float = 0.5
    WINDOW_P_HIGH: float = 99.5
    TRAIN_SPLIT: float = 0.80
    DATALOADER_WORKERS: int = 2
    RANDOM_SEED: int = 19970507
    VIS_N_PATIENTS: int = 3
    VIS_TILE: int = 300
    VIS_GAP: int = 10
    DOWNSAMPLE_NEGATIVE: bool = True
    DOWNSAMPLE_RATIO: float = 1.0
    AGE_IMPUTE: str = "mean"
    AGE_NORM: str = "minmax"
    CAT_COLS: List[str] = field(default_factory=lambda: ["view", "laterality", "implant"])


# ----------------------- Utility Functions -----------------------

def seed_everything(seed: int = 42):
    """Seed Python/NumPy/Torch for reasonable reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_best_data_dir(pref: str) -> str:
    """Resolve common typos to locate the data directory more forgivingly."""
    if os.path.isdir(pref):
        return pref
    alt = pref.replace("archieve", "archive")
    if os.path.isdir(alt):
        print(f"[info] data_dir '{pref}' não encontrado; usando '{alt}'")
        return alt
    if os.path.isdir("/kaggle/input"):
        print("[aviso] data_dir não encontrado; verifique o caminho. Usando /kaggle/input para inspeção.")
        return "/kaggle/input"
    return pref


def load_labels_dict(csv_path: str) -> Dict[str, int]:
    """Read the classification CSV ensuring AccessionNumber stays as a string."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV '{csv_path}' não encontrado. Ajuste --csv_path.")
    df_csv = pd.read_csv(
        csv_path,
        dtype={"AccessionNumber": str, "Classification": int},
        parse_dates=["ClassificationDate"],
        dayfirst=False,
    )
    df_csv["AccessionNumber"] = df_csv["AccessionNumber"].str.strip()
    return {row["AccessionNumber"]: int(row["Classification"]) for _, row in df_csv.iterrows()}
