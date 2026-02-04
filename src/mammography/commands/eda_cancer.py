#
# eda_cancer.py
# mammography-pipelines
#
# Educational notebook-style pipeline for RSNA Breast Cancer Detection, covering data exploration, preprocessing, training, and visualization.
#
# Thales Matheus Mendonça Santos - November 2025
#
"""Educational notebook-style pipeline for RSNA Breast Cancer Detection.

REFACTORING NOTE
----------------
This file has been refactored from a monolithic script into a modular architecture.
The refactoring separates concerns into dedicated modules:

- **cancer_config**: Configuration and hyperparameters (HP class, seed management, path finding)
- **cancer_dataset**: Dataset classes and data loading utilities (MammoDicomDataset, MammographyDataset)
- **cancer_models**: Model architectures (ResNet50 classifier, device management)
- **cancer_trainer**: Training loop logic (train_one_epoch, evaluate, fit_classifier)
- **cancer_plots**: Visualization utilities (DICOM preview, transforms, plotting helpers)
- **mammography.io.dicom**: DICOM processing utilities (windowing, conversion to PIL)

This modular structure provides:
1. **Reusability**: Each module can be imported and used independently
2. **Testability**: Individual components can be unit tested
3. **Maintainability**: Changes to one component don't affect others
4. **Clarity**: This script now focuses on orchestrating the pipeline workflow

English overview
----------------
This script walks through a full supervised workflow: cohort inspection, quick
visualizations, basic preprocessing, a PyTorch dataset backed by PNGs, a
single-channel ResNet50 model, and end-to-end evaluation plots. It is meant to
be read and tweaked as a guided tutorial.

Context (translated from the original Portuguese notes)
-------------------------------------------------------
This executable-notebook script bundles the most common steps in a supervised
deep-learning workflow for mammography:

1. **Cohort characterization**: read the official RSNA 2022 competition CSVs,
   quickly inspect missing values, and create helper columns for visualization.
2. **Visual exploration**: charts showing cancer distribution (overall and by
   invasiveness) plus normalized grayscale DICOM samples.
3. **Data preparation**: balance the minority class, impute/normalize age,
   one-hot encode categorical metadata, and perform a stratified train/validation split.
4. **PyTorch dataset**: load balanced 256×256 PNG images with their binary labels.
5. **Model and training**: use a pretrained ResNet50 (adapted to a single channel)
   directly for binary classification, followed by a training loop with relevant
   clinical metrics (sensitivity/specificity).
6. **Graphical evaluation**: track loss, accuracy, and sensitivity/specificity per
   epoch using Lets-Plot.

Practical notes
---------------
To run the script outside Kaggle, adjust input/output paths via ``--csv-dir``,
``--dicom-dir``, ``--png-dir``, and ``--output-dir``. They accept local folders,
letting you point either to the original CSVs or to preprocessed DICOM/PNG images.
"""

# ------------------------------- Imports -------------------------------
import argparse
import copy
import json
import os
import random
import subprocess
import sys
import warnings

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from torch.amp.grad_scaler import GradScaler
import torch.optim as optim
import torchvision
import pydicom
from PIL import Image
from tqdm import tqdm
from torchvision.utils import make_grid

import matplotlib.pyplot as plt
from lets_plot import (
    GGBunch,
    LetsPlot,
    aes,
    coord_flip,
    element_blank,
    element_line,
    element_text,
    geom_bar,
    geom_imshow,
    geom_line,
    ggplot,
    labs,
    scale_fill_manual,
    scale_x_continuous,
    theme,
    theme_minimal,
)
from lets_plot.mapping import as_discrete
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms
from torchvision.models import ResNet50_Weights
from pydicom.pixel_data_handlers.util import apply_modality_lut

# ========================== Modular Mammography Pipeline Imports ==========================
# The following imports demonstrate the refactored modular architecture. Previously, all
# functionality was contained in this single file. Now, each module has a specific purpose:

# -------------------- Configuration Module --------------------
# Provides centralized configuration, hyperparameters, and utility functions
from mammography.commands.cancer_config import (
    HP,                      # Hyperparameters dataclass (batch size, epochs, learning rate, etc.)
    seed_everything,         # Sets random seeds for reproducibility across numpy, torch, random
    find_best_data_dir,      # Intelligently locates data directories (handles Kaggle paths)
    load_labels_dict,        # Loads classification labels from CSV files
)

# -------------------- Dataset Module --------------------
# Handles data loading, dataset creation, and preprocessing for both DICOM and PNG formats
from mammography.data.cancer_dataset import (
    SampleInfo,              # Dataclass for storing sample metadata (path, label, etc.)
    MammoDicomDataset,       # PyTorch Dataset for loading DICOM mammography images
    MammographyDataset,      # PyTorch Dataset for loading preprocessed PNG images
    dataset_summary,         # Computes class distribution statistics
    split_dataset,           # Creates train/validation splits with proper stratification
    make_dataloader,         # Factory function for creating configured DataLoaders
)

# -------------------- DICOM I/O Module --------------------
# Low-level DICOM processing utilities for image manipulation and format conversion
from mammography.io.dicom import (
    robust_window,           # Applies percentile-based windowing to DICOM pixel data
    dicom_to_pil_rgb,        # Converts DICOM to PIL Image with windowing and RGB replication
    is_dicom_path,           # Validates if a path points to a valid DICOM file
)

# -------------------- Models Module --------------------
# Neural network architectures and model building utilities
from mammography.models.cancer_models import (
    MammographyModel,        # Simple ResNet50-based binary classifier for cancer detection
    build_resnet50_classifier, # Factory function for building ResNet50 with custom number of classes
    resolve_device,          # Intelligently selects computation device (cuda/mps/cpu)
)

# -------------------- Training Module --------------------
# Training loop logic, evaluation metrics, and optimization utilities
from mammography.training.cancer_trainer import (
    DensityHistoryEntry,     # Dataclass for storing per-epoch training metrics
    get_sens_spec,           # Computes sensitivity and specificity from predictions
    train_one_epoch,         # Executes one training epoch with gradient updates
    evaluate,                # Evaluates model on validation set without gradients
    collect_predictions,     # Collects all predictions and labels for analysis
    fit_classifier,          # High-level training loop for the density classifier
)

# -------------------- Visualization Module --------------------
# Plotting and visualization utilities for DICOM images and training metrics
from mammography.vis.cancer_plots import (
    dicom_debug_preprocess,  # Debugging function to visualize DICOM preprocessing steps
    get_dicom_plt,           # Creates a lets-plot visualization of a DICOM image
    preview_transformed_samples, # Shows grid of transformed samples from dataset
    get_transforms,          # Returns configured torchvision transforms for training/inference
)

# ==================== Configuration and Constants ====================
# Refactoring note: Hyperparameters are now centralized in the HP dataclass (cancer_config module).
# This script imports HP and extracts commonly-used values for backward compatibility with the
# original notebook-style code. In new code, prefer accessing HP directly (e.g., HP.RANDOM_SEED).

NO_CANCER_LABEL = "No Cancer"
COLOR_HEALTHY = "#d8e2dc"
COLOR_PRESENT = "#f4acb7"
COLOR_INVASIVE = "#ee4266"

# Extract key hyperparameters from centralized HP configuration
# This allows the rest of the script to use familiar variable names while benefiting
# from centralized configuration management
RANDOM_SEED = HP.RANDOM_SEED              # Ensures reproducibility across runs
TRAIN_SPLIT_RATIO = HP.TRAIN_SPLIT        # Train/validation split ratio (default: 0.8)
DATALOADER_WORKERS = HP.DATALOADER_WORKERS # Number of workers for DataLoader (default: 2)

# Initialize plotting and random seed for reproducibility
LetsPlot.setup_html()
random.seed(RANDOM_SEED)

# --------------------------- Path Arguments ---------------------------

parser = argparse.ArgumentParser(
    description="RSNA Breast Cancer Detection pipeline com caminhos configuráveis.",
)
parser.add_argument(
    "--csv-dir",
    default=os.environ.get("RSNA_CSV_DIR", "/kaggle/input/rsna-breast-cancer-detection"),
    help="Diretório contendo train.csv/test.csv da competição.",
)
parser.add_argument(
    "--dicom-dir",
    "--dicom-root",
    default=os.environ.get(
        "RSNA_DICOM_DIR", "/kaggle/input/rsna-breast-cancer-detection/train_images"
    ),
    help="Diretório com as imagens DICOM originais para visualização.",
)
parser.add_argument(
    "--png-dir",
    default=os.environ.get(
        "RSNA_PNG_DIR", "/kaggle/input/rsna-breast-cancer-256-pngs"
    ),
    help="Diretório com as imagens PNG 256×256 utilizadas no dataset balanceado.",
)
parser.add_argument(
    "--output-dir",
    "--outdir",
    default=os.environ.get("RSNA_OUTPUT_DIR", "/kaggle/working"),
    help="Diretório raiz onde as cópias balanceadas (train/valid) serão salvas.",
)

# In notebook environments, extra args may be injected by the kernel.
ARGS, _ = parser.parse_known_args()

CSV_DIR = ARGS.csv_dir
TRAIN_CSV_PATH = os.path.join(CSV_DIR, "train.csv")
TEST_CSV_PATH = os.path.join(CSV_DIR, "test.csv")

# ==================== Intelligent Path Resolution ====================
# Refactoring note: find_best_data_dir() is now imported from cancer_config module.
# This function intelligently handles Kaggle-specific paths and falls back to provided
# paths when files don't exist. This makes the pipeline portable between local and
# Kaggle environments without code changes.

if not os.path.isfile(TRAIN_CSV_PATH):
    alt_csv_dir = find_best_data_dir(CSV_DIR)
    if alt_csv_dir != CSV_DIR:
        print(
            f"[info] train.csv não encontrado em '{CSV_DIR}'; usando diretório alternativo '{alt_csv_dir}'."
        )
        CSV_DIR = alt_csv_dir
        TRAIN_CSV_PATH = os.path.join(CSV_DIR, "train.csv")
        TEST_CSV_PATH = os.path.join(CSV_DIR, "test.csv")

# Use intelligent path resolution for all data directories
# This handles both local development paths and Kaggle competition paths
DICOM_DATA_DIR = find_best_data_dir(ARGS.dicom_dir)
PNG_DATA_DIR = find_best_data_dir(ARGS.png_dir)
OUTPUT_DIR = ARGS.output_dir
TRAIN_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "train")
VALID_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "valid")

# -------------------- Data and Cohort Characteristics --------------------

# Reading the official CSVs. On Kaggle, the files live in /kaggle/input.
# `glimpse` (Polars) is equivalent to `df.info()`, showing type and a sample.
# Discussion: we chose to keep the direct read of the original CSVs to highlight
# the competition's native format. In clinical settings this first inspection
# often reveals integrity issues (duplicate patients, age outliers) before
# considering more sophisticated techniques.
df_train = pl.read_csv(TRAIN_CSV_PATH)
df_test = pl.read_csv(TEST_CSV_PATH)

df_train.glimpse()

# Quick check of missing values to see where to impute or drop records.
# The print acts as a friendly header.
print('Missing values by column')
df_train.select(pl.all().is_null().sum())

# ----------------------- Exploratory Data Analysis ----------------------

# Helper set with human-readable categorical columns (strings) for plots.
# `target` is not used directly but keeps the code aligned with other notebooks
# where we might toggle labels.
# Discussion: we prefer deriving textual columns instead of mapping legends
# inside the chart. This makes it easier to reuse the same table in clinical
# reports without depending on the visualization layer.
target = 'cancer'

df_plt = df_train.with_columns(
    pl.when(pl.col('cancer') == 1).then('Cancer Present').otherwise(NO_CANCER_LABEL).alias('cancer'),
    # Column `invasive2`: combines cancer status + invasiveness as text.
    # Used to color plots with three main clinical categories.
    pl.when((pl.col('cancer') == 1) & (pl.col('invasive') == 1))
        .then('Invasive')
        .when((pl.col('cancer') == 1) & (pl.col('invasive') == 0))
        .then('Non-Invasive')
        .when((pl.col('cancer') == 0) & (pl.col('invasive') == 0))
        .then(NO_CANCER_LABEL)
        .otherwise(NO_CANCER_LABEL)
        .alias('invasive2')
)

# --------------------- Target Variable Plot ---------------------
def create_bar_plot(df, var, title, colors):
    """Helper to create a horizontal bar plot with consistent styling."""
    return (
        ggplot(df) +
        geom_bar(aes(x=as_discrete(var), fill=as_discrete(var)), color=None, size=0.5) +
        scale_fill_manual(values=colors) +
        theme_minimal() +
        theme(
            plot_title=element_text(hjust=0.5, face='bold'),
            legend_position="top",
            panel_grid_major=element_blank(),
            panel_grid_minor=element_blank(),
            legend_title=element_blank(),
            axis_title_x=element_blank(),
            axis_line_y=element_line(size=1)
        ) +
        coord_flip() +
        labs(y="Count", title=title)
    )

plt1 = create_bar_plot(df_plt, 'cancer', 'Cancer Distribution', [COLOR_HEALTHY, COLOR_PRESENT])
plt2 = create_bar_plot(df_plt, 'invasive2', 'Cancer Distribution by Invasiveness', [COLOR_HEALTHY, COLOR_INVASIVE, COLOR_PRESENT])

bunch = GGBunch()
bunch.add_plot(plt1, 0, 0, 500, 250)
bunch.add_plot(plt2, 520, 0, 500, 250)
bunch.show()

# --------------------- Optional: Density Pipeline ---------------------
# Light integration: set RUN_DENSITY=1 in the environment to run the density
# pipeline (ResNet50) using your classificacao.csv and the DICOMs in `archive/`.
# Optional environment variables:
#   DENSITY_CSV (default: classificacao.csv)
#   DENSITY_DICOM_ROOT (default: archive)
#   DENSITY_OUTDIR (default: outputs/mammo_resnet50_density)
#   DENSITY_EPOCHS, DENSITY_BATCH_SIZE, DENSITY_IMG_SIZE, DENSITY_DEVICE
try:
    if os.environ.get("RUN_DENSITY", "0") == "1":
        cmd = [
            sys.executable,
            "-m",
            "mammography.commands.train",
            "--arch",
            "resnet50",
            "--classes",
            "density",
            "--csv",
            os.environ.get("DENSITY_CSV", "classificacao.csv"),
            "--dicom-root",
            os.environ.get("DENSITY_DICOM_ROOT", "archive"),
            "--outdir",
            os.environ.get("DENSITY_OUTDIR", "outputs/mammo_resnet50_density"),
            "--epochs",
            os.environ.get("DENSITY_EPOCHS", "8"),
            "--batch-size",
            os.environ.get("DENSITY_BATCH_SIZE", "16"),
            "--img-size",
            os.environ.get("DENSITY_IMG_SIZE", "512"),
            "--device",
            os.environ.get("DENSITY_DEVICE", "auto"),
        ]
        print("[info] RUN_DENSITY=1 detectado — iniciando pipeline de densidade...")
        subprocess.run(cmd, check=True)
except Exception as _density_err:
    print(f"[aviso] Pipeline de densidade não executado: {_density_err}")

# ==================== DICOM Visualization (Using Refactored Modules) ====================
# Refactoring note: get_dicom_plt() is imported from cancer_plots module.
# Previously, DICOM loading and plotting logic was embedded in this file. Now it's
# encapsulated in the vis.cancer_plots module, making it reusable across the codebase.
# The function handles DICOM reading, windowing, and lets-plot visualization automatically.

# Retrieve patient subsets by clinical category and create a 3×3 visualization grid
categories = [
    ('Invasive', 'Invasive Cancer', 0),
    ('Non-Invasive', 'Non-Invasive Cancer', 1),
    (NO_CANCER_LABEL, NO_CANCER_LABEL, 2)
]
dicom_dir = DICOM_DATA_DIR
# Visualization parameters from centralized HP configuration
npatients = HP.VIS_N_PATIENTS  # Number of patients to show per category
tile = HP.VIS_TILE             # Size of each plot tile
gap = HP.VIS_GAP               # Gap between plot rows
bunch = GGBunch()

for cat_filter, cat_title, row_idx in categories:
    patients = df_plt.filter(pl.col('invasive2') == cat_filter).select(['patient_id', 'image_id'])
    patient_ids = patients.get_column('patient_id')
    img_ids = patients.get_column('image_id')

    for i in range(npatients):
        dcm_path = os.path.join(dicom_dir, str(patient_ids[i]), f"{img_ids[i]}.dcm")
        # get_dicom_plt() handles DICOM loading, windowing, and plot creation
        bunch.add_plot(get_dicom_plt(dcm_path, title=cat_title), i * tile, row_idx * (tile + gap), tile, tile)

bunch.show()

# ------------------------- Data Set and Data Loader ------------------------

# Simple balancing: the "cancer" class is quite rare. Here we perform random
# negative downsampling to match the number of positives.
# In production we would consider more robust strategies (e.g., `class_weight`).
# Discussion: downsampling reduces statistical sensitivity but is a fast way to
# make the prototype trainable without recalculating weights per batch. The
# intent is to emphasize balancing mechanics, not to optimize AUC.
df_target1 = df_train.filter(pl.col('cancer') == 1)
df_target0 = df_train.filter(pl.col('cancer') == 0)

# Didactic choice: when HP.DOWNSAMPLE_NEGATIVE=True, we cut the number of
# negatives to keep a controlled ratio vs positives. This speeds training and
# prevents the majority class from dominating iterations.
if HP.DOWNSAMPLE_NEGATIVE:
    n_pos = len(df_target1)
    n_neg_total = len(df_target0)
    n_neg_keep = min(n_neg_total, int(round(HP.DOWNSAMPLE_RATIO * n_pos)))
    df_target0 = (
        df_target0
        .with_row_count()
        .filter(pl.col('row_nr').is_in(random.sample(range(n_neg_total), n_neg_keep)))
        .drop('row_nr')
    )

# Merge the classes (balanced or not) and shuffle rows to break ordering.
df_keep = (
    pl.concat([df_target1, df_target0], how='vertical')
    .select(pl.all().shuffle(seed=RANDOM_SEED))
)

# --- Tabular metadata preparation ---
# Age has missing values: we impute with the mean and normalize linearly to
# [0, 1], helpful for numerical stability of the tabular network.
if HP.AGE_IMPUTE == 'median':
    age_fill = float(df_keep.get_column('age').median())
else:
    age_fill = float(round(df_keep.get_column('age').mean()))

age_min = df_keep.get_column('age').min()
age_max = df_keep.get_column('age').max()
age_mean = float(df_keep.get_column('age').mean())
age_std = float(df_keep.get_column('age').std()) or 1.0

# Imputation
df_keep = df_keep.with_columns(
    pl.when(pl.col('age') == None)
    .then(age_fill)
    .otherwise(pl.col('age'))
    .alias('age')
)

# Normalization
if HP.AGE_NORM.lower() == 'zscore':
    df_keep = df_keep.with_columns(((pl.col('age') - age_mean) / age_std).alias('age'))
else:
    df_keep = df_keep.with_columns(((pl.col('age') - age_min) / (age_max - age_min)).alias('age'))

# One-hot encode categorical columns defined in HP
df_keep = df_keep.to_dummies(HP.CAT_COLS)

# Add a "trainvalid" column partitioned by patient to avoid the same individual
# appearing in both sides of the split. `GroupShuffleSplit` preserves the
# desired ratio (80/20, defined by TRAIN_SPLIT_RATIO) and mirrors the clinical
# practice of evaluating each patient in only one set, minimizing contralateral
# leakage.
patients = df_keep.get_column('patient_id').to_list()
gss = GroupShuffleSplit(
    n_splits=1,
    train_size=TRAIN_SPLIT_RATIO,
    random_state=RANDOM_SEED,
)
train_idx, _ = next(gss.split(np.zeros(len(patients)), groups=patients))
train_patients = list(
    set(df_keep.get_column('patient_id').take(list(train_idx)).to_list())
)

df_keep = df_keep.with_columns(
    pl.when(pl.col('patient_id').is_in(train_patients))
    .then('train')
    .otherwise('valid')
    .alias('trainvalid')
)
    
# Build the PNG filename (format `{patient_id}_{image_id}.png`).
# This step links tabular metadata to the image file.
df_keep = df_keep\
    .with_columns(pl.lit('_').alias('underscore'))\
    .with_columns(
        pl.concat_str(
            [
                pl.col('patient_id'),
                pl.col('underscore'),
                pl.col('image_id')
            ]
        ).alias('fname')
    ).drop('underscore')

df_train_meta = df_keep.filter(pl.col('trainvalid') == 'train')
df_valid_meta = df_keep.filter(pl.col('trainvalid') == 'valid')

train_fnames = df_train_meta.get_column('fname')
valid_fnames = df_valid_meta.get_column('fname')

train_labels = df_train_meta.get_column('cancer').to_numpy()
valid_labels = df_valid_meta.get_column('cancer').to_numpy()

print('Training Metadata Characteristics')
df_train_meta.glimpse()

print('Validation Metadata Characteristics')
df_valid_meta.glimpse()

# Destination directories where balanced copies converted from the DICOMs will be saved.
# On Kaggle notebooks, /kaggle/working is writable, but `--output-dir` lets you
# customize the folder when running locally.
# Discussion: saving balanced copies avoids hitting the original directories
# every epoch, reducing I/O bottlenecks on Kaggle. In orchestrated pipelines we
# would use cached data loaders or dedicated datastores.
train_dir = TRAIN_OUTPUT_DIR
valid_dir = VALID_OUTPUT_DIR

# Ensure folders exist before saving images.
os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)

# ==================== DICOM to PNG Conversion (Using Refactored I/O Module) ====================
# Refactoring note: dicom_to_pil_rgb() is imported from mammography.io.dicom module.
# This function encapsulates complex DICOM processing logic:
#   1. Reads DICOM file using pydicom
#   2. Applies robust percentile-based windowing
#   3. Normalizes to [0, 255] uint8 range
#   4. Replicates single-channel to RGB for compatibility with pretrained models
# By centralizing this logic, we ensure consistent preprocessing across the pipeline.

for fnames, out_dir in [(train_fnames, train_dir), (valid_fnames, valid_dir)]:
    for file in fnames:
        patient_id, image_id = file.split("_", 1)
        dcm_path = os.path.join(dicom_dir, patient_id, f"{image_id}.dcm")
        img = dicom_to_pil_rgb(dcm_path)  # Handles DICOM → PIL conversion with windowing
        img.save(os.path.join(out_dir, f"{file}.png"))

# ==================== Dataset Creation (Using Refactored Dataset Module) ====================
# Refactoring note: MammographyDataset is imported from cancer_dataset module.
# This PyTorch Dataset class handles:
#   - Loading PNG images from disk
#   - Applying transforms
#   - Returning (image, label) tuples
# Previously, dataset logic was mixed with the main script. Now it's a reusable component.

# Minimal transform pipeline: convert PIL RGB -> tensor (C×H×W, float [0, 1]).
# Discussion: we keep only `ToTensor` to isolate the effect of balancing.
# In real clinical applications, geometric augmentations would be validated
# with the clinical team to ensure anatomical plausibility.
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Instantiate train/validation datasets pointing to the previously built
# balanced directories.
train_dataset = MammographyDataset(
    meta_df=df_train_meta,
    img_dir=train_dir,
    transform=transform,
)

valid_dataset = MammographyDataset(
    meta_df=df_valid_meta,
    img_dir=valid_dir,
    transform=transform,
)

# DataLoaders with batch size 64 and *pin_memory* enabled when CUDA is present.
# Discussion: `num_workers=2` is conservative for RAM-constrained environments.
# Adjusting this number directly impacts read throughput.
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=DATALOADER_WORKERS, pin_memory=torch.cuda.is_available())
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False, num_workers=DATALOADER_WORKERS, pin_memory=torch.cuda.is_available())

# Quick visualization to check contrast, channels, and labels before training.
# Refactoring note: preview_transformed_samples() is from cancer_plots module
preview_transformed_samples(train_dataset)

# ==================== Model Creation (Using Refactored Models Module) ====================
# Refactoring note: MammographyModel is imported from cancer_models module.
# This class encapsulates the ResNet50 architecture with:
#   - Pretrained ImageNet weights
#   - Modified first conv layer for single-channel mammography input
#   - Custom final layer for binary classification
# By separating model definition from training logic, we can easily:
#   - Test different architectures
#   - Reuse the model in other scripts
#   - Modify the architecture without touching training code

model = MammographyModel()

# ------------------------- Loss Function and Optimizer -------------------------
# Binary (BCELoss) because the last layer applies sigmoid. Classic SGD optimizer
# with momentum/weight decay to illustrate the traditional setup.
# Discussion: BCELoss + sigmoid is didactic, but in production we'd prefer
# `BCEWithLogitsLoss` for numerical stability and the ability to calibrate the
# clinical decision with logits.
loss_fn = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

# -------------------------------- Training Loop -------------------------------

# Select GPU if available; otherwise use CPU. The print helps track runs in Kaggle logs.
# Discussion: keeping `to(device)` explicit reinforces the PyTorch pattern for
# those moving from CPU-only notebooks. Device visibility avoids performance
# surprises in detached sessions.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f'Using device: {device}')

# Number of epochs and buffers to store metrics per epoch.
npochs = 25
train_loss, valid_loss = [], []
train_accuracy, valid_accuracy = [], []
train_sensitivity, valid_sensitivity = [], []
train_specificity, valid_specificity = [], []

# ==================== Training Loop (Using Refactored Trainer Module) ====================
# Refactoring note: get_sens_spec() is imported from cancer_trainer module.
# While this script contains a manual training loop for educational purposes, the
# cancer_trainer module provides higher-level functions (train_one_epoch, evaluate,
# fit_classifier) that encapsulate common training patterns. For production workflows,
# prefer using those functions. This manual loop is kept for tutorial clarity.
#
# The trainer module provides:
#   - train_one_epoch(): Handles one epoch of training with gradient updates
#   - evaluate(): Runs validation without gradients
#   - get_sens_spec(): Computes clinical metrics (sensitivity/specificity)
#   - fit_classifier(): High-level training loop with automatic metric tracking

# Main training loop: `model.train()` enables dropout/BN, and we accumulate
# losses and predictions to compute metrics at the end of each epoch.
for epoch in range(npochs):

    # Training section
    model.train()
    running_loss = 0.0
    train_preds, train_labels = [], []

    for batch, (img, label) in enumerate(train_loader):
        img = img.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        outputs = model(img)

        loss = loss_fn(outputs.view(-1), label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Store binary predictions (threshold 0.5) for aggregated metrics.
        preds = outputs.detach().cpu().view(-1).numpy().round()
        train_preds.extend(preds)
        train_labels.extend(label.cpu().numpy())

        if batch%5 == 0:
            print(f'epoch {epoch + 1}  batch {batch + 1}  train loss: {loss.item():10.8f}')

    # Training metrics: sensitivity/specificity + mean epoch loss.
    train_sens, train_spec = get_sens_spec(train_labels, train_preds)
    train_sensitivity.append(train_sens)
    train_specificity.append(train_spec)
    avg_train_loss = running_loss/len(train_loader)
    train_loss.append(avg_train_loss)
    train_accuracy.append(accuracy_score(train_labels, train_preds))
    
    # Eval section
    model.eval()
    running_loss = 0.0
    valid_preds, valid_labels = [], []

    with torch.no_grad():
        for batch, (img, label) in enumerate(valid_loader):
            img = img.to(device)
            label = label.to(device)

            outputs = model(img)
            loss = loss_fn(outputs.view(-1), label)
            
            running_loss += loss.item()
            
            # Store predictions to compute validation metrics.
            preds = outputs.detach().cpu().view(-1).numpy().round()
            valid_preds.extend(preds)
            valid_labels.extend(label.cpu().numpy())

    # Validation metrics mirror the training ones to compare generalization.
    # Discussion: comparing train/validation pairs per epoch helps spot early
    # overfitting — especially important when using few negatives and the model
    # tends to memorize patterns.
    valid_sens, valid_spec = get_sens_spec(valid_labels, valid_preds)
    valid_sensitivity.append(valid_sens)
    valid_specificity.append(valid_spec)
    avg_valid_loss = running_loss/len(valid_loader)
    valid_loss.append(avg_valid_loss)
    valid_accuracy.append(accuracy_score(valid_labels, valid_preds))
            
    print('---------------------------------------------------------------------------------')
    print(f'Metrics for epoch {epoch + 1}')
    print(f'Accuracy     train: {train_accuracy[epoch]}  valid: {valid_accuracy[epoch]}')
    print(f'Sensitivity  train: {train_sensitivity[epoch]}  valid: {valid_sensitivity[epoch]}')
    print(f'Specificity  train: {train_specificity[epoch]}  valid: {valid_specificity[epoch]}')
    print('---------------------------------------------------------------------------------')

# ----------------------------------- Evaluation ----------------------------------

def create_metric_plot(train_vals: List[float], valid_vals: List[float], metric_name: str, title: str, npochs: int):
    """Helper to create a line plot for train/valid metrics."""
    epoch = list(range(1, npochs + 1)) * 2
    set_type = ['Train'] * npochs + ['Valid'] * npochs
    values = train_vals + valid_vals
    df = pl.DataFrame({'epoch': epoch, 'set_type': set_type, metric_name: values})
    return (
        ggplot(df) +
        geom_line(aes(x='epoch', y=metric_name, color='set_type'), size=2) +
        labs(x='Epoch', y=metric_name.capitalize(), title=title, color='') +
        scale_x_continuous(breaks=list(range(1, npochs + 1))) +
        theme(
            plot_title=element_text(hjust=0.5, face='bold'),
            legend_position='top',
            axis_line_y=element_line(size=1),
            axis_line_x=element_line(size=1),
        )
    )

# Create plots for all metrics
plt_loss = create_metric_plot(train_loss, valid_loss, 'loss', 'Loss Tracking', npochs)
plt_acc = create_metric_plot(train_accuracy, valid_accuracy, 'accuracy', 'Accuracy Tracking', npochs)
plt_sens = create_metric_plot(train_sensitivity, valid_sensitivity, 'sensitivity', 'Sensitivity Tracking', npochs)
plt_spec = create_metric_plot(train_specificity, valid_specificity, 'specificity', 'Specificity Tracking', npochs)

# Group the four charts in a bunch for joint visualization.
bunch = GGBunch()
bunch.add_plot(plt_loss, 0, 0, 800, 400)
bunch.add_plot(plt_acc, 0, 410, 800, 400)
bunch.add_plot(plt_sens, 0, 820, 800, 400)
bunch.add_plot(plt_spec, 0, 1230, 800, 400)
bunch.show()

# ==================== Density Classifier (Demonstrating Full Modular Integration) ====================
# Refactoring note: The density classifier section demonstrates how all refactored modules
# work together in a cohesive pipeline. This is a complete example of using the modular
# architecture for a different classification task (breast density vs cancer detection).
#
# Module usage in this section:
#   - cancer_config: HP, seed_everything, find_best_data_dir, load_labels_dict
#   - cancer_dataset: MammoDicomDataset, dataset_summary, split_dataset, make_dataloader
#   - cancer_models: build_resnet50_classifier, resolve_device
#   - cancer_trainer: DensityHistoryEntry, train_one_epoch, evaluate, collect_predictions, fit_classifier
#   - cancer_plots: get_transforms
#
# Benefits of modular approach visible here:
#   1. Same modules used for different tasks (cancer detection + density classification)
#   2. No code duplication - datasets, models, training logic are reused
#   3. Clean separation allows this to be extracted into a separate script if needed

# ------------------------------- Utility Functions -------------------------------

def indices_from_subset(subset: Optional[Dataset]) -> Iterable[int]:
    """Recover original indices even when the set is a ``Subset``."""
    if subset is None:
        return []
    if isinstance(subset, Subset):
        return subset.indices
    return range(len(subset))


def history_to_dict(history: List[DensityHistoryEntry]) -> List[Dict[str, Optional[float]]]:
    """Transform the history (dataclasses) into a JSON-serializable structure."""
    return [
        {
            "epoch": h.epoch,
            "train_loss": h.train_loss,
            "train_acc": h.train_acc,
            "val_loss": h.val_loss,
            "val_acc": h.val_acc,
        }
        for h in history
    ]


# ------------------------------- Interface CLI -------------------------------

def parse_density_classifier_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Dedicated parser for the density classifier (standalone and reusable)."""
    parser = argparse.ArgumentParser(
        description="Treino de ResNet50 para classificação de densidade mamária (categorias 1–4)."
    )
    parser.add_argument("--data_dir", type=str, default="./archive", help="Diretório com subpastas por AccessionNumber.")
    parser.add_argument(
        "--csv_path",
        type=str,
        default="./classificacao.csv",
        help="CSV com colunas AccessionNumber/Classification.",
    )
    parser.add_argument("--out_dir", type=str, default="./outputs_classifier", help="Onde salvar pesos e relatórios.")
    parser.add_argument("--batch_size", type=int, default=8, help="Tamanho de batch para treino/validação.")
    parser.add_argument("--num_workers", type=int, default=2, help="Workers do DataLoader.")
    parser.add_argument("--epochs", type=int, default=10, help="Número de épocas de treino.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Taxa de aprendizado do AdamW.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay do AdamW.")
    parser.add_argument("--val_fraction", type=float, default=0.2, help="Fração de validação (0 desativa).")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Dispositivo de execução.",
    )
    parser.add_argument("--weights_path", type=str, default=None, help="Checkpoint ResNet50 prévio para iniciar o treino.")
    parser.add_argument("--avoid_download", action="store_true", help="Não baixa pesos caso não encontre localmente.")
    parser.add_argument("--torch_home", type=str, default=None, help="Diretório TORCH_HOME alternativo.")
    parser.add_argument("--freeze_backbone", action="store_true", help="Congela todas as camadas exceto a FC final.")
    parser.add_argument("--seed", type=int, default=42, help="Seed global para reprodutibilidade.")
    return parser.parse_args(argv)


def run_density_classifier_cli(argv: Optional[List[str]] = None) -> None:
    """Run the complete pipeline using the provided arguments.

    Refactoring note: This function demonstrates the power of the modular architecture.
    By composing functions from different modules, we build a complete ML pipeline in ~100 lines:

    Flow:
        1. Parse arguments and set random seed (cancer_config)
        2. Resolve computation device (cancer_models)
        3. Load data and labels (cancer_config, cancer_dataset)
        4. Create datasets and dataloaders (cancer_dataset)
        5. Build model (cancer_models)
        6. Train and evaluate (cancer_trainer)
        7. Save results (standard Python)

    Each step uses well-tested, reusable components. New pipelines can be built by
    mixing and matching these same modules with minimal code changes.
    """
    args = parse_density_classifier_args(argv)
    seed_everything(args.seed)

    device = resolve_device(args.device)
    print(f"[info] Usando dispositivo: {device}")

    data_dir = find_best_data_dir(args.data_dir)
    labels_map = load_labels_dict(args.csv_path)

    transform_model, _ = get_transforms()
    dataset = MammoDicomDataset(
        data_dir=data_dir,
        labels_by_accession=labels_map,
        exclude_class_5=True,
        include_unlabeled=False,
        transform=transform_model,
    )
    print(f"[info] Amostras disponíveis: {len(dataset)}")
    if len(dataset) == 0:
        raise RuntimeError("Nenhuma amostra disponível para treino (verifique diretórios e CSV).")

    counts = dataset_summary(dataset)
    print("[info] Distribuição por classe:", counts)

    train_subset, val_subset = split_dataset(dataset, args.val_fraction, args.seed)
    train_loader = make_dataloader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        device=device,
    )
    val_loader = None
    if val_subset is not None:
        val_loader = make_dataloader(
            val_subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            device=device,
        )

    model = build_resnet50_classifier(
        device,
        num_classes=4,
        weights_path=args.weights_path,
        avoid_download=args.avoid_download,
        torch_home=args.torch_home,
        freeze_backbone=args.freeze_backbone,
    )

    print("[info] Iniciando treino...")
    model, history = fit_classifier(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    history_path = os.path.join(args.out_dir, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_to_dict(history), f, indent=2, ensure_ascii=False)
    print(f"[ok] Histórico salvo em {history_path}")

    model_path = os.path.join(args.out_dir, "resnet50_density_classifier.pth")
    torch.save(model.state_dict(), model_path)
    print(f"[ok] Pesos salvos em {model_path}")

    full_loader = make_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
    )
    prediction_records = collect_predictions(model, full_loader, device)
    df_preds = pd.DataFrame(prediction_records)

    if not df_preds.empty:
        val_indices = set(indices_from_subset(val_subset))
        df_preds["split"] = np.where(df_preds["idx"].isin(val_indices), "val", "train")

        report = classification_report(
            df_preds["true_class"],
            df_preds["predicted_class"],
            labels=[1, 2, 3, 4],
            output_dict=True,
            zero_division=0,
        )
        cm = confusion_matrix(df_preds["true_class"], df_preds["predicted_class"], labels=[1, 2, 3, 4])

        metrics_path = os.path.join(args.out_dir, "classification_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({"classification_report": report, "confusion_matrix": cm.tolist()}, f, indent=2, ensure_ascii=False)
        print(f"[ok] Métricas salvas em {metrics_path}")

        preds_path = os.path.join(args.out_dir, "predictions.csv")
        df_preds.sort_values("idx").to_csv(preds_path, index=False)
        print(f"[ok] Previsões salvas em {preds_path}")
    else:
        print("[aviso] Nenhuma previsão válida gerada (todas as amostras sem rótulo?).")
