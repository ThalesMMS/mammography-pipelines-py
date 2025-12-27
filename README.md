# Mammography Pipelines

Consolidated mammography pipeline with a single CLI, modular Python package, and a scientific article workspace. The focus is breast density classification (BI-RADS A–D) plus reproducible reporting and research artifacts.

## Entrypoints

- `mammography` (primary CLI entrypoint)

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Show CLI help
mammography --help

# Interactive wizard
mammography wizard

# Stage 1: embeddings
mammography embed -- \
  --csv classificacao.csv \
  --dicom-root archive \
  --outdir outputs/stage1

# Stage 2: density training
mammography train-density -- \
  --csv classificacao.csv \
  --dicom-root archive \
  --outdir outputs/stage2 \
  --epochs 8 \
  --arch resnet50

# Visualization
mammography visualize -- \
  --input outputs/stage1/features.npy \
  --outdir outputs/visualizations

# Inference
mammography inference -- \
  --checkpoint outputs/stage2/results_1/best_model.pt \
  --input archive \
  --output outputs/preds.csv

# Augmentation
mammography augment -- \
  --source-dir archive \
  --output-dir outputs/augmented \
  --num-augmentations 2

# Report packaging (Article integration)
mammography report-pack --run outputs/stage2/results_1
```

## CLI Overview

The CLI is built around two stages plus utilities:

- **Stage 1 — `embed`**: Extract ResNet/EfficientNet embeddings and optional PCA/t-SNE/UMAP analysis.
- **Stage 2 — `train-density`**: Train density classifiers (EfficientNetB0/ResNet50) with cache modes and reporting artifacts.
- **`visualize`**: Generate plots from embeddings or run directories.
- **`inference`**: Run checkpointed inference over image folders or single files.
- **`augment`**: Generate augmented samples from a directory.
- **`report-pack` / `eval-export`**: Prepare figures/tables for the scientific article.
- **`wizard`**: Interactive, step-by-step menu for the core workflows.

Common flags across the CLI and scripts:
- `--outdir` for outputs
- `--dicom-root` for DICOM roots
- `--cache-mode` for dataset caching

Dataset presets:
- `archive`: DICOMs + `classificacao.csv`
- `mamografias`: PNGs por subpasta com `featureS.txt`
- `patches_completo`: PNGs na raiz com `featureS.txt`

See `CLI_CHEATSHEET.md` for command matrices and common recipes.

## Repository Structure

- `src/mammography/`: Main package (data, models, training, tools).
- `scripts/`: Script entrypoints (embed, train, visualize, inference, augment, labeling, EDA).
- `configs/`: YAML presets for datasets and training.
- `Article/`: Scientific article and build assets.
- `tools/`: Reporting/audit helpers (also available as `mammography.tools`).
- `tests/`: Unit, integration, performance, and smoke tests.

## Scientific Article Workflow

The article lives in `Article/` and integrates with pipeline outputs via:

- `mammography report-pack` (CLI) or `python -m mammography.tools.report_pack`
- `python -m mammography.tools.data_audit` for dataset manifest/audit artifacts

See `Article/README.md` for LaTeX build instructions.

## Testing

Some tests require local datasets (DICOM archives, RSNA folders). For dataset-free smoke checks, run:

```bash
python -m pytest \
  tests/unit/test_dicom_validation.py \
  tests/unit/test_dimensionality_reduction.py \
  tests/unit/test_evaluation_metrics.py \
  tests/unit/test_clustering_algorithms.py \
  tests/test_cache_mode.py \
  tests/test_dataset_transforms.py
```

## Disclaimer

⚠️ This is an educational research project. It must NOT be used for clinical or medical diagnostic purposes.
