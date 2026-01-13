# Mammography Pipelines

Consolidated mammography pipeline with a single CLI, modular Python package, and a scientific article workspace. The focus is breast density classification (BI-RADS A–D) plus reproducible reporting and research artifacts.

## Entrypoints

- `mammography` (primary CLI entrypoint)

## Installation

```bash
# Recommended: reproducible install via uv lockfile
pip install uv
uv sync --frozen

# Legacy (pip)
# pip install -r requirements.txt
```

## Docker

```bash
docker build -t mammography-pipelines .
docker run --rm -it -v "$PWD":/app -w /app mammography-pipelines
```

## Quick Start

```bash
# Show CLI help
mammography --help

# Interactive wizard
mammography wizard

# Embeddings
mammography embed -- \
  --csv classificacao.csv \
  --dicom-root archive \
  --outdir outputs/embeddings_resnet50

# Baselines classicos (embeddings)
mammography embeddings-baselines -- \
  --embeddings-dir outputs/embeddings_resnet50 \
  --outdir outputs/embeddings_baselines

# Treinamento de densidade
mammography train-density -- \
  --csv classificacao.csv \
  --dicom-root archive \
  --outdir outputs/mammo_efficientnetb0_density \
  --epochs 8 \
  --arch resnet50

# Visualization
mammography visualize -- \
  --input outputs/embeddings_resnet50/features.npy \
  --outdir outputs/visualizations

# Inference
mammography inference -- \
  --checkpoint outputs/mammo_efficientnetb0_density/results_1/best_model.pt \
  --input archive \
  --output outputs/preds.csv

# Augmentation
mammography augment -- \
  --source-dir archive \
  --output-dir outputs/augmented \
  --num-augmentations 2

# Report packaging (Article integration)
mammography report-pack --run outputs/mammo_efficientnetb0_density/results_1
```

## CLI Overview

The CLI is built around two core workflows plus utilities:

- **`embed`**: Extract ResNet/EfficientNet embeddings and optional PCA/t-SNE/UMAP analysis.
- **`embeddings-baselines`**: Compare embeddings against classical descriptors (LogReg/SVM/RF).
- **`train-density`**: Train density classifiers (EfficientNetB0/ResNet50) with cache modes and reporting artifacts.
- **`visualize`**: Generate plots from embeddings or run directories.
- **`inference`**: Run checkpointed inference over image folders or single files.
- **`augment`**: Generate augmented samples from a directory.
- **`report-pack` / `eval-export`**: Prepare figures/tables for the scientific article.
- **`wizard`**: Interactive, step-by-step menu for the core workflows.
- **`eda-cancer`**: RSNA Breast Cancer Detection exploratory pipeline (CSV/PNG/DICOM inputs).

Common flags across the CLI:
- `--outdir` for outputs
- `--dicom-root` for DICOM roots
- `--cache-mode` for dataset caching

Dataset presets:
- `archive`: DICOMs + `classificacao.csv`
- `mamografias`: PNGs por subpasta com `featureS.txt`
- `patches_completo`: PNGs na raiz com `featureS.txt`

See `CLI_CHEATSHEET.md` for command matrices and common recipes.

## RSNA (Cancer) EDA

The `eda-cancer` subcommand runs the internal RSNA Breast Cancer Detection pipeline.

```bash
mammography eda-cancer -- \
  --csv-dir /path/to/rsna \
  --png-dir /path/to/rsna-256-pngs \
  --dicom-dir /path/to/rsna-dicoms \
  --outdir outputs/rsna_eda
```

## Repository Structure

- `src/mammography/`: Main package (data, models, training, tools).
- `src/mammography/commands/`: Internal CLI command modules (invoked by `mammography`).
- `configs/`: YAML presets for datasets and training.
- `Article/`: Scientific article and build assets.
- `tests/`: Unit, integration, performance, and smoke tests.

## Scientific Article Workflow

The article lives in `Article/` and integrates with pipeline outputs via:

- `mammography report-pack` (CLI) or `python -m mammography.tools.report_pack`
- `mammography data-audit` (CLI) or `python -m mammography.tools.data_audit`

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
