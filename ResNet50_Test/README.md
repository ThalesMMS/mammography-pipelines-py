# ResNet50_Test: Breast Density Exploration

Warning: Educational research only. Do not use for clinical or diagnostic purposes.

## Overview

This repository explores breast density patterns in mammography with ResNet-50 embeddings. It includes data preparation, feature extraction, clustering, reporting, and documentation. The code is geared toward reproducible research and careful handling of medical data.

## Repository Layout

```
ResNet50_Test/
├── archive/    # Raw DICOM data (local only; not committed)
├── article/    # Manuscript workspace and assets
├── configs/    # YAML configuration profiles
├── data/       # Processed datasets and artifacts
├── docs/       # Documentation
├── reports/    # Experiment summaries and visuals
├── results/    # Generated outputs per run
├── scripts/    # Utility and compliance scripts
├── src/        # Python source (pipeline, CLI, utilities)
├── specs/      # Project specifications
└── tests/      # Unit, integration, contract, performance
```

## Requirements

- Python 3.11+
- Optional GPU: CUDA (NVIDIA) or MPS (Apple Silicon)
- Git; Docker optional

## Setup

```bash
git clone <repository-url>
cd ResNet50_Test
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
mkdir -p archive data/raw data/processed results reports docs
```

Place DICOM studies under `archive/`, one folder per patient, to keep splits at the patient level.

## CLI

Subcommands (see `src/cli/`):
- `preprocess` — validate and preprocess DICOM studies
- `embed` — extract and cache ResNet-50 embeddings
- `cluster` — dimensionality reduction and clustering
- `analyze` — generate reports and visuals

Each command supports `--help`. Config lives in `configs/` and can be overridden via flags.

## Quality

- `make format` — Black
- `make lint` — Ruff
- `make type-check` — MyPy
- `make test` — Pytest with coverage
- `make quality-check` — formatting, linting, typing

Coverage HTML is written to `htmlcov/`. Ruff JSON report is `ruff-report.json`.

## Compliance

- Do not commit medical data; keep `archive/` local.
- Split datasets at patient level to avoid leakage.
- Include the research disclaimer in outputs.

## GPU

- Apple Silicon: MPS detection with CPU fallback.
- NVIDIA: CUDA detection with CPU fallback when unavailable.

For a quick start, see `QUICKSTART.md` and the specs in `specs/`.
