# data

## Purpose
Source of truth for dataset ingestion. This package discovers dataset formats, validates CSV
schemas, resolves paths, builds splits, and exposes PyTorch-compatible dataset objects.

## Entry Points and Key Modules
- This package is imported by training, inference, preprocessing, and cross-validation flows rather
than run directly.

### Key Files
- `cancer_dataset.py`: Dataset classes for RSNA Breast Cancer Detection.
- `csv_loader.py`: Loads dataset rows from CSVs or presets, validates schemas, normalizes labels,
and resolves image paths and accessions.
- `dataset.py`: Density dataset implementation with caching, optional embedding lookup,
normalization handling, and robust collation helpers.
- `format_detection.py`: Dataset format auto-detection for mammography pipelines.
- `splits.py`: Group-aware split utilities for train/validation/test and k-fold workflows, with
leakage checks and class-balance safeguards.

## How It Fits into the Pipeline
- Normalizes multiple dataset layouts into a consistent tabular representation.
- Provides split logic and dataset classes that the training stack can rely on.
- Absorbs most path validation and auto-detection logic so commands stay focused on workflow
orchestration.

## Inputs and Outputs
- Inputs: CSV files, preset names, archive-style directory trees, DICOM roots, image paths, and
optional split CSVs.
- Outputs: normalized DataFrames, validated schemas, split tuples, and `Dataset`/`DataLoader`-ready
objects.

## Dependencies
- Internal: [`io`](../io/README.md), [`preprocess`](../preprocess/README.md),
[`training`](../training/README.md), [`utils`](../utils/README.md).
- External: `pandas`, `numpy`, `pandera` or fallback schema helpers, `pydicom`, `torch`.

## Extension and Maintenance Notes
- Treat column names such as `image_path`, `professional_label`, `accession`, and `view` as
contract-level fields; downstream code assumes them.
- Format auto-detection should stay conservative and transparent because it influences loader
behavior for many CLI commands.
- Keep split code group-aware where possible so patient or accession leakage does not creep into
training and validation workflows.

## Related Directories
- [`io`](../io/README.md): Low-level image I/O helpers, especially for DICOM handling.
- [`preprocess`](../preprocess/README.md): Image preprocessing abstractions for mammography data.
- [`training`](../training/README.md): Core training and validation logic for supervised mammography
models.
- [`commands`](../commands/README.md): Internal command handlers behind the top-level `mammography`
CLI.
