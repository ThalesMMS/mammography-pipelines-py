# utils

## Purpose
Cross-cutting helpers shared across the repository. This package collects runtime setup, device
detection, normalization, reproducibility, fallbacks for optional dependencies, statistics, and
wizard helpers.

## Entry Points and Key Modules
- These modules are support code consumed by many packages; they do not define a standalone workflow
of their own.

### Key Files
- `common.py`: General runtime helpers for seeding, logging, device resolution, path incrementing,
and reproducibility metadata.
- `device_detection.py`: Device detection and configuration utilities for the mammography pipelines.
- `normalization.py`: Normalization-statistic utilities for computing, validating, and applying
dataset-level z-score normalization.
- `numpy_warnings.py`: Helpers for taming noisy NumPy warnings in numerical routines.
- `pandera_fallback.py`: Minimal pandera fallback used when the dependency is unavailable.
- `patient.py`: Patient model for patient-level data management and split assignment.
- `pydantic_fallback.py`: Minimal fallback helpers when pydantic is unavailable.
- `reproducibility.py`: Reproducibility utilities for deterministic training and evaluation.
- `smart_defaults.py`: Smart defaults calculator based on hardware detection and dataset
characteristics.
- `statistics.py`: Statistical utilities for cross-validation and model evaluation.
- `wizard_helpers.py`: Shared prompt and help-formatting helpers used by the interactive CLI wizard
and guided workflows.

## How It Fits into the Pipeline
- Holds low-level helpers that multiple workflows need, without tying them to one command or model
family.
- Provides compatibility shims for optional dependencies so the rest of the code can degrade more
gracefully.
- Centralizes concerns such as seeding, device selection, and metric aggregation that would
otherwise be repeated widely.

## Inputs and Outputs
- Inputs: runtime environment details, config values, numeric samples, and validation requests from
higher-level packages.
- Outputs: normalized settings, seeded random states, fallback model classes, and utility
calculations used across the codebase.

## Dependencies
- Internal: [`data`](../data/README.md), [`training`](../training/README.md),
[`tuning`](../tuning/README.md).
- External: `numpy`, `scipy`, `subprocess`, `platform`.

## Extension and Maintenance Notes
- Keep this package general-purpose; if a helper only serves one workflow, it usually belongs closer
to that workflow instead of here.
- Fallback modules should match the subset of third-party APIs that callers expect, otherwise
failures become harder to diagnose than a hard dependency error.
- Utility growth is easy to abuse; revisit organization if large thematic clusters emerge instead of
continuing to add unrelated helpers to one file.

## Related Directories
- [`data`](../data/README.md): Source of truth for dataset ingestion.
- [`training`](../training/README.md): Core training and validation logic for supervised mammography
models.
- [`tuning`](../tuning/README.md): Hyperparameter optimization utilities built around Optuna and
related tuning helpers.
- [`mammography`](../README.md): Primary package root for the mammography pipelines.
