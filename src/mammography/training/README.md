# training

## Purpose
Core training and validation logic for supervised mammography models. This package contains reusable
epoch loops, classifier-focused helpers, and cross-validation orchestration.

## Entry Points and Key Modules
- Training commands call into these modules; the package itself is designed for reuse, not direct
CLI execution.

### Key Files
- `cancer_trainer.py`: Training and evaluation utilities for breast cancer detection models.
- `cv_engine.py`: K-fold training orchestrator that reuses the core training engine and writes
aggregated fold metrics.
- `engine.py`: Reusable train/validate loops plus artifact-saving helpers for predictions, GradCAM
batches, histories, and metrics figures.

## How It Fits into the Pipeline
- Implements the mechanics of fitting models, validating them, saving checkpoints, and aggregating
metrics.
- Separates reusable training primitives from command-layer argument parsing and output-directory
management.
- Owns the cross-validation engine used for fold-based evaluation workflows.

## Inputs and Outputs
- Inputs: datasets or dataloaders, model instances, optimization settings, and validated
`TrainConfig` values.
- Outputs: checkpoints, histories, prediction exports, metrics figures, and aggregated cross-
validation summaries.

## Dependencies
- Internal: [`mammography`](../README.md), [`data`](../data/README.md),
[`models`](../models/README.md), [`tools`](../tools/README.md), [`utils`](../utils/README.md),
[`vis`](../vis/README.md).
- External: `torch`, `pandas`, `numpy`, `matplotlib`, `scikit-learn`.

## Extension and Maintenance Notes
- Keep artifact names and metric keys stable because registry and reporting code in `tools/` often
assume specific filenames.
- `engine.py` should stay generic and reusable; task-specific branching belongs in higher-level
trainer modules or the commands layer.
- Cross-validation behavior should reuse the same model and dataset contracts as single-run training
so results remain comparable.

## Related Directories
- [`data`](../data/README.md): Source of truth for dataset ingestion.
- [`models`](../models/README.md): Trainable model architectures and builders for density and
cancer-related mammography tasks.
- [`tuning`](../tuning/README.md): Hyperparameter optimization utilities built around Optuna and
related tuning helpers.
- [`tools`](../tools/README.md): Support utilities for reporting, registries, data audits, and
artifact packaging.
- [`vis`](../vis/README.md): Visualization and explainability package for the mammography pipelines.
