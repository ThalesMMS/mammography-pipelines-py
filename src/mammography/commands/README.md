# commands

## Purpose
Internal command handlers behind the top-level `mammography` CLI. Each module parses subcommand
arguments and orchestrates lower-level packages that do the real work.

## Entry Points and Key Modules
- These modules are dispatched from `mammography/cli.py`; they are not usually invoked directly by
path.
- Most command modules expose `parse_args()` and `main()` so the root CLI can forward unknown
arguments without losing backward compatibility.

### Key Files
- `augment.py`: Augment images from a directory and save to a new folder.
- `automl.py`: Automated ML pipeline for breast density classification with AutoML features.
- `batch_inference.py`: Batch inference with progress tracking and checkpoint/resume capabilities.
- `cancer_config.py`: Configuration and hyperparameters for breast cancer detection pipeline.
- `compare_models.py`: Compare Models CLI — Compare multiple trained models with side-by-side
metrics and statistical tests.
- `cross_validate.py`: K-fold cross-validation for breast density classification.
- `eda_cancer.py`: Educational notebook-style pipeline for RSNA Breast Cancer Detection.
- `embeddings_baselines.py`: Baseline classifiers for mammography embeddings.
- `eval_export.py`: Export evaluation artifacts and register them in MLflow/registry.
- `explain.py`: Explainability CLI — Generate GradCAM heatmaps and attention maps to explain model
predictions.
- `extract_features.py`: Embedding extraction plus optional PCA/t-SNE/UMAP/clustering analysis.
- `inference.py`: Run inference with a trained EfficientNetB0/ResNet50 checkpoint.
- `label_density.py`: Wrapper to launch the density classifier Streamlit UI.
- `label_patches.py`: Wrapper to launch the patch marking Streamlit UI.
- `preprocess.py`: Preprocess mammography datasets with normalization, resizing, and format
conversion.
- `train.py`: Train EfficientNetB0/ResNet50/ViT for breast density with optional caches and AMP.
- `tune.py`: Automated hyperparameter optimization for EfficientNetB0/ResNet50 using Optuna.
- `visualize.py`: Visualization CLI — Generate t-SNE, heatmaps, scatterplots and more from
embeddings.
- `web.py`: Wrapper to launch the web-based UI dashboard Streamlit app.

## How It Fits into the Pipeline
- Translates user-facing CLI intent into calls across `data/`, `models/`, `training/`, `tools/`, and
`vis/`.
- Defines the practical workflow surface of the repository: training, extraction, visualization,
inference, preprocessing, audits, tuning, and UI launchers.
- Keeps the CLI surface stable while delegating reusable logic to lower-level packages.

## Inputs and Outputs
- Inputs: CLI flags, config files, dataset presets, output directories, checkpoints, and run
locations.
- Outputs: training runs, exported artifacts, reports, plots, audits, predictions, and launched UI
sessions depending on the command.

## Dependencies
- Internal: [`mammography`](../README.md), [`apps`](../apps/README.md), [`data`](../data/README.md),
[`models`](../models/README.md), [`tools`](../tools/README.md), [`training`](../training/README.md),
[`tuning`](../tuning/README.md), [`vis`](../vis/README.md).
- External: `argparse`, `torch`, `pandas`, `numpy`, `matplotlib`, `Pillow`.

## Extension and Maintenance Notes
- Keep argument names synchronized with `TrainConfig` and related config models so CLI parsing and
validation do not drift.
- Command modules should orchestrate and validate, not absorb core training or visualization logic
that belongs in the reusable packages.
- Because `cli.py` forwards unknown args, be careful when removing or renaming flags; backward
compatibility matters here more than in the lower-level modules.

## Related Directories
- [`mammography`](../README.md): Primary package root for the mammography pipelines.
- [`data`](../data/README.md): Source of truth for dataset ingestion.
- [`training`](../training/README.md): Core training and validation logic for supervised mammography
models.
- [`tools`](../tools/README.md): Support utilities for reporting, registries, data audits, and
artifact packaging.
- [`vis`](../vis/README.md): Visualization and explainability package for the mammography pipelines.
