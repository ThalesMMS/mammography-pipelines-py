# mammography

## Purpose
Primary package root for the mammography pipelines. It houses the CLI entrypoint, validated
configuration models, the interactive wizard, and the subpackages that implement data loading,
modeling, training, evaluation, visualization, and operator UIs.

## Entry Points and Key Modules
- `mammography` -> `cli.py` dispatches top-level subcommands and forwards unknown args to command
handlers.
- `python -m mammography.cli` provides the same CLI surface for environments without script
installation.
- `wizard.py` exposes the guided setup flow used by `mammography wizard`.

### Key Files
- `cli.py`: CLI entrypoint for the mammography pipelines.
- `config.py`: Validated configuration models and centralized hyperparameter defaults for training,
extraction, inference, and related workflows.
- `wizard.py`: Interactive CLI wizard that gathers workflow choices and translates them into
command-ready arguments.

### Subdirectories
- [`analysis/`](analysis/README.md): Numerical post-processing helpers for embedding exploration.
- [`apps/`](apps/README.md): Umbrella package for operator-facing applications.
- [`clustering/`](clustering/README.md): Structured clustering layer for embedding experiments.
- [`commands/`](commands/README.md): Internal command handlers behind the top-level `mammography`
CLI.
- [`data/`](data/README.md): Source of truth for dataset ingestion.
- [`eval/`](eval/README.md): Evaluation helpers for unsupervised embedding analysis.
- [`features/`](features/README.md): Small feature-extraction package centered on a ResNet50-based
extractor.
- [`io/`](io/README.md): Low-level image I/O helpers, especially for DICOM handling.
- [`models/`](models/README.md): Trainable model architectures and builders for density and cancer-
related mammography tasks.
- [`pipeline/`](pipeline/README.md): High-level end-to-end orchestration helpers that chain data
loading, preprocessing, model execution, clustering, evaluation, and visualization into a single
pipeline object.
- [`preprocess/`](preprocess/README.md): Image preprocessing abstractions for mammography data.
- [`tools/`](tools/README.md): Support utilities for reporting, registries, data audits, and
artifact packaging.
- [`tracking/`](tracking/README.md): Local experiment tracking support for environments where a
lightweight SQLite-backed registry is preferable to or combined with MLflow.
- [`training/`](training/README.md): Core training and validation logic for supervised mammography
models.
- [`tuning/`](tuning/README.md): Hyperparameter optimization utilities built around Optuna and
related tuning helpers.
- [`utils/`](utils/README.md): Cross-cutting helpers shared across the repository.
- [`vis/`](vis/README.md): Visualization and explainability package for the mammography pipelines.

## How It Fits into the Pipeline
- Defines the public package boundary that other tooling imports.
- Centralizes config validation and defaults before work reaches command modules.
- Acts as the bridge between the user-facing CLI and the lower-level training, inference, and
reporting packages.

## Inputs and Outputs
- Inputs: CLI arguments, optional YAML/JSON config files, environment-specific path hints, and
interactive answers from the wizard.
- Outputs: validated config objects, resolved command invocations, and user guidance that hands
control to `commands/` or `apps/`.

## Dependencies
- Internal: [`commands`](commands/README.md), [`data`](data/README.md),
[`models`](models/README.md), [`training`](training/README.md), [`tools`](tools/README.md),
[`vis`](vis/README.md).
- External: `argparse`, `pydantic` or `mammography.utils.pydantic_fallback`, `pathlib`, optional
YAML parsing for config files.

## Extension and Maintenance Notes
- When adding a new CLI command, update the parser in `cli.py`, keep the forwarded argument behavior
intact, and document the command in the package root README and top-level repo README when
appropriate.
- Keep config model field names aligned with command-line flags so `BaseConfig.from_args()`
continues to work without adapter code.
- Prefer putting reusable logic in subpackages and keeping `cli.py` focused on orchestration,
logging, and config expansion.

## Related Directories
- [`commands`](commands/README.md): Internal command handlers behind the top-level `mammography`
CLI.
- [`data`](data/README.md): Source of truth for dataset ingestion.
- [`training`](training/README.md): Core training and validation logic for supervised mammography
models.
- [`vis`](vis/README.md): Visualization and explainability package for the mammography pipelines.
