# apps/web_ui/pages

## Purpose
Ordered Streamlit page modules for the main dashboard. Each file represents one visible page in the
application and owns the page-specific state, controls, and rendering flow.

## Entry Points and Key Modules
- Streamlit auto-discovers these modules from the `pages/` directory; each page file exposes its own
`main()` function.
- Filename prefixes and emoji names determine navigation ordering, so renaming pages changes the UI
structure.

### Key Files
- `0_📁_Dataset_Browser.py`: Dataset browser page for exploring mammography images and metadata.
- `1_📊_Inference.py`: Inference page for breast density classification with trained models.
- `2_🔍_Explainability.py`: Explainability page for visualizing GradCAM heatmaps on mammography
predictions.
- `3_📈_Experiments.py`: Experiments page for viewing and analyzing MLflow training runs.
- `4_⚙️_Training.py`: Training configuration page for launching new training jobs.
- `5_🔬_Hyperparameter_Tuning.py`: Hyperparameter Tuning page for viewing and analyzing Optuna
optimization studies.

## How It Fits into the Pipeline
- Separates major workflows such as browsing, inference, explainability, experiment review,
training, and tuning into isolated modules.
- Keeps the root dashboard bootstrap small while letting each page coordinate the components and
lower-level packages it needs.
- Forms the highest UI layer in the repository, closest to user actions and operator workflows.

## Inputs and Outputs
- Inputs: datasets, checkpoints, images, metrics, MLflow experiments, Optuna studies, and session-
state selections.
- Outputs: page-specific charts, explanations, job launches, exports, and navigation state for the
dashboard session.

## Dependencies
- Internal: [`apps/web_ui`](../README.md), [`apps/web_ui/components`](../components/README.md),
[`models`](../../../models/README.md), [`tuning`](../../../tuning/README.md),
[`vis`](../../../vis/README.md).
- External: `streamlit`, `plotly`, `mlflow`, `optuna`, `Pillow`.

## Extension and Maintenance Notes
- Preserve page filename ordering unless the navigation order should change intentionally.
- Cross-page session keys should be synchronized through the shared utilities rather than recreated
ad hoc inside individual page files.
- Favor component reuse for repeated plotting or export UI; page files should focus on flow control
and page-specific orchestration.

## Related Directories
- [`apps/web_ui`](../README.md): Main Streamlit dashboard for interactive dataset browsing,
inference, explainability, experiment review, training configuration, and hyperparameter tuning.
- [`apps/web_ui/components`](../components/README.md): Reusable Streamlit widgets and service-style
UI helpers used across the web dashboard.
- [`commands`](../../../commands/README.md): Internal command handlers behind the top-level
`mammography` CLI.
- [`vis`](../../../vis/README.md): Visualization and explainability package for the mammography
pipelines.
