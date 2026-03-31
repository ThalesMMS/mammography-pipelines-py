# vis

## Purpose
Visualization and explainability package for the mammography pipelines. It covers static plots,
dashboards, GradCAM-style explanations, clustering views, model comparison, and export helpers.

## Entry Points and Key Modules
- Visualization commands and the web UI import these helpers; some modules also expose programmatic
dashboard launchers.

### Key Files
- `advanced.py`: Advanced visualization utilities for mammography pipeline analysis.
- `cancer_plots.py`: Visualization utilities for DICOM preprocessing and model debugging.
- `cluster_visualizer.py`: Cluster visualization module for mammography embedding analysis.
- `dashboard.py`: Interactive dashboard components for mammography analysis pipeline.
- `explainability.py`: Explainability utilities for model interpretability.
- `export.py`: Figure export helpers for training curves, confusion matrices, and metrics-comparison
artifacts.
- `gradcam.py`: GradCAM wrapper module for backward compatibility.
- `model_comparison.py`: Model comparison utilities for comparing multiple trained models.
- `plots.py`: Basic plotting helpers for scatter plots and clustering-metric summaries.

## How It Fits into the Pipeline
- Turns embeddings, histories, predictions, and metrics into figures that humans can interpret.
- Supports both exploratory analysis and formal report/export flows.
- Provides the visual layer consumed by CLI commands, Streamlit pages, and article/report packaging
utilities.

## Inputs and Outputs
- Inputs: feature matrices, labels, prediction outputs, training histories, DICOM images, and model
checkpoints.
- Outputs: PNG/SVG-style figures, interactive dashboards, GradCAM overlays, comparison tables, and
exported visual assets.

## Dependencies
- Internal: [`analysis`](../analysis/README.md), [`clustering`](../clustering/README.md),
[`eval`](../eval/README.md), [`io`](../io/README.md), [`models`](../models/README.md),
[`tools`](../tools/README.md).
- External: `matplotlib`, `plotly`, `pandas`, `numpy`, `seaborn`, `lets_plot`.

## Extension and Maintenance Notes
- Separate data preparation from rendering where possible so figures can be reused in CLI commands,
tests, and dashboards without rewriting logic.
- Output filenames matter because report packing and experiment dashboards may look for stable
artifact names.
- Explainability utilities depend on architecture-specific layer choices; coordinate changes here
with model builders and command defaults.

## Related Directories
- [`analysis`](../analysis/README.md): Numerical post-processing helpers for embedding exploration.
- [`eval`](../eval/README.md): Evaluation helpers for unsupervised embedding analysis.
- [`commands`](../commands/README.md): Internal command handlers behind the top-level `mammography`
CLI.
- [`apps/web_ui/components`](../apps/web_ui/components/README.md): Reusable Streamlit widgets and
service-style UI helpers used across the web dashboard.
- [`tools`](../tools/README.md): Support utilities for reporting, registries, data audits, and
artifact packaging.
