# apps/web_ui/components

## Purpose
Reusable Streamlit widgets and service-style UI helpers used across the web dashboard. These modules
keep complex rendering logic out of the top-level page files.

## Entry Points and Key Modules
- This directory does not expose its own launcher; page modules import these components and call
their render or helper methods directly.

### Key Files
- `dataset_viewer.py`: Dataset viewer component for browsing mammography images with metadata.
- `metrics_monitor.py`: Metrics monitor component for real-time training metrics display.
- `report_exporter.py`: Report export component for packaging training results and visualizations.
- `results_visualizer.py`: Results visualizer component for model evaluation metrics display.

## How It Fits into the Pipeline
- Provides focused rendering units for dataset browsing, metric inspection, report export, and
results visualization.
- Improves page-level maintainability by concentrating plotting and export logic in shared modules.
- Acts as the UI-facing bridge to reporting and visualization code elsewhere in the package.

## Inputs and Outputs
- Inputs: DataFrames, run directories, metrics JSON payloads, prediction outputs, and Streamlit
session state.
- Outputs: rendered charts and tables, downloadable artifacts, and export manifests ready for
packaging.

## Dependencies
- Internal: [`apps/web_ui`](../README.md), [`tools`](../../../tools/README.md),
[`vis`](../../../vis/README.md).
- External: `streamlit`, `pandas`, `matplotlib`, `mlflow`, `seaborn`.

## Extension and Maintenance Notes
- Keep business logic light here; heavy artifact generation should remain in `tools/` or `vis/` and
be called from the component layer.
- If multiple pages share the same session-state keys, define and document them centrally rather
than creating page-local conventions.
- When adding a component that writes files, make its output layout match the existing report/export
tooling.

## Related Directories
- [`apps/web_ui`](../README.md): Main Streamlit dashboard for interactive dataset browsing,
inference, explainability, experiment review, training configuration, and hyperparameter tuning.
- [`apps/web_ui/pages`](../pages/README.md): Ordered Streamlit page modules for the main dashboard.
- [`tools`](../../../tools/README.md): Support utilities for reporting, registries, data audits, and
artifact packaging.
- [`vis`](../../../vis/README.md): Visualization and explainability package for the mammography
pipelines.
