# apps/web_ui

## Purpose
Main Streamlit dashboard for interactive dataset browsing, inference, explainability, experiment
review, training configuration, and hyperparameter tuning.

## Entry Points and Key Modules
- `streamlit_app.py` owns the dashboard bootstrap and shared app shell.
- `run()` and `main()` are the runtime hooks used by Streamlit launchers and wrappers.
- `mammography web` in `commands/web.py` is the CLI-facing launcher for this dashboard.

### Key Files
- `streamlit_app.py`: Streamlit web UI dashboard for model inference, visualization, and training
configuration.
- `utils.py`: Utility functions for web UI operations and data handling.

### Subdirectories
- [`components/`](components/README.md): Reusable Streamlit widgets and service-style UI helpers
used across the web dashboard.
- [`pages/`](pages/README.md): Ordered Streamlit page modules for the main dashboard.

## How It Fits into the Pipeline
- Exposes an integrated web front end on top of datasets, checkpoints, MLflow runs, and tuning
studies.
- Coordinates reusable components and page modules through shared session state.
- Serves as the operator-facing counterpart to the batch-oriented CLI commands.

## Inputs and Outputs
- Inputs: training runs, metrics files, model checkpoints, dataset locations, MLflow tracking data,
and Optuna studies.
- Outputs: rendered dashboards, ad hoc exports, launched jobs, and operator decisions captured in
session state.

## Dependencies
- Internal: [`apps/web_ui/components`](components/README.md),
[`apps/web_ui/pages`](pages/README.md), [`commands`](../../commands/README.md),
[`tracking`](../../tracking/README.md).
- External: `streamlit`, `mlflow`, `torch`.

## Extension and Maintenance Notes
- Keep page discovery conventions intact: Streamlit relies on the `pages/` directory and filename
ordering.
- Shared state should be introduced through `utils.ensure_shared_session_state()` so pages do not
silently diverge on key names.
- If a feature is reusable across pages, implement it in `components/` instead of copying rendering
code between page files.

## Related Directories
- [`apps`](../README.md): Umbrella package for operator-facing applications.
- [`apps/web_ui/components`](components/README.md): Reusable Streamlit widgets and service-style UI
helpers used across the web dashboard.
- [`apps/web_ui/pages`](pages/README.md): Ordered Streamlit page modules for the main dashboard.
- [`commands`](../../commands/README.md): Internal command handlers behind the top-level
`mammography` CLI.
