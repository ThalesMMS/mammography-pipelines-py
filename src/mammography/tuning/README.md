# tuning

## Purpose
Hyperparameter optimization utilities built around Optuna and related tuning helpers. This package
also contains search-space validation, LR range testing, and hardware-aware resource management.

## Entry Points and Key Modules
- CLI tuning workflows import these modules through `commands/tune.py` or web dashboard pages rather
than running them directly.

### Key Files
- `lr_finder.py`: Learning rate finder implementation using range test method for optimal LR
discovery.
- `optuna_tuner.py`: Optuna integration for automated hyperparameter tuning with MedianPruner.
- `resource_manager.py`: Resource management utilities for AutoML with GPU memory detection and
search space constraints.
- `search_space.py`: Pydantic-style models and YAML loader for Optuna-compatible hyperparameter
search spaces.
- `study_utils.py`: Helpers for loading Optuna study metadata without running new trials.

## How It Fits into the Pipeline
- Defines and executes study logic for automated search over training hyperparameters.
- Keeps tuning concerns such as pruning, search-space validation, and study summaries separate from
the baseline training loop.
- Lets the UI and CLI share the same study model and search constraints.

## Inputs and Outputs
- Inputs: validated training configs, Optuna study paths, search-space YAML, and hardware/resource
constraints.
- Outputs: Optuna studies, best-trial metadata, LR finder traces, and summary views of existing
studies.

## Dependencies
- Internal: [`training`](../training/README.md),
[`apps/web_ui/pages`](../apps/web_ui/pages/README.md), [`commands`](../commands/README.md),
[`utils`](../utils/README.md).
- External: `optuna`, `torch`, `pydantic`, `yaml`, `matplotlib`.

## Extension and Maintenance Notes
- `search_space.py` is the contract for YAML-defined tuning ranges; keep it strict so invalid
studies fail early.
- Resource-aware logic should constrain experiments predictably rather than silently changing the
meaning of the search space.
- Study-summary helpers are intentionally lightweight and should stay usable even when a full
optimization run is not being launched.

## Related Directories
- [`training`](../training/README.md): Core training and validation logic for supervised mammography
models.
- [`apps/web_ui/pages`](../apps/web_ui/pages/README.md): Ordered Streamlit page modules for the main
dashboard.
- [`commands`](../commands/README.md): Internal command handlers behind the top-level `mammography`
CLI.
- [`utils`](../utils/README.md): Cross-cutting helpers shared across the repository.
