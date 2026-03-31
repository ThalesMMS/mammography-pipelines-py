# tracking

## Purpose
Local experiment tracking support for environments where a lightweight SQLite-backed registry is
preferable to or combined with MLflow.

## Entry Points and Key Modules
- This package is consumed programmatically by tooling or dashboards; it is not a standalone CLI
surface by itself.

### Key Files
- `local_tracker.py`: LocalTracker: SQLite-based experiment tracking for offline use.

## How It Fits into the Pipeline
- Provides an offline-friendly place to store run metadata and lightweight experiment history.
- Complements the registry utilities in `tools/` when a simple local database is enough.
- Can support dashboard or reporting features that need fast local lookup of runs.

## Inputs and Outputs
- Inputs: experiment params, metrics, artifact references, and timestamps from training or export
workflows.
- Outputs: SQLite rows and serialized metadata payloads that other tooling can query later.

## Dependencies
- Internal: [`tools`](../tools/README.md), [`apps/web_ui`](../apps/web_ui/README.md).
- External: `sqlite3`, `json`.

## Extension and Maintenance Notes
- Database schema stability matters if any existing runs depend on this store, so prefer additive
changes over incompatible rewrites.
- Keep this layer focused on storage and querying concerns rather than analytics or rendering.
- If MLflow and local tracking overlap, document the source of truth for each workflow to avoid
duplicated or conflicting metadata.

## Related Directories
- [`tools`](../tools/README.md): Support utilities for reporting, registries, data audits, and
artifact packaging.
- [`apps/web_ui`](../apps/web_ui/README.md): Main Streamlit dashboard for interactive dataset
browsing, inference, explainability, experiment review, training configuration, and hyperparameter
tuning.
- [`training`](../training/README.md): Core training and validation logic for supervised mammography
models.
