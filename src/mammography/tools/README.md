# tools

## Purpose
Support utilities for reporting, registries, data audits, and artifact packaging. These modules are
the glue that turns run outputs into tracked experiments and publication-friendly assets.

## Entry Points and Key Modules
- Most of these modules are called from CLI commands rather than launched directly.
- `report_pack.py` is the central packaging helper for article/report asset consolidation.

### Key Files
- `baselines_registry.py`: Register embeddings baseline outputs in MLflow and local registry files.
- `data_audit.py`: Utility script that inventories the archive/ directory and emits audit artifacts.
- `data_audit_registry.py`: Register data-audit outputs in MLflow and local registry files.
- `embedding_registry.py`: Register embedding extraction outputs in MLflow and local registry files.
- `eval_export_registry.py`: Register evaluation export outputs in MLflow and local registry files.
- `explain_registry.py`: Register explainability outputs in MLflow and local registry files.
- `inference_registry.py`: Register inference outputs in MLflow and local registry files.
- `report_pack.py`: Helper utilities to consolidate density runs for reporting/Article exports.
- `report_pack_registry.py`: Register report-pack outputs in MLflow and local registry files.
- `train_registry.py`: Register training outputs in MLflow and local registry files.
- `tune_registry.py`: Register tuning outputs in MLflow and local registry files.
- `visualization_registry.py`: Register visualization outputs in MLflow and local registry files.

## How It Fits into the Pipeline
- Collects cross-workflow support logic that does not belong inside training or visualization code.
- Turns outputs from training, inference, explainability, and audits into MLflow or local registry
entries.
- Bridges raw run directories with publication and experiment-management needs.

## Inputs and Outputs
- Inputs: run directories, metrics JSON files, image assets, registry files, and audit manifests.
- Outputs: appended registry rows, MLflow logs, packaged article assets, and audit summaries.

## Dependencies
- Internal: [`commands`](../commands/README.md), [`tracking`](../tracking/README.md),
[`vis`](../vis/README.md).
- External: `mlflow`, `pydicom`, `Pillow`, `csv`, `json`.

## Extension and Maintenance Notes
- Output naming is part of the practical contract here because downstream report generation and
local registries depend on predictable file layouts.
- If a new workflow needs registry support, follow the existing pattern of a focused helper plus a
dedicated `*_registry.py` module instead of expanding one generic registry file.
- Keep side effects explicit; these helpers often append to registries or copy assets, so silent
behavior changes can break reproducibility.

## Related Directories
- [`commands`](../commands/README.md): Internal command handlers behind the top-level `mammography`
CLI.
- [`tracking`](../tracking/README.md): Local experiment tracking support for environments where a
lightweight SQLite-backed registry is preferable to or combined with MLflow.
- [`vis`](../vis/README.md): Visualization and explainability package for the mammography pipelines.
- [`training`](../training/README.md): Core training and validation logic for supervised mammography
models.
