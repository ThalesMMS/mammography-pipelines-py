# pipeline

## Purpose
High-level end-to-end orchestration helpers that chain data loading, preprocessing, model execution,
clustering, evaluation, and visualization into a single pipeline object.

## Entry Points and Key Modules
- This package exposes programmatic orchestration through `MammographyPipeline`; the CLI primarily
reaches similar workflows through command modules.

### Key Files
- `mammography_pipeline.py`: Mammography analysis pipeline for end-to-end processing.

## How It Fits into the Pipeline
- Provides one place to run a multi-stage mammography analysis pipeline from a higher-level
configuration.
- Coordinates subsystems without requiring callers to assemble every lower-level dependency
themselves.
- Acts as a convenience layer for scripting or future workflow automation.

## Inputs and Outputs
- Inputs: pipeline configuration, paths to datasets or artifacts, and stage-specific runtime
options.
- Outputs: combined processing results, evaluation summaries, and visualization-ready data or saved
artifacts depending on the invoked stages.

## Dependencies
- Internal: [`clustering`](../clustering/README.md), [`eval`](../eval/README.md),
[`io`](../io/README.md), [`models`](../models/README.md), [`preprocess`](../preprocess/README.md),
[`vis`](../vis/README.md).
- External: `torch`, `yaml`, `typer`.

## Extension and Maintenance Notes
- Keep orchestration thin: core algorithms and file-format details should remain in the specialist
packages that this layer coordinates.
- If command modules and `MammographyPipeline` begin to diverge in behavior, prefer extracting
shared stage helpers rather than duplicating workflow logic.
- Because this package spans many subsystems, it is a useful place to document assumptions about
stage ordering and shared data contracts.

## Related Directories
- [`mammography`](../README.md): Primary package root for the mammography pipelines.
- [`data`](../data/README.md): Source of truth for dataset ingestion.
- [`preprocess`](../preprocess/README.md): Image preprocessing abstractions for mammography data.
- [`models`](../models/README.md): Trainable model architectures and builders for density and
cancer-related mammography tasks.
- [`vis`](../vis/README.md): Visualization and explainability package for the mammography pipelines.
