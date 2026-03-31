# clustering

## Purpose
Structured clustering layer for embedding experiments. This package wraps algorithm selection,
result typing, and timing/metadata handling around unsupervised clustering runs.

## Entry Points and Key Modules
- No direct CLI entrypoint lives here; clustering is usually triggered by embedding analysis,
pipeline orchestration, or evaluation code.

### Key Files
- `clustering_algorithms.py`: Clustering algorithms module for mammography embedding analysis.
- `clustering_result.py`: ClusteringResult model for clustering algorithm results representation.

## How It Fits into the Pipeline
- Offers a higher-level clustering facade than the simple helpers in `analysis/`.
- Provides consistent result models so evaluation and visualization layers can consume algorithm
outputs without special cases.
- Supports embedding quality assessment and exploratory subgroup discovery.

## Inputs and Outputs
- Inputs: embedding tensors or arrays plus algorithm settings such as k values, distance
assumptions, or runtime options.
- Outputs: `ClusteringResult` objects, cluster labels, fit metadata, and timing information.

## Dependencies
- Internal: [`analysis`](../analysis/README.md), [`eval`](../eval/README.md),
[`vis`](../vis/README.md).
- External: `scikit-learn`, `hdbscan`, `torch`, `numpy`.

## Extension and Maintenance Notes
- Add new clustering backends through the algorithm facade so result serialization and evaluation
stay uniform.
- If a new algorithm needs extra metrics, update the shared result model rather than returning side-
channel structures.
- Keep any medical-use disclaimer or scope messaging consistent with the rest of the research
pipeline.

## Related Directories
- [`analysis`](../analysis/README.md): Numerical post-processing helpers for embedding exploration.
- [`eval`](../eval/README.md): Evaluation helpers for unsupervised embedding analysis.
- [`pipeline`](../pipeline/README.md): High-level end-to-end orchestration helpers that chain data
loading, preprocessing, model execution, clustering, evaluation, and visualization into a single
pipeline object.
- [`vis`](../vis/README.md): Visualization and explainability package for the mammography pipelines.
