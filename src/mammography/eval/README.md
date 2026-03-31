# eval

## Purpose
Evaluation helpers for unsupervised embedding analysis. Right now the package focuses on clustering
evaluation and acts as the scoring layer that sits beside the clustering code.

## Entry Points and Key Modules
- No standalone entrypoint is defined here; evaluation is usually invoked from pipeline, analysis,
or visualization workflows.

### Key Files
- `clustering_evaluator.py`: Clustering evaluation module for mammography embedding analysis.

## How It Fits into the Pipeline
- Scores clustering outputs so exploratory embedding experiments can be compared systematically.
- Provides a clean boundary between algorithm execution and metric computation.
- Feeds evaluation summaries into dashboards, reports, or experiment exports.

## Inputs and Outputs
- Inputs: cluster assignments, feature embeddings, optional labels, and result metadata from
clustering runs.
- Outputs: evaluation summaries, metric bundles, and helper objects for downstream reporting.

## Dependencies
- Internal: [`clustering`](../clustering/README.md), [`pipeline`](../pipeline/README.md),
[`vis`](../vis/README.md).
- External: `numpy`, `torch`.

## Extension and Maintenance Notes
- Keep metric naming stable so visualization and report code can consume outputs without special-
case adapters.
- If supervised metrics are added later, separate them clearly from the clustering-focused API
instead of overloading existing evaluators.
- This is a good location for reusable evaluation primitives, not for CLI orchestration or artifact
export.

## Related Directories
- [`analysis`](../analysis/README.md): Numerical post-processing helpers for embedding exploration.
- [`clustering`](../clustering/README.md): Structured clustering layer for embedding experiments.
- [`pipeline`](../pipeline/README.md): High-level end-to-end orchestration helpers that chain data
loading, preprocessing, model execution, clustering, evaluation, and visualization into a single
pipeline object.
- [`vis`](../vis/README.md): Visualization and explainability package for the mammography pipelines.
