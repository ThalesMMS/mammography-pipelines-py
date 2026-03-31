# analysis

## Purpose
Numerical post-processing helpers for embedding exploration. This package is where dimensionality
reduction and lightweight clustering-oriented analysis run after features have already been
extracted.

## Entry Points and Key Modules
- No standalone CLI entrypoint lives here; the functions are typically invoked from embedding
extraction or visualization workflows.

### Key Files
- `clustering.py`: Dimensionality-reduction and clustering helpers for PCA, t-SNE, UMAP, k-means,
and cluster-count selection.

## How It Fits into the Pipeline
- Turns high-dimensional feature matrices into PCA, t-SNE, or UMAP projections for inspection.
- Provides helper routines such as k selection and k-means runs that downstream visualization code
can reuse.
- Supports exploratory workflows rather than core training loops.

## Inputs and Outputs
- Inputs: embedding matrices, optional labels or metadata tables, and projection/clustering
hyperparameters.
- Outputs: reduced coordinates, cluster assignments, and summary statistics that feed plotting or
evaluation steps.

## Dependencies
- Internal: [`utils`](../utils/README.md).
- External: `numpy`, `pandas`, `scikit-learn`, `umap-learn`.

## Extension and Maintenance Notes
- Most routines assume the caller already handled feature extraction and basic cleaning; keep that
separation instead of adding dataset I/O here.
- Projection methods can be sensitive to sample count and randomness, so preserve explicit seeds and
parameter validation when extending them.
- If a new analysis primitive also needs standardized result objects or metrics, consider pairing it
with `clustering/` or `eval/` instead of growing ad hoc outputs here.

## Related Directories
- [`models/embeddings`](../models/embeddings/README.md): Backbone-specific embedding extractors and
typed vector helpers.
- [`clustering`](../clustering/README.md): Structured clustering layer for embedding experiments.
- [`eval`](../eval/README.md): Evaluation helpers for unsupervised embedding analysis.
- [`vis`](../vis/README.md): Visualization and explainability package for the mammography pipelines.
