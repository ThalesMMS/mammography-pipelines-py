# Visualisation Methods for Mammography Research

**Educational documentation for the Breast Density Exploration pipeline**

**WARNING: This is an EDUCATIONAL RESEARCH project. It must NOT be used for clinical or medical diagnostic purposes. No medical decision should rely on these results.**

## Purpose

Visualisations translate high-dimensional embeddings and clustering results into insights that researchers and clinicians can interpret. This guide introduces recommended techniques and practices.

## Dimensionality Reduction

Use dimensionality reduction to project 2048-D embeddings to 2D or 3D:

- **PCA** (linear) – deterministic, preserves global variance, ideal for initial exploration and as a preprocessing step for other methods.
- **UMAP** (non-linear) – balances local and global structure, fast, suits exploratory plots for manuscripts.
- **t-SNE** (non-linear) – preserves local neighbourhoods, useful for highlighting cluster separation but requires careful parameter tuning.

Example PCA helper:
```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_pca(embeddings, labels, title="PCA projection"):
    pca = PCA(n_components=2, random_state=42)
    projected = pca.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(projected[:, 0], projected[:, 1], c=labels, cmap="viridis", alpha=0.8)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title(title)
    fig.colorbar(scatter, label="Cluster")
    return fig
```

Best practices:
- Fix random seeds for deterministic UMAP/t-SNE results (`random_state`, `init='spectral'`).
- Record hyperparameters (perplexity, number of neighbours, min_dist) in experiment logs.
- Combine multiple projections to cross-check interpretations.

## Visualisation Principles

- Use consistent colour palettes across figures and reports.
- Provide legends, axis labels, and contextual annotations (e.g., cluster IDs, density categories).
- Include sample counts and summary statistics alongside plots.
- For manuscripts, export to vector formats (PDF/SVG) at ≥300 DPI.

## Cluster Visualisation

Recommended plots:

1. **Prototype grids** – display representative mammograms per cluster (ensure PHI is removed).
2. **Silhouette plots** – visualise cohesion and separation of clusters.
3. **Confusion matrices** – when labels are available in semi-supervised experiments.
4. **Mosaic plots** – compare metadata distributions (projection type, laterality, manufacturer) by cluster.

Prototype selection example:
```python
import numpy as np

def prototype_indices(embeddings, cluster_labels, top_k=5):
    indices = {}
    for cluster in set(cluster_labels):
        if cluster == -1:
            continue
        mask = cluster_labels == cluster
        centroid = embeddings[mask].mean(axis=0)
        distances = np.linalg.norm(embeddings[mask] - centroid, axis=1)
        cluster_idx = np.where(mask)[0]
        indices[cluster] = cluster_idx[np.argsort(distances)[:top_k]].tolist()
    return indices
```

## Interactive Dashboards

Use Plotly or Bokeh to build dashboards (`src/viz/dashboard.py`) combining:
- 2D scatter plots with hover metadata.
- Linked image viewers to inspect prototype mammograms.
- Filters for projection type, manufacturer, or acquisition date.

Ensure dashboards display the research disclaimer prominently.

## Mathematical Notes

- PCA works via eigenvalue decomposition of the covariance matrix; components are orthogonal directions of maximum variance.
- t-SNE minimises the KL divergence between high-dimensional and low-dimensional similarity distributions.
- UMAP constructs a fuzzy simplicial complex and optimises cross-entropy between high- and low-dimensional graphs.

Understanding these formulations helps justify parameter choices in academic publications.

## Troubleshooting

- **Crowded embeddings:** reduce perplexity (t-SNE) or increase `min_dist` (UMAP).
- **Inconsistent projections across runs:** set seeds and reduce perplexity/nearest neighbours.
- **Colour mapping confusion:** map clusters to descriptive labels or add textual annotations.
- **Large datasets:** subsample for interactive plots while keeping full-resolution plots for quantitative evaluation.

Visualisations are storytelling tools—pair them with narrative text explaining what each plot reveals about breast density patterns and always reiterate the research-only disclaimer.
