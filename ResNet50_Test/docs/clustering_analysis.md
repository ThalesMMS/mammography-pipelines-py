# Clustering Analysis in Mammography Research

**Educational documentation for the Breast Density Exploration pipeline**

**WARNING: This is an EDUCATIONAL RESEARCH project. It must NOT be used for clinical or medical diagnostic purposes. No medical decision should rely on these results.**

## Learning Objectives

After completing this guide you should be able to:

1. Understand the fundamentals of unsupervised learning for medical imaging.
2. Explain how K-Means, Gaussian Mixture Models (GMM), and HDBSCAN operate and when to apply each algorithm.
3. Select and interpret evaluation metrics for unlabeled data.
4. Connect clustering behaviour to breast density assessment tasks.
5. Review the mathematical formulation that underpins the algorithms used in this project.
6. Implement clustering experiments inside the ResNet50_Test pipeline.
7. Communicate and validate clustering results with clinical collaborators.

## Table of Contents

1. [Unsupervised Learning Fundamentals](#unsupervised-learning-fundamentals)
2. [Clustering Algorithms](#clustering-algorithms)
3. [Evaluation Metrics and Interpretation](#evaluation-metrics-and-interpretation)
4. [Mathematical Principles](#mathematical-principles)
5. [Clinical Relevance](#clinical-relevance)
6. [Implementation Notes](#implementation-notes)
7. [Dimensionality Reduction](#dimensionality-reduction)
8. [Troubleshooting and Best Practices](#troubleshooting-and-best-practices)

## Unsupervised Learning Fundamentals

Unsupervised learning discovers latent structure in data without labelled examples. In this context we group breast density embeddings and analyse whether the discovered clusters match meaningful patterns.

Key properties:
- No labelled categories are provided during training.
- The objective is exploratory (pattern discovery) rather than predictive.
- Validation relies on internal metrics and qualitative assessment instead of accuracy.
- Results require human interpretation before they can inform clinical hypotheses.

Clustering vs. classification:

| Aspect | Clustering (Unsupervised) | Classification (Supervised) |
| --- | --- | --- |
| Inputs | Feature vectors only | Features plus ground-truth labels |
| Goal | Reveal structure and groups | Predict a known label |
| Validation | Internal indices, silhouette, stability | Accuracy, precision, recall |
| Interpretation | Exploratory, hypothesis generation | Predictive, decision support |

Success criteria for clustering inside this project:
- High intra-cluster similarity and clear inter-cluster separation.
- Stability across different random seeds and subsamples.
- Alignment with breast density knowledge (BI-RADS patterns, projection types, laterality).
- Interpretability through visual prototypes and statistical summaries.

## Clustering Algorithms

### K-Means

K-Means partitions the embedding space into `k` spherical regions by minimising the sum of squared distances between samples and the corresponding centroid.

Objective function:
```
J = Σ_{i=1}^{k} Σ_{x∈C_i} ||x − μ_i||²
```

Practical tips:
- Use `k-means++` initialisation for better convergence.
- Run multiple restarts (`n_init`) and keep the solution with the lowest inertia.
- Standardise or normalise embeddings when magnitude varies across features.

### Gaussian Mixture Models (GMM)

A GMM represents the data distribution as a weighted sum of Gaussian components. Each sample receives a probability of belonging to each component, enabling soft assignments.

Density model:
```
p(x) = Σ_{i=1}^{k} π_i 𝒩(x | μ_i, Σ_i)
```

Implementation notes:
- Fit the model with Expectation-Maximisation (EM).
- Inspect AIC/BIC to compare different numbers of components.
- Covariance matrices provide information about orientation and variance of each cluster.

### HDBSCAN

HDBSCAN is a density-based method that extracts clusters of varying shape and automatically labels outliers. It does not require specifying the number of clusters in advance.

Key configuration parameters:
- `min_cluster_size`: smallest admissible cluster (tune to data volume).
- `min_samples`: controls how conservative the algorithm is regarding noise.
- Use the cluster persistence scores to rank cluster reliability.

## Evaluation Metrics and Interpretation

Because we do not have ground-truth annotations, we rely on internal metrics:

- **Silhouette Score**: measures separation between clusters on a scale of -1 to 1.
- **Davies–Bouldin Index**: average ratio of within-cluster scatter to between-cluster distance (lower is better).
- **Calinski–Harabasz Index**: ratio of inter-cluster dispersion to intra-cluster dispersion (higher is better).

Always complement numerical metrics with qualitative reviews:
- Examine representative images (cluster prototypes) to understand anatomical patterns.
- Compare metadata distributions (projection type, laterality, acquisition device) across clusters.
- Investigate intensity histograms and texture descriptors to validate separation.

Example helper to inspect projection distribution:
```python
from collections import Counter
from typing import Iterable, Mapping

def projection_distribution(metadata: Iterable[Mapping[str, str]], labels):
    """Count projection types for each cluster."""
    result = {}
    for cluster in set(labels):
        if cluster == -1:
            continue  # skip noise when using HDBSCAN
        entries = [m["projection_type"] for i, m in enumerate(metadata) if labels[i] == cluster]
        result[cluster] = Counter(entries)
    return result
```

## Mathematical Principles

- **Euclidean distance** is the default dissimilarity metric for K-Means and PCA projections.
- **Mahalanobis distance** arises implicitly in GMMs through covariance matrices.
- **Mutual reachability distance** underpins density-based approaches such as HDBSCAN.
- Optimisation relies on iterative refinement: Lloyd’s updates for K-Means and EM updates for GMMs.

Understanding these foundations helps diagnose convergence problems, select sensible hyperparameters, and justify results in academic writing.

## Clinical Relevance

Breast density categories (BI-RADS A–D) influence cancer risk and the visibility of lesions. Desired outcomes:
- Clusters that correspond to density levels (low, scattered, heterogeneous, extremely dense).
- Consistency between projections: CC images grouping together and MLO images grouping together unless density dominates the signal.
- Ability to flag atypical patterns or potential asymmetries for further investigation.

**Important:** clustering outputs are exploratory evidence. They require consultation with medical experts before they can inform clinical hypotheses.

## Implementation Notes

The project exposes clustering utilities via the CLI (`src/cli/cluster_cli.py`) and Python APIs under `src/clustering/` and `src/eval/`.

Recommended workflow:
1. Generate embeddings with `embed` CLI or programmatic helper.
2. Apply dimensionality reduction (PCA, UMAP) for visualisation.
3. Run one or more clustering algorithms.
4. Evaluate metrics, review prototypes, and compile a short report in `reports/`.

Configuration profiles in `configs/` capture hyperparameters so experiments remain reproducible.

## Dimensionality Reduction

High-dimensional embeddings are difficult to interpret directly. Use:
- **PCA** for deterministic compression and variance analysis.
- **UMAP** or **t-SNE** for non-linear visualisations (ensure deterministic seeds).
- Document parameter choices (perplexity, neighbours, minimum distance) in experiment logs.

## Troubleshooting and Best Practices

- Monitor silhouette and stability across runs; large swings indicate sensitivity to random seeds.
- Remove obvious outliers before fitting K-Means or GMM to avoid centroid drift.
- When HDBSCAN labels most points as `-1`, increase `min_cluster_size` or reduce noise via dimensionality reduction.
- Keep a changelog of parameter modifications to simplify reproducibility and manuscript preparation.

Clustering is a hypothesis-generation tool. Combine quantitative metrics, qualitative inspection, and clinical review to build trustworthy insights.
