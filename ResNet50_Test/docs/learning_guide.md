# Learning Guide: Breast Density Exploration with ResNet-50 Embeddings

**Comprehensive educational roadmap for the ResNet50_Test research project**

**WARNING: This is an EDUCATIONAL RESEARCH project. It must NOT be used for clinical or medical diagnostic purposes. No medical decision should rely on these results.**

## Overview

This guide helps new contributors build the theoretical and practical knowledge required to work on the project. It combines mathematics, implementation checklists, and recommended readings.

### Learning Objectives

By following the guide you will:
- Understand the end-to-end pipeline: DICOM handling → preprocessing → embedding extraction → clustering → evaluation.
- Gain confidence in the linear algebra, probability, and optimisation concepts used throughout the codebase.
- Learn how technical choices map to clinical expectations for mammography studies.
- Practise designing experiments, validating results, and presenting findings.

### Module Breakdown

1. **DICOM Processing** – standards, metadata, cropping, normalisation, and validation.
2. **Embedding Extraction** – ResNet-50 architecture, transfer learning, batching, and determinism.
3. **Clustering Analysis** – unsupervised algorithms, evaluation metrics, and qualitative review.
4. **Visualisation Methods** – PCA/UMAP/t-SNE, confusion matrices, calibration curves, and dashboards.

## Mathematical Foundations

Refresh or deepen the following topics:

- **Linear algebra:** vector norms, dot products, matrix multiplication, eigenvalue decomposition, singular value decomposition.
- **Probability and statistics:** Gaussian distributions, covariance matrices, expectation, variance, conditional probability.
- **Optimisation:** gradient descent, convexity, convergence criteria, regularisation.
- **Information theory:** entropy, KL divergence, mutual information (useful when evaluating clustering quality).

Short practice exercise:
```text
Given two embedding vectors x and y, compute Euclidean, cosine, and Mahalanobis distances.
Explain what each metric emphasises in the context of mammography images.
```

## Step-by-Step Process

1. **Data ingestion** – run `scripts/check_medical_data.py` to confirm directory layout and metadata integrity.
2. **Preprocessing** – inspect `src/preprocess/image_preprocessor.py`; replicate the pipeline on a single study and visualise intermediate steps.
3. **Embedding extraction** – use `src/cli/embed_cli.py` to cache embeddings; review the cached files under `results/`.
4. **Clustering** – execute `src/cli/cluster_cli.py` with configuration options for K-Means, GMM, and HDBSCAN; log metrics.
5. **Evaluation** – compute silhouette and Davies–Bouldin scores and produce prototype grids.

## Suggested Learning Path

Beginner to intermediate path:
1. Review `docs/dicom_processing.md` and experiment with basic DICOM loading using `pydicom`.
2. Read `docs/embedding_extraction.md` and reproduce the embedding extraction notebook (create one if absent).
3. Study `docs/clustering_analysis.md`; implement metric functions from scratch as a coding exercise.
4. Build exploratory plots following `docs/visualization_methods.md`.
5. Extend tests under `tests/` to cover new scenarios you create during practice.

Prerequisites:
- Python programming (functions, classes, typing, unit testing).
- Basic PyTorch knowledge (tensors, modules, inference).
- Familiarity with Git, virtual environments, and command-line tooling.

## Hands-On Exercises

1. **Intensity Normalisation Audit** – write a script to compute pre- and post-normalisation histograms for 10 studies; discuss the effect on contrast.
2. **Embedding Drift Check** – run embeddings with two different seeds and measure cosine similarity; report findings.
3. **Clustering Sensitivity** – vary `k` in K-Means and plot changes in silhouette score.
4. **Bias Exploration** – group clusters by manufacturer and investigate potential systematic differences.

## Assessment and Validation

- Maintain an experiment log (date, configuration, metrics, observations) in Markdown or a spreadsheet.
- Present findings in lab meetings or peer reviews; emphasise hypotheses, methods, and limitations.
- Use `pytest` plus additional scripts to validate data splits, disclaimers, and deterministic behaviour.

## Advanced Topics

- Semi-supervised learning: active learning loops, label propagation, and linear probes.
- Model interpretability: Grad-CAM, feature attribution on ResNet-50 feature maps.
- Confidence estimation and calibration for semi-supervised classifiers.
- Differential privacy and secure handling of sensitive medical data.

## Resources and References

- *DICOM Standard Browser* – official documentation from NEMA.
- *Deep Learning* by Goodfellow, Bengio, and Courville – chapters on CNNs and optimisation.
- *Pattern Recognition and Machine Learning* by Bishop – clusters, EM, and Bayesian methods.
- PyTorch tutorials on transfer learning and mixed precision inference.

Take notes as you progress, keep the research disclaimer visible in every shared artefact, and coordinate with supervisors or clinical experts for domain-specific questions.
