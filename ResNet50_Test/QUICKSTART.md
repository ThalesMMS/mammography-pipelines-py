# Quick Start Guide – ResNet50_Test

**WARNING: This is an EDUCATIONAL RESEARCH project. It must NOT be used for clinical or medical diagnostic purposes.**

## Project Status Snapshot

✅ Completed
- Project constitution (governance and guiding principles)
- Three detailed specifications (unsupervised, semi-supervised, baseline)
- Critical technical clarifications and risk register
- Go/No-Go decision summary
- Python environment initialised and verified
- 16 patients with 82 DICOM files detected during discovery
- Dependencies installed (PyTorch, scikit-learn, etc.)
- Automated compliance checks in place

Next step: Implement the Python codebase starting with Phase 1 (unsupervised pipeline).

## Project Goal

Investigate breast density categorisation (BI-RADS A, PA, PD, D) by combining:
1. ResNet-50 embeddings for unsupervised representation learning
2. Clustering (K-Means, Gaussian Mixture, HDBSCAN)
3. Semi-supervised mapping with minimal annotations
4. Supervised baseline for grounded comparison

## 15-Minute Setup

### 1. Environment Preparation

```bash
# Clone and enter the repository
git clone <repository-url>
cd ResNet50_Test

# Set up a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate   # Windows

# Install core dependencies
pip install -r requirements.txt
```

### 2. Data Layout

```bash
mkdir -p archive data/raw data/processed results reports docs
# Place DICOM studies under ./archive/
# Example: archive/patient_001/image_001.dcm
```

### 3. Smoke Test

```bash
source venv/bin/activate
python scripts/check_status.py
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected outcome: all checks passing with automatic GPU detection when available.

## Phase-by-Phase Delivery

### Phase 1 – Unsupervised (Weeks 1–2)

Goal: Preprocessing to embeddings to clustering.

Implement in this order:

```bash
# Preprocessing utilities
src/preprocess/image_preprocessor.py
src/preprocess/preprocessed_tensor.py

# Embedding extraction
src/embedding/resnet_extractor.py  # see README for module plan

# Clustering logic
src/clustering/clustering_algorithms.py
```

Helpful scripts:

```bash
python scripts/check_medical_data.py --config configs/base.yaml
python scripts/check_patient_splitting.py --data-dir archive/
```

Success criteria: silhouette score at least 0.25 and clusters documented with qualitative analysis.

### Phase 2 – Semi-supervised (Weeks 3–4)

- Active learning sampling strategies
- Linear probe or lightweight classifier on embeddings
- Mapping between clusters and BI-RADS labels

Success criteria: macro F1 above 0.40 with clear bias analysis.

### Phase 3 – Supervised Baseline (Week 5)

- Deterministic train/validation/test splits by patient
- Supervised classifier with calibration curves
- Comparison against the semi-supervised pipeline

## Tests & Quality Gates

```
tests/
├── unit/            # Unit tests (target at least 85% coverage)
├── integration/     # End-to-end validation
├── contract/        # Compliance and disclaimer enforcement
├── performance/     # Stress tests for large datasets
└── reproducibility/ # Determinism and seed checks
```

Commands:

```bash
pytest tests/ -v --cov=src
ruff check src/
black src/
mypy src/
```

Use `make quality-fix` to run format, lint, type-check, and tests sequentially.

## Key Configuration Files

`configs/preprocessing.yaml`
```yaml
preprocessing:
  input_size: [512, 512]
  normalization: z_score_per_image
  remove_borders: true
  preserve_pixel_spacing: true
```

`configs/embedding.yaml`
```yaml
embedding:
  model: resnet50
  layer: avgpool
  grayscale_strategy: replicate_1_to_3
  batch_size: 16
  deterministic: true
```

`configs/clustering.yaml`
```yaml
clustering:
  algorithms: [kmeans, gmm, hdbscan]
  kmeans_k: 4
  min_silhouette: 0.25
  random_state: 42
```

## Compliance & Governance

```python
RESEARCH_DISCLAIMER = """
WARNING: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
"""
```

- Store DICOM data locally only (`archive/` stays unversioned).
- Always split by patient; never split by image.
- Fix random seeds and enable deterministic behaviour for reproducibility.
- Run `scripts/check_disclaimers.py` before distributing outputs.

## Troubleshooting Cheat Sheet

**GPU out of memory**
```bash
python scripts/check_status.py --batch-size 8
python scripts/check_status.py --device cpu
```

**Poor clustering performance (silhouette < 0.25)**
```bash
python scripts/cluster_analysis.py --algorithm gmm
python scripts/cluster_analysis.py --algorithm hdbscan
```

**Potential data leakage**
```bash
python scripts/check_patient_splitting.py --data-dir archive/
```

## Success Metrics

- Phase 1: executable pipeline, silhouette at least 0.25, qualitative review complete.
- Phase 2: macro F1 above 0.40, clear mapping to BI-RADS, documented bias assessment.
- Phase 3: stable baseline, comparison report, reproducible results with confidence intervals.

## Immediate Action Plan

**Today (30 minutes)**
- [ ] Configure the Python environment
- [ ] Create directory structure
- [ ] Validate DICOM placement under `archive/`

**This week**
- [ ] Implement DICOM preprocessing
- [ ] Build the ResNet-50 embedding extractor
- [ ] Implement K-Means clustering
- [ ] Run the initial end-to-end experiment

**Next week**
- [ ] Evaluate cluster quality and visualise with UMAP/t-SNE
- [ ] Document limitations and bias observations
- [ ] Prepare tooling for semi-supervised mapping

## Additional Resources

- `README.md` – full project guide
- `.specify/memory/constitution.md` – governance
- `.specify/memory/clarifications.md` – technical decisions
- `.specify/memory/risks.md` – risk tracking

Need help?
1. Enable debug logs with `--log-level DEBUG`.
2. Run `python scripts/check_status.py` for a full health report.
3. Review the documentation under `docs/`.

All documentation now defaults to English for collaborative work.
