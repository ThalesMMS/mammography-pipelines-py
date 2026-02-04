# Mammography Pipelines

Consolidated mammography pipeline with a single CLI, modular Python package, and a scientific article workspace. The focus is breast density classification (BI-RADS A–D) plus reproducible reporting and research artifacts.

## Entrypoints

- `mammography` (primary CLI entrypoint)

## Installation

```bash
# Recommended: reproducible install via uv lockfile
pip install uv
uv sync --frozen

# Legacy (pip)
# pip install -r requirements.txt
```

## Docker

```bash
docker build -t mammography-pipelines .
docker run --rm -it -v "$PWD":/app -w /app mammography-pipelines
```

## Quick Start

```bash
# Show CLI help
mammography --help

# Interactive wizard
mammography wizard

# Embeddings
mammography embed -- \
  --csv classificacao.csv \
  --dicom-root archive \
  --outdir outputs/embeddings_resnet50

# Baselines classicos (embeddings)
mammography embeddings-baselines -- \
  --embeddings-dir outputs/embeddings_resnet50 \
  --outdir outputs/embeddings_baselines

# Treinamento de densidade
mammography train-density -- \
  --csv classificacao.csv \
  --dicom-root archive \
  --outdir outputs/mammo_efficientnetb0_density \
  --epochs 8 \
  --arch resnet50

# Visualization
mammography visualize -- \
  --input outputs/embeddings_resnet50/features.npy \
  --outdir outputs/visualizations

# Inference
mammography inference -- \
  --checkpoint outputs/mammo_efficientnetb0_density/results_1/best_model.pt \
  --input archive \
  --output outputs/preds.csv

# Augmentation
mammography augment -- \
  --source-dir archive \
  --output-dir outputs/augmented \
  --num-augmentations 2

# Report packaging (Article integration)
mammography report-pack --run outputs/mammo_efficientnetb0_density/results_1
```

## View-Specific Model Training

View-specific training trains **separate models for CC (craniocaudal) and MLO (mediolateral oblique) views**, addressing the finding that different mammography views may have distinct predictive characteristics. An optional ensemble method combines predictions from both view-specific models.

### Why Use View-Specific Training?

- **View-Specific Patterns**: CC and MLO views show breast tissue from different angles and may capture different diagnostic features
- **Improved Accuracy**: Ensemble prediction combining both views can outperform single-model approaches
- **Research Analysis**: Compare predictive power of CC vs MLO views to understand which is more informative
- **Clinical Workflow**: Aligns with real clinical practice where radiologists examine both views

### Quick Start

```bash
# Train view-specific models with ensemble prediction
mammography train \
  --csv classificacao.csv \
  --dicom-root archive \
  --outdir outputs/view_specific_density \
  --view-specific-training \
  --views CC,MLO \
  --ensemble-method average \
  --epochs 20

# For research: train separate models without ensemble to compare views
mammography train \
  --csv classificacao.csv \
  --dicom-root archive \
  --outdir outputs/view_comparison \
  --view-specific-training \
  --views CC,MLO \
  --ensemble-method none \
  --epochs 20
```

### Configuration Options

- **`--view-specific-training`**: Enable view-specific training mode (flag)
- **`--view-column`**: Column name in CSV containing view labels (default: `view`)
- **`--views`**: Comma-separated list of views to train (default: `CC,MLO`)
- **`--ensemble-method`**: Method for combining view predictions
  - `none`: Keep models separate (for analysis)
  - `average`: Simple average of probabilities (recommended)
  - `weighted`: Weighted average by validation performance
  - `max`: Take maximum probability across views

### Example Configuration

See `configs/view_specific_training.yaml` for a complete example configuration with all parameters and usage notes.

### Output Structure

View-specific training creates separate output directories for each view plus ensemble metrics:

```
outputs/view_specific_density/
├── view_specific_density_CC/          # CC view model
│   ├── checkpoints/
│   │   ├── best_model_cc.pt
│   │   └── checkpoint_cc.pt
│   └── metrics/
│       ├── val_metrics_cc.json
│       └── best_metrics_cc.json
├── view_specific_density_MLO/         # MLO view model
│   ├── checkpoints/
│   │   ├── best_model_mlo.pt
│   │   └── checkpoint_mlo.pt
│   └── metrics/
│       ├── val_metrics_mlo.json
│       └── best_metrics_mlo.json
└── view_specific_density/             # Main directory
    └── metrics/
        ├── ensemble_metrics.json      # Combined ensemble results
        └── view_comparison.png        # Visualization comparing views
```

### Interpreting Results

After training, examine the metrics to compare views:

```bash
# View CC-specific metrics
cat outputs/view_specific_density_CC/metrics/val_metrics_cc.json

# View MLO-specific metrics
cat outputs/view_specific_density_MLO/metrics/val_metrics_mlo.json

# View ensemble metrics (if ensemble_method != "none")
cat outputs/view_specific_density/metrics/ensemble_metrics.json
```

The `view_comparison.png` visualization shows side-by-side comparison of accuracy, F1 score, and AUC across CC, MLO, and ensemble models.

### Use Cases

1. **Research Analysis**: Compare CC vs MLO predictive power using `--ensemble-method none`
2. **Production Deployment**: Use `--ensemble-method average` for best overall accuracy
3. **View-Specific Inference**: Load individual view models for view-specific predictions
4. **Ablation Studies**: Train with different ensemble methods to find optimal combination strategy

## CLI Overview

The CLI is built around two core workflows plus utilities:

- **`embed`**: Extract ResNet/EfficientNet embeddings and optional PCA/t-SNE/UMAP analysis.
- **`embeddings-baselines`**: Compare embeddings against classical descriptors (LogReg/SVM/RF).
- **`train-density`**: Train density classifiers (EfficientNetB0/ResNet50) with cache modes and reporting artifacts.
- **`visualize`**: Generate plots from embeddings or run directories.
- **`inference`**: Run checkpointed inference over image folders or single files.
- **`augment`**: Generate augmented samples from a directory.
- **`report-pack` / `eval-export`**: Prepare figures/tables for the scientific article.
- **`wizard`**: Interactive, step-by-step menu for the core workflows.
- **`eda-cancer`**: RSNA Breast Cancer Detection exploratory pipeline (CSV/PNG/DICOM inputs).

Common flags across the CLI:
- `--outdir` for outputs
- `--dicom-root` for DICOM roots
- `--cache-mode` for dataset caching

Dataset presets:
- `archive`: DICOMs + `classificacao.csv`
- `mamografias`: PNGs por subpasta com `featureS.txt`
- `patches_completo`: PNGs na raiz com `featureS.txt`

See `CLI_CHEATSHEET.md` for command matrices and common recipes.

## RSNA (Cancer) EDA

The `eda-cancer` subcommand runs the internal RSNA Breast Cancer Detection pipeline.

```bash
mammography eda-cancer -- \
  --csv-dir /path/to/rsna \
  --png-dir /path/to/rsna-256-pngs \
  --dicom-dir /path/to/rsna-dicoms \
  --outdir outputs/rsna_eda
```

## DICOM Caching and Lazy Loading

The pipeline now includes **efficient DICOM loading** with lazy loading and intelligent caching to dramatically reduce memory usage and improve loading times:

- **Lazy Loading**: Defers pixel data loading until needed, reducing memory usage by 50%+
- **LRU Caching**: Caches frequently accessed DICOM files in memory with configurable size
- **Disk Persistence**: Save and restore cache state across sessions
- **Zero Breaking Changes**: Fully backward compatible with existing code

### Quick Start

```python
from mammography.io import DicomReader, DicomLRUCache

# Enable lazy loading (defers pixel data until accessed)
reader = DicomReader(lazy_load=True)
ds = reader.read("path/to/mammogram.dcm")

# Access metadata without loading pixel data
patient_id = ds.PatientID  # Fast, no pixel loading

# Pixel data loads on first access
pixels = ds.pixel_array  # Loading happens here

# Or use LRU cache for repeated access
cache = DicomLRUCache(max_size=100, cache_dir="./dicom_cache")
ds = cache.get("path/to/mammogram.dcm")
print(f"Cache hit rate: {cache.hit_rate:.2%}")
```

**Performance Improvements** (validated by benchmarks):
- Memory reduction: ≥50% for metadata-only operations
- Loading time reduction: ≥80% for cached files
- Cache miss overhead: <5%

See [`docs/dicom_caching.md`](docs/dicom_caching.md) for comprehensive usage guide, examples, and performance tuning.

## Repository Structure

- `src/mammography/`: Main package (data, models, training, tools).
- `src/mammography/commands/`: Internal CLI command modules (invoked by `mammography`).
- `configs/`: YAML presets for datasets and training.
- `Article/`: Scientific article and build assets.
- `tests/`: Unit, integration, performance, and smoke tests.
- `docs/`: Comprehensive documentation including DICOM caching guide.

## Scientific Article Workflow

The article lives in `Article/` and integrates with pipeline outputs via:

- `mammography report-pack` (CLI) or `python -m mammography.tools.report_pack`
- `mammography data-audit` (CLI) or `python -m mammography.tools.data_audit`

See `Article/README.md` for LaTeX build instructions.

## Testing

Some tests require local datasets (DICOM archives, RSNA folders). For dataset-free smoke checks, run:

```bash
python -m pytest \
  tests/unit/test_dicom_validation.py \
  tests/unit/test_dimensionality_reduction.py \
  tests/unit/test_evaluation_metrics.py \
  tests/unit/test_clustering_algorithms.py \
  tests/test_cache_mode.py \
  tests/test_dataset_transforms.py
```

## Disclaimer

⚠️ This is an educational research project. It must NOT be used for clinical or medical diagnostic purposes.
