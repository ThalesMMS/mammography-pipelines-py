# Quick Start Guide — Mammography Pipelines

This guide uses the consolidated CLI (`mammography`) backed by the modular package in `src/mammography`.

## 1) Install

```bash
# Recommended: reproducible install via uv lockfile
pip install uv
uv sync --frozen

# Legacy (pip)
# pip install -r requirements.txt
```

## 2) CLI Help

```bash
mammography --help
```

If you want to run the CLI without installing the entrypoint:

```bash
python -m mammography.cli --help
```

Opcional: abra o assistente interativo:

```bash
mammography wizard
```

## 3) Embeddings

```bash
mammography embed -- \
  --csv classificacao.csv \
  --dicom-root archive \
  --outdir outputs/quickstart_embeddings
```

Dataset presets (auto-preenche `--csv`/`--dicom-root` quando aplicavel):
- `archive`
- `mamografias`
- `patches_completo`

Use `--include-class-5` se quiser manter BI-RADS 5 ao carregar `classificacao.csv`.

## 3b) Baselines classicos (opcional)

```bash
mammography embeddings-baselines -- \
  --embeddings-dir outputs/quickstart_embeddings \
  --outdir outputs/embeddings_baselines
```

## 4) Treinamento de Densidade

```bash
mammography train-density -- \
  --csv classificacao.csv \
  --dicom-root archive \
  --outdir outputs/quickstart_density \
  --epochs 5 \
  --arch resnet50
```

## 5) Visualize Results

```bash
mammography visualize -- \
  --input outputs/quickstart_embeddings/features.npy \
  --outdir outputs/quickstart_visuals \
  --tsne
```

## 6) Article Report Pack

```bash
mammography report-pack --run outputs/quickstart_density/results_1
```

## 7) Tests (Dataset-Free Subset)

```bash
python -m pytest \
  tests/unit/test_dicom_validation.py \
  tests/unit/test_dimensionality_reduction.py \
  tests/unit/test_evaluation_metrics.py \
  tests/unit/test_clustering_algorithms.py \
  tests/test_cache_mode.py \
  tests/test_dataset_transforms.py
```

For full test coverage, additional datasets are required (DICOM archives, RSNA folders).

## Optional: RSNA Breast Cancer EDA

```bash
mammography eda-cancer -- \
  --csv-dir /path/to/rsna \
  --png-dir /path/to/rsna-256-pngs \
  --dicom-dir /path/to/rsna-dicoms \
  --outdir outputs/rsna_eda
```

## Optional: Windows CUDA install (resumo)

```powershell
py -3.12 -m venv venv_gpu
.\venv_gpu\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-base.txt
```

Depois, rode o treino com `--device cuda` e ajuste `--num-workers`/`--prefetch-factor` conforme a sua GPU.
