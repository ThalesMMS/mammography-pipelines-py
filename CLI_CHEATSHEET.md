# CLI Cheatsheet — Mammography Pipelines

Atalho para configurar os fluxos mais comuns da CLI consolidada (`mammography`).

## Dataset presets

- `archive`: DICOMs em `archive/` + `classificacao.csv`.
- `mamografias`: PNGs em subpastas, cada uma com `featureS.txt`.
- `patches_completo`: PNGs na raiz, com `featureS.txt`.

Use `--dataset <preset>` para preencher `--csv` e `--dicom-root` automaticamente. Para manter BI-RADS 5 do `classificacao.csv`, inclua `--include-class-5`.

## Flags comuns

- Modelos: `--arch efficientnet_b0|resnet50`
- Classes: `--classes binary|multiclass|density`
- Caching: `--cache-mode auto|memory|disk|tensor-disk|tensor-memmap`
- Balanceamento: `--class-weights auto|none`, `--sampler-weighted`, `--sampler-alpha`
- Augment: `--augment`, `--augment-vertical`, `--augment-color`, `--augment-rotation-deg`
- Normalizacao: `--mean 0.485,0.456,0.406 --std 0.229,0.224,0.225`
- Projecoes: `--run-reduction` (atalho para PCA/t-SNE/UMAP), `--run-clustering` (atalho para k-means)

## Receitas por dataset

### archive (DICOM + classificacao.csv)
- Treino EfficientNetB0 (binario)
  ```bash
  mammography train-density -- \
    --dataset archive \
    --arch efficientnet_b0 \
    --classes binary \
    --epochs 10 \
    --batch-size 16 \
    --cache-mode auto \
    --gradcam \
    --save-val-preds \
    --export-val-embeddings \
    --outdir outputs/archive_effnet_train
  ```
- Embeddings + projecoes (ResNet50)
  ```bash
  mammography embed -- \
    --dataset archive \
    --arch resnet50 \
    --classes multiclass \
    --run-reduction \
    --run-clustering \
    --outdir outputs/archive_resnet_extract
  ```

### mamografias (featureS.txt em subpastas)
- Treino ResNet50
  ```bash
  mammography train-density -- \
    --dataset mamografias \
    --arch resnet50 \
    --classes multiclass \
    --epochs 10 \
    --batch-size 16 \
    --cache-mode auto \
    --outdir outputs/mamografias_resnet_train
  ```
- Embeddings + projecoes (EfficientNetB0)
  ```bash
  mammography embed -- \
    --dataset mamografias \
    --arch efficientnet_b0 \
    --classes multiclass \
    --run-reduction \
    --run-clustering \
    --outdir outputs/mamografias_effnet_extract
  ```

### patches_completo (featureS.txt na raiz)
- Treino EfficientNetB0
  ```bash
  mammography train-density -- \
    --dataset patches_completo \
    --arch efficientnet_b0 \
    --classes multiclass \
    --epochs 10 \
    --batch-size 32 \
    --cache-mode auto \
    --outdir outputs/patches_effnet_train
  ```
- Embeddings + projecoes (ResNet50)
  ```bash
  mammography embed -- \
    --dataset patches_completo \
    --arch resnet50 \
    --classes multiclass \
    --run-reduction \
    --run-clustering \
    --outdir outputs/patches_resnet_extract
  ```

## Saidas do Stage 1

- `features.npy` + `metadata.csv` (embeddings brutas)
- `joined.csv` (quando `--save-csv`)
- `preview/` com `first_image_loaded.png`, `samples_grid.png`, `labels_distribution.png`

Use `mammography wizard` para montar comandos interativamente.
