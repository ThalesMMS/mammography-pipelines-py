# Unified Mammography Classifier Cheatsheet

`Unified_Mammo_Classifier.py` wraps all repo features in a single CLI: training (binary or BI-RADS 1–4), embeddings extraction and visualization, Grad-CAM, history/metrics export, confusion matrix, and utility subcommands.

## Dataset presets
- `archive`: DICOMs under `archive/` with labels in `classificacao.csv`.
- `mamografias`: PNGs inside subfolders, labels in each `featureS.txt`.
- `patches_completo`: PNG patches and labels in the root `featureS.txt`.

Use `--dataset <name>` to auto-fill the expected paths (`--csv` + `--dicom-root` for archive). You can still pass `--csv` manually for custom locations.

## Key flags
- `--mode`: `train`, `extract`, `eda`, `eval-export`, `report-pack`, `rl-refine`.
- `--model`: `efficientnet_b0` or `resnet50`. Swap to cover both backbones.
- `--task`: `binary` (1–2 vs 3–4) or `multiclass` (1..4).
- Train artifacts: `train_history.csv/png`, `best_model.pt`, `best_metrics.json`, `val_predictions.csv` (with `--save-val-preds`), Grad-CAMs (`--gradcam`), validation embeddings (`--export-val-embeddings`).
- Extract artifacts: `features.npy`, `metadata.csv`, optional PCA/t-SNE/UMAP/clustering tables (`--save-csv --pca --tsne --umap --cluster-auto`), previews in `preview/`.

## Combination matrix (commands)
Swap `--task binary`/`--task multiclass` as needed. All commands emit history charts, confusion matrix, and metrics by default; add `--gradcam` to save overlays during validation.

### archive (DICOM + classificacao.csv)
- Train EfficientNetB0  
  ```bash
  python Unified_Mammo_Classifier.py --mode train --dataset archive --model efficientnet_b0 --task binary --epochs 10 --batch-size 16 --cache-mode auto --gradcam --save-val-preds --export-val-embeddings --outdir outputs/archive_effnet_train
  ```
- Train ResNet50  
  ```bash
  python Unified_Mammo_Classifier.py --mode train --dataset archive --model resnet50 --task binary --epochs 10 --batch-size 16 --cache-mode auto --gradcam --save-val-preds --export-val-embeddings --outdir outputs/archive_resnet_train
  ```
- Extract embeddings + projections (EfficientNet)  
  ```bash
  python Unified_Mammo_Classifier.py --mode extract --dataset archive --model efficientnet_b0 --task multiclass --save-csv --pca --tsne --umap --cluster-auto --batch-size 24 --outdir outputs/archive_effnet_extract
  ```
- Extract embeddings + projections (ResNet)  
  ```bash
  python Unified_Mammo_Classifier.py --mode extract --dataset archive --model resnet50 --task multiclass --save-csv --pca --tsne --umap --cluster-auto --batch-size 24 --outdir outputs/archive_resnet_extract
  ```

### mamografias (featureS.txt inside subfolders)
- Train EfficientNetB0  
  ```bash
  python Unified_Mammo_Classifier.py --mode train --dataset mamografias --model efficientnet_b0 --task multiclass --epochs 10 --batch-size 16 --cache-mode auto --gradcam --save-val-preds --export-val-embeddings --outdir outputs/mamografias_effnet_train
  ```
- Train ResNet50  
  ```bash
  python Unified_Mammo_Classifier.py --mode train --dataset mamografias --model resnet50 --task multiclass --epochs 10 --batch-size 16 --cache-mode auto --gradcam --save-val-preds --export-val-embeddings --outdir outputs/mamografias_resnet_train
  ```
- Extract embeddings + projections (EfficientNet)  
  ```bash
  python Unified_Mammo_Classifier.py --mode extract --dataset mamografias --model efficientnet_b0 --task multiclass --save-csv --pca --tsne --umap --cluster-auto --batch-size 24 --outdir outputs/mamografias_effnet_extract
  ```
- Extract embeddings + projections (ResNet)  
  ```bash
  python Unified_Mammo_Classifier.py --mode extract --dataset mamografias --model resnet50 --task multiclass --save-csv --pca --tsne --umap --cluster-auto --batch-size 24 --outdir outputs/mamografias_resnet_extract
  ```

### patches_completo (featureS.txt in root)
- Train EfficientNetB0  
  ```bash
  python Unified_Mammo_Classifier.py --mode train --dataset patches_completo --model efficientnet_b0 --task multiclass --epochs 10 --batch-size 32 --cache-mode auto --gradcam --save-val-preds --export-val-embeddings --outdir outputs/patches_effnet_train
  ```
- Train ResNet50  
  ```bash
  python Unified_Mammo_Classifier.py --mode train --dataset patches_completo --model resnet50 --task multiclass --epochs 10 --batch-size 32 --cache-mode auto --gradcam --save-val-preds --export-val-embeddings --outdir outputs/patches_resnet_train
  ```
- Extract embeddings + projections (EfficientNet)  
  ```bash
  python Unified_Mammo_Classifier.py --mode extract --dataset patches_completo --model efficientnet_b0 --task multiclass --save-csv --pca --tsne --umap --cluster-auto --batch-size 48 --outdir outputs/patches_effnet_extract
  ```
- Extract embeddings + projections (ResNet)  
  ```bash
  python Unified_Mammo_Classifier.py --mode extract --dataset patches_completo --model resnet50 --task multiclass --save-csv --pca --tsne --umap --cluster-auto --batch-size 48 --outdir outputs/patches_resnet_extract
  ```

### Notes
- Add `--gradcam-limit N` to control how many overlays are saved; outputs land in `<outdir>/gradcam/`.
- `--cache-mode auto` will materialize PNG/tensor caches per dataset size/type; override with `memory`, `disk`, `tensor-disk`, or `tensor-memmap` if you prefer a fixed strategy.
- Use `--class-weights auto` or `--sampler-weighted` to balance classes; `--early-stop-patience` and `--lr-reduce-*` control training patience/decay.

## Native scripts parity
The core scripts now mirror these combinations without relying on `Unified_Mammo_Classifier.py`:
- Train (history + confusion + Grad-CAM + val preds/embeddings):  
  `python mammography/scripts/train.py --dataset archive --arch efficientnet_b0 --classes binary --epochs 10 --batch-size 16 --cache-mode auto --gradcam --save-val-preds --export-val-embeddings --outdir outputs/archive_effnet_train`
- Extract embeddings + PCA/t-SNE/UMAP/k-means:  
  `python mammography/scripts/extract_features.py --dataset mamografias --arch resnet50 --pca --tsne --umap --cluster --save-csv --outdir outputs/mamografias_resnet_extract`

Swap `--dataset` among `archive`, `mamografias`, `patches_completo` and `--arch` between `efficientnet_b0`/`resnet50` to cover the full matrix.
