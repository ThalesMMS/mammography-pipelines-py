# Pipeline CLI Cheatsheet (Unified script retired)

`Unified_Mammo_Classifier.py` was consolidated into the native CLIs. The same functionality now lives in:
- `mammography/scripts/train.py` — training (binary or BI-RADS 1–4) with EfficientNetB0/ResNet50, Grad-CAM, history/metrics, validation predictions/embeddings, cache/loader heuristics, warmup/backbone LR, and optional profiling.
- `mammography/scripts/extract_features.py` — embedding extraction + PCA/t-SNE/UMAP/k-means, joined CSVs, sample previews (`preview/`), and example embeddings.

## Dataset presets
- `archive`: DICOMs under `archive/` with labels in `classificacao.csv`.
- `mamografias`: PNGs inside subfolders, labels in each `featureS.txt`.
- `patches_completo`: PNG patches and labels in the root `featureS.txt`.

Pass `--dataset <name>` to auto-fill paths (`--csv` + `--dicom-root` for archive) or point to custom locations with `--csv`/`--dicom-root`. Add `--include-class-5` if you need BI-RADS 5 rows from `classificacao.csv`.

## Common flags
- Models: `--arch efficientnet_b0|resnet50`
- Tasks: `--classes binary|multiclass|density` (`binary` maps 1/2 vs 3/4)
- Caching/loader: `--cache-mode auto|memory|disk|tensor-disk|tensor-memmap`, `--num-workers`, `--prefetch-factor`, `--loader-heuristics/--no-loader-heuristics`
- Balancing: `--class-weights auto`, `--sampler-weighted`
- Extras: `--gradcam`, `--save-val-preds`, `--export-val-embeddings`, `--amp`, `--warmup-epochs`, `--backbone-lr`, `--lr-reduce-*`, `--torch-compile`, `--profile`

## Command matrix (swap `--classes binary|multiclass` as needed)

### archive (DICOM + classificacao.csv)
- Train EfficientNetB0  
  ```bash
  python mammography/scripts/train.py --dataset archive --arch efficientnet_b0 --classes binary --epochs 10 --batch-size 16 --cache-mode auto --gradcam --save-val-preds --export-val-embeddings --outdir outputs/archive_effnet_train
  ```
- Train ResNet50  
  ```bash
  python mammography/scripts/train.py --dataset archive --arch resnet50 --classes binary --epochs 10 --batch-size 16 --cache-mode auto --gradcam --save-val-preds --export-val-embeddings --outdir outputs/archive_resnet_train
  ```
- Extract embeddings + projections (EfficientNet)  
  ```bash
  python mammography/scripts/extract_features.py --dataset archive --arch efficientnet_b0 --classes multiclass --save-csv --pca --tsne --umap --cluster --batch-size 24 --outdir outputs/archive_effnet_extract
  ```
- Extract embeddings + projections (ResNet)  
  ```bash
  python mammography/scripts/extract_features.py --dataset archive --arch resnet50 --classes multiclass --save-csv --pca --tsne --umap --cluster --batch-size 24 --outdir outputs/archive_resnet_extract
  ```

### mamografias (featureS.txt inside subfolders)
- Train EfficientNetB0  
  ```bash
  python mammography/scripts/train.py --dataset mamografias --arch efficientnet_b0 --classes multiclass --epochs 10 --batch-size 16 --cache-mode auto --gradcam --save-val-preds --export-val-embeddings --outdir outputs/mamografias_effnet_train
  ```
- Train ResNet50  
  ```bash
  python mammography/scripts/train.py --dataset mamografias --arch resnet50 --classes multiclass --epochs 10 --batch-size 16 --cache-mode auto --gradcam --save-val-preds --export-val-embeddings --outdir outputs/mamografias_resnet_train
  ```
- Extract embeddings + projections (EfficientNet)  
  ```bash
  python mammography/scripts/extract_features.py --dataset mamografias --arch efficientnet_b0 --classes multiclass --save-csv --pca --tsne --umap --cluster --batch-size 24 --outdir outputs/mamografias_effnet_extract
  ```
- Extract embeddings + projections (ResNet)  
  ```bash
  python mammography/scripts/extract_features.py --dataset mamografias --arch resnet50 --classes multiclass --save-csv --pca --tsne --umap --cluster --batch-size 24 --outdir outputs/mamografias_resnet_extract
  ```

### patches_completo (featureS.txt in root)
- Train EfficientNetB0  
  ```bash
  python mammography/scripts/train.py --dataset patches_completo --arch efficientnet_b0 --classes multiclass --epochs 10 --batch-size 32 --cache-mode auto --gradcam --save-val-preds --export-val-embeddings --outdir outputs/patches_effnet_train
  ```
- Train ResNet50  
  ```bash
  python mammography/scripts/train.py --dataset patches_completo --arch resnet50 --classes multiclass --epochs 10 --batch-size 32 --cache-mode auto --gradcam --save-val-preds --export-val-embeddings --outdir outputs/patches_resnet_train
  ```
- Extract embeddings + projections (EfficientNet)  
  ```bash
  python mammography/scripts/extract_features.py --dataset patches_completo --arch efficientnet_b0 --classes multiclass --save-csv --pca --tsne --umap --cluster --batch-size 48 --outdir outputs/patches_effnet_extract
  ```
- Extract embeddings + projections (ResNet)  
  ```bash
  python mammography/scripts/extract_features.py --dataset patches_completo --arch resnet50 --classes multiclass --save-csv --pca --tsne --umap --cluster --batch-size 48 --outdir outputs/patches_resnet_extract
  ```

### Notes
- Previews (`first_image_loaded.png`, `samples_grid.png`, label histograms) live in `<outdir>/preview` after extraction.
- Grad-CAMs land in `<outdir>/gradcam/`; tune via `--gradcam-limit`.
- `Projeto.py` exposes the same flows plus `eval-export`/`report-pack` and the RL stub:  
  `python Projeto.py embed|train-density|eval-export|report-pack|rl-refine ...`
