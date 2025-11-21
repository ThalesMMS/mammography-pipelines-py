# Mammography Pipelines

This repository brings together two core scripts for working with mammograms and three supporting subdirectories:

1. `extract_mammo_resnet50.py` - generates embeddings from DICOMs using a pre-trained ResNet50, producing artifacts ready for exploration (NPY/CSV, PCA, t-SNE, clusters).
2. `RSNA_Mammography_EDA.py` - a script-format notebook covering EDA, data preparation, and multimodal training for the RSNA Breast Cancer Detection challenge. The same file also offers an optional fine-tuning pipeline to classify breast density (classes 1-4) by reusing the embedding extractor preprocessing.

## Unified classifier (all datasets/backbones)
- Use `Unified_Mammo_Classifier.py` for a single CLI that trains EfficientNetB0 or ResNet50, extracts embeddings, saves Grad-CAMs, histories/metrics, confusion matrices, and validation predictions/embeddings.
- It understands the three bundled datasets out of the box: `--dataset archive` (DICOM + `classificacao.csv`), `--dataset mamografias` (per-folder `featureS.txt`), and `--dataset patches_completo` (root `featureS.txt`). You can also pass `--csv` manually.
- See `docs/Unified_Mammo_Classifier.md` for a command matrix covering every combination (train/extract × ResNet50/EfficientNet × archive/mamografias/patches_completo).
- Quick example (binary EfficientNet + Grad-CAM on the DICOM archive):
  ```bash
  python Unified_Mammo_Classifier.py --mode train --dataset archive --model efficientnet_b0 --task binary --epochs 10 --batch-size 16 --cache-mode auto --gradcam --save-val-preds --export-val-embeddings --outdir outputs/archive_effnet_train
  ```

Main subdirectories:
- `ResNet50_Test/` - research suite with CLIs to preprocess, extract embeddings, cluster, and analyze breast density.
- `density_classifier/` - quick review app that shows all DICOMs of an exam and records BI-RADS density via keyboard.
- `patch_marking/` - ROI annotation tool that exports PNG crops from DICOM studies using `train.csv`.

Use each script according to your goal: the first focuses on feature extraction and clustering; the second covers the full supervised modeling cycle (cancer) and, on demand, the density classifier.

---

## Requirements
- Python 3.10+
- Install dependencies with:
  ```bash
  pip install -r requirements.txt
  ```
- For the RSNA pipeline you need to download the official Kaggle data (CSV, PNG and, if desired, DICOM).

---

## Pipeline 1 - Embedding Extraction (ResNet50)
File: `extract_mammo_resnet50.py`

- Input: directory with per-exam subfolders containing DICOMs and the CSV `classificacao.csv`.
- Process: reading and preprocessing DICOMs (windowing, resize, fake RGB), inference with ResNet50 (`torchvision`), PCA/t-SNE projections, k-means with automatic K selection.
- Main outputs: `outputs/features.npy`, `outputs/joined.csv`, plots under `outputs/preview/`, clustering metrics/curves.
- How to run (summary):
  ```bash
  python extract_mammo_resnet50.py \
    --data_dir ./archive \
    --csv_path ./classificacao.csv \
    --out_dir ./outputs \
    --save_csv \
    --tsne \
    --weights_path ./resnet50.pth \
    --avoid_download
  ```
- Detailed docs:
  - `ResNet50_Test/QUICKSTART.md` - step-by-step CLI usage for preprocessing, embedding, clustering, and analysis.
  - `ResNet50_Test/docs/embedding_extraction.md` - in-depth description of the embedding flow and artifacts.

---

## Pipeline 2 - RSNA Breast Cancer Detection
File: `RSNA_Mammography_EDA.py`

- Input: Kaggle files (`train.csv`, `test.csv`, 256x256 PNGs or original DICOMs).
- Process: EDA with `lets-plot`, negative-class downsampling, build a PyTorch dataset with images + metadata, fine-tune ResNet50 (1 channel), and compute metrics (accuracy, sensitivity, specificity).
- Outputs: EDA plots and training curves displayed inline (`LetsPlot`), metrics printed to the console across epochs. In addition, the script saves artifacts under `--out_dir` (default: `./outputs_classifier`):
  - `training_history.json`, `classification_metrics.json`, `predictions.csv`
  - `resnet50_density_classifier.pth` (model weights)
- How to run (summary):
  ```bash
  python RSNA_Mammography_EDA.py
  ```
  > Adjust file/directory paths in the script if you are not running in Kaggle's default environment.
- Complementary docs:
  - See the top-of-file docstring and CLI help in `RSNA_Mammography_EDA.py` for path configuration and training options.

### Optional Pipeline - Density Classifier
- Purpose: fit a ResNet50 to BI-RADS 1-4 categories using one DICOM incidence per exam.
- Recommended execution: use the dedicated script `RSNA_Mammo_ResNet50_Density.py` (full CLI, logs, persistent cache). The original functions remain in `RSNA_Mammography_EDA.py`, but the new script simplifies automation.
- Quick start:
  ```bash
  python RSNA_Mammo_ResNet50_Density.py \
    --csv classificacao.csv \
    --dicom-root archive \
    --outdir outputs/mammo_resnet50_density \
    --epochs 10 \
    --batch-size 8 \
    --cache-mode auto \
    --auto-increment-outdir
  ```
  The folder specified by `--outdir` becomes the shared root of the experiment. Each run creates or reuses:
  - `<outdir>/results/` (or `results_1`, `results_2`, ... when `--auto-increment-outdir` is active) for histories, metrics, embeddings, and logs of the current run.
  - `<outdir>/cache/` for decoded PNGs/tensors reused across runs until the folder is removed.

  Adjust arguments as needed (`--warmup-epochs`, `--unfreeze-last-block`, `--cache-mode tensor-memmap`, etc.).

---

## Usage Suggestions
- Run the ResNet50 pipeline to generate reusable embeddings for other models (e.g., classic classifiers or analogous clustering analyses).
- Use the RSNA pipeline to experiment with supervised training strategies with metadata and compare results to those obtained from embeddings alone.

---

## Relevant File Structure
- `extract_mammo_resnet50.py`
- `RSNA_Mammography_EDA.py`
- `ResNet50_Test/QUICKSTART.md`
- `ResNet50_Test/docs/` (e.g., `embedding_extraction.md`, `dicom_processing.md`)
- `requirements.txt`
- `outputs/` (generated after running the ResNet50 pipeline)
- `outputs_classifier/` (generated when running density training inside `RSNA_Mammography_EDA.py`; the dedicated density script uses `--outdir`, defaulting to `outputs/mammo_resnet50_density`)

---

With these materials you can choose the flow that best fits your goal: feature extraction for exploration or supervised training based on the RSNA dataset. If in doubt, consult the corresponding quickstarts or open an issue describing your environment/scenario. Happy experimenting!

---

Use the density pipeline when your goal is to train an explicit parenchymal density classifier. Prefer the dedicated script `RSNA_Mammo_ResNet50_Density.py` for automation, or use the functionality embedded in `RSNA_Mammography_EDA.py` if you want to stay within that script.

---

## Quick Guide (Windows + RTX 5080)

To train with GPU on Windows machines equipped with an RTX 5080 / CUDA 12.8+, follow the flow below. It installs only official packages (PyTorch 2.9 +cu128) in an isolated environment.

1. Prerequisites
   - NVIDIA R570 series driver or later (`nvidia-smi` should show CUDA 12.8/13).
   - Python 3.12 installed (64-bit).

2. Create environment and install dependencies
   ```powershell
   cd D:\mammo
   py -3.12 -m venv venv_gpu_5080
   .\venv_gpu_5080\Scripts\Activate.ps1
   python -m pip install --upgrade pip setuptools wheel

   # Official PyTorch with RTX 5080 support (CUDA 12.8)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

   # Other libraries used by the scripts
   pip install -r requirements-base.txt
   ```

   Verify the GPU is detected:
   ```powershell
   python -c "import torch; print(torch.__version__); print('cuda?', torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
   ```

3. Run the density classifier with GPU
   ```powershell
   python RSNA_Mammo_ResNet50_Density.py ^
     --csv classificacao.csv ^
     --dicom-root archive ^
     --outdir outputs/mammo_resnet50_density_cuda ^
     --epochs 20 ^
     --batch-size 16 ^
     --img-size 512 ^
     --num-workers 8 ^
     --prefetch-factor 4 ^
     --cache-mode disk ^
     --warmup-epochs 2 ^
     --unfreeze-last-block ^
     --class-weights auto ^
     --lr 1e-4 ^
     --backbone-lr 1e-5 ^
     --preset windows ^
     --log-level info
   ```

   > AMP (autocast + GradScaler) is automatically enabled on CUDA/MPS runs to reduce memory usage during forward/backward and during validation/embedding extraction. Use `--no-amp` to keep full FP32. Metrics are still computed after recasting to float32, but small numerical differences (<1e-4) can arise when comparing to purely FP32 runs.

   > `--cache-mode` controls where base images are stored after first access. `disk` materializes the outputs of `dicom_to_pil_rgb` under `outdir/cache/` (saves CPU on DICOM rereads), `memory` replicates the old behavior in RAM, and `none` disables caching. The default `auto` chooses `disk` for datasets coming from DICOM folders, but falls back to `memory`/`none` when the dataset is too large to justify the extra cost or is already in PNG/JPG.

   > The `--preset windows` flag internally limits `num_workers` to 2 and adjusts `prefetch_factor` to reduce `spawn` overhead. Use `--preset mps` (or `--mps-sync-loaders`) on Apple GPUs when you want to compare asynchronous vs synchronous loading.

   The script automatically detects `cuda:0`; add `--device cuda` if you want to force it explicitly.

4. Optional profiling

   The new DataLoader presets (`--preset windows` or `--preset mps`) automatically adjust `num_workers`, `prefetch_factor`, and `persistent_workers`. To compare the impact of each preset, generate dedicated traces with the PyTorch Profiler:

   ```powershell
   # Default run (auto)
   python RSNA_Mammo_ResNet50_Density.py ... --preset auto --profile --profile-dir outputs/profiler_auto

   # Run with Windows heuristics
   python RSNA_Mammo_ResNet50_Density.py ... --preset windows --profile --profile-dir outputs/profiler_windows

   tensorboard --logdir outputs/profiler_auto,outputs/profiler_windows
   ```

   On Apple Silicon machines, use `--preset mps` (optionally combine with `--mps-sync-loaders` for debugging) and compare against the asynchronous mode. This makes it easy to validate whether reducing workers actually improves step time and CPU usage.

5. CPU only

   If you want a CPU-only environment:
   ```powershell
   py -3.12 -m venv venv_cpu
   .\venv_cpu\Scripts\Activate.ps1
   pip install -r requirements-windows-cpu.txt
   python RSNA_Mammo_ResNet50_Density.py ... --device cpu
   ```

This flow avoids the old manual builds and uses only the official binaries already compatible with the Blackwell architecture (sm_120).
