# Mammography Pipeline Package

This package consolidates the training and inference logic for the Mammography project.

## Structure

- **`scripts/`**: Entry points.
  - `train.py`: Unified training script for Density (4-class/Binary) and Patches.
- **`src/mammography/`**: Core library.
  - `config.py`: Hyperparameters and configuration.
  - `data/`: Dataset loading and CSV handling.
  - `models/`: Model definitions (EfficientNet, ResNet).
  - `training/`: Training loops and validation engines.
  - `utils/`: Helper functions (logging, seed, DICOM I/O).

## Usage

To train a model (replaces old `RSNA_Mammo_*.py` scripts):

```bash
python mammography/scripts/train.py \
  --csv path/to/data.csv \
  --dicom-root path/to/dicoms \
  --arch efficientnet_b0 \
  --classes density \
  --epochs 10
```

Or via `Projeto.py`:

```bash
python Projeto.py train-density -- --epochs 10
```

