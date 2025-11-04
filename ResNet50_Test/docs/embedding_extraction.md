# Embedding Extraction in Mammography Analysis

**Educational documentation for the Breast Density Exploration pipeline**

**WARNING: This is an EDUCATIONAL RESEARCH project. It must NOT be used for clinical or medical diagnostic purposes. No medical decision should rely on these results.**

## Learning Objectives

This document explains how the project extracts feature embeddings from preprocessed mammography images. You will learn:

1. The structure of ResNet-50 and why it is suitable for transfer learning.
2. The end-to-end embedding extraction pipeline implemented in `src/`.
3. Mathematical concepts that support convolutional feature learning.
4. How grayscale mammograms are adapted to a network pretrained on RGB images.
5. Practical tips for performance, determinism, and troubleshooting.

## ResNet-50 and Transfer Learning

ResNet-50 is a 50-layer convolutional neural network built from residual blocks. Skip connections allow gradients to flow deeper, solving the vanishing gradient problem and enabling stable training of very deep models.

Layer groups (with bottleneck structure `1×1 → 3×3 → 1×1`):

```
Input (3 × 224 × 224)
→ Conv7×7 + MaxPool
→ Block1 (256 channels, 3 residual units)
→ Block2 (512 channels, 4 residual units)
→ Block3 (1024 channels, 6 residual units)
→ Block4 (2048 channels, 3 residual units)
→ Global Average Pooling
→ Fully connected classifier (removed for feature extraction)
```

**Transfer learning** reuses networks trained on ImageNet. Low-level filters (edges, textures) transfer well to medical imaging; we freeze the backbone weights and keep only the average pooling layer to obtain a 2048-dimensional embedding per image.

## Feature Extraction Pipeline

```
Preprocessed mammogram (1 × 512 × 512)
→ channel replication (3 × 512 × 512)
→ resize / centre crop to 224 × 224
→ ImageNet normalisation
→ ResNet-50 backbone
→ 2048-D embedding (avgpool output)
```

Key steps:
1. **Input adaptation** – replicate the single grayscale channel to three channels.
2. **Resolution handling** – resize to 224 × 224 while maintaining aspect ratio (padding when needed).
3. **Normalisation** – apply ImageNet mean and standard deviation per channel.
4. **Forward pass** – run the tensor through the frozen backbone in evaluation mode.
5. **Pooling and flattening** – collect the `avgpool` output and flatten to a vector.
6. **Caching** – optionally store embeddings on disk to accelerate repeated experiments.

Example implementation sketch:
```python
import torch
from torchvision import models

class ResNet50EmbeddingExtractor:
    def __init__(self, device: torch.device):
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.feature_extractor = torch.nn.Sequential(*(list(backbone.children())[:-1]))
        self.feature_extractor.eval().to(device)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.device = device

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.feature_extractor(batch.to(self.device))
        return features.squeeze((-1, -2))  # (batch, 2048)
```

## Mathematical Background

- **Convolution:** `y_{i,j} = Σ_c Σ_m Σ_n w_{c,m,n} · x_{c,i+m,j+n}` captures local patterns.
- **Residual learning:** each block learns a residual function `F(x)` and adds the input `x`, enabling deeper networks.
- **Global Average Pooling:** `z_k = (1/HW) Σ_{i,j} a_{k,i,j}` aggregates spatial activations into channel descriptors.
- **Batch Normalisation:** maintains zero mean and unit variance during training (frozen during inference).

## Handling Grayscale Inputs

Mammograms are single-channel images. The project standardises input using:
- Channel replication (`torch.repeat_interleave`) to obtain three channels.
- Optional intensity augmentation (e.g., histogram equalisation) is disabled during embedding extraction to preserve determinism.
- Photometric interpretation is verified to ensure the breast appears in correct contrast (invert images when `MONOCHROME1`).

## Implementation Details

- Module layout: `src/preprocess/image_preprocessor.py` prepares tensors; `src/cli/embed_cli.py` exposes command-line execution; `src/clustering/clustering_algorithms.py` consumes the embeddings.
- Device selection is handled by `src/utils/device_detection.py`, supporting CUDA, MPS, and CPU.
- Deterministic mode (`torch.use_deterministic_algorithms(True)`) is enabled for reproducibility when feasible.
- Batched inference reduces latency; adjust `batch_size` to match GPU memory.

## Clinical Relevance

The resulting 2048-D embeddings summarise mammography textures, density patterns, and structural cues. They serve as input for clustering, dimensionality reduction, and semi-supervised classifiers. Always validate clusters against clinical knowledge and include the mandatory research disclaimer in reports.

## Performance Optimisation

- Enable mixed precision (AMP) on CUDA hardware when memory-constrained; keep it disabled on MPS due to current framework limitations.
- Cache embeddings in `results/embeddings/` to avoid recomputing when running multiple clustering experiments.
- Profile data loading; use `num_workers` in `DataLoader` to parallelise preprocessing.

## Troubleshooting

- **Dimension mismatch**: confirm resize/cropping outputs 224 × 224 × 3 tensors.
- **Unexpected artefacts**: review preprocessing visualisations before extraction.
- **NaNs in embeddings**: check for zero variance images or invalid normalisation parameters.
- **Slow inference**: ensure the model is on the intended device and gradients are disabled.

Embedding extraction is the link between raw imaging data and downstream analytics. Maintain reproducibility logs (model version, weights, preprocessing configuration) to support future audits and academic publication.
