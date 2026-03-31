# models

## Purpose
Trainable model architectures and builders for density and cancer-related mammography tasks. This
package defines backbone construction, metadata fusion variants, and view-specific inference
helpers.

## Entry Points and Key Modules
- Model builders in this package are consumed by training, inference, explainability, and web UI
code rather than launched directly.

### Key Files
- `cancer_models.py`: Neural network models for breast cancer detection in mammography images.
- `nets.py`: Backbone builders for EfficientNet, ResNet, ViT, and DeiT variants with optional
metadata-fusion heads.

### Subdirectories
- [`embeddings/`](embeddings/README.md): Backbone-specific embedding extractors and typed vector
helpers.

## How It Fits into the Pipeline
- Creates the neural network instances used in supervised workflows.
- Encapsulates backbone selection and classifier-head construction so commands and trainers can stay
model-agnostic.
- Provides view-specific and ensemble abstractions that support multi-view experiments.

## Inputs and Outputs
- Inputs: architecture choices, pretrained-backbone flags, tensor batches, and optional fusion
features.
- Outputs: initialized PyTorch models, logits/probabilities, and checkpoint-compatible modules for
training and inference.

## Dependencies
- Internal: [`models/embeddings`](embeddings/README.md), [`training`](../training/README.md),
[`vis`](../vis/README.md).
- External: `torch`, `torchvision`, `timm`.

## Extension and Maintenance Notes
- Checkpoint compatibility depends on stable module naming and head definitions; be deliberate when
changing constructor defaults or state-dict structure.
- If a new backbone is used for both supervised classification and standalone embedding extraction,
coordinate the implementation with `models/embeddings`.
- Keep task-specific output dimensions explicit so command modules and trainers do not have to infer
them from model internals.

## Related Directories
- [`models/embeddings`](embeddings/README.md): Backbone-specific embedding extractors and typed
vector helpers.
- [`training`](../training/README.md): Core training and validation logic for supervised mammography
models.
- [`commands`](../commands/README.md): Internal command handlers behind the top-level `mammography`
CLI.
- [`vis`](../vis/README.md): Visualization and explainability package for the mammography pipelines.
