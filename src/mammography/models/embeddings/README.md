# models/embeddings

## Purpose
Backbone-specific embedding extractors and typed vector helpers. This package is the main home for
feature extraction across multiple architectures such as ResNet, EfficientNet, and ViT.

## Entry Points and Key Modules
- Embedding extraction commands import these classes and factories; there is no standalone script in
this directory.

### Key Files
- `efficientnet_extractor.py`: EfficientNet-B0 embedding extractor for mammography feature
extraction.
- `embedding_vector.py`: EmbeddingVector model for CNN feature representation.
- `resnet50_extractor.py`: ResNet-50 embedding extractor for mammography feature extraction.
- `vit_extractor.py`: Vision Transformer (ViT) embedding extractor for mammography feature
extraction.

## How It Fits into the Pipeline
- Turns preprocessed images into fixed-dimensional representations for analysis, baselines, and
downstream tasks.
- Provides a typed abstraction around extracted vectors so pipeline code can reason about metadata
and dimensions.
- Lets the repository support multiple backbones without pushing architecture-specific logic into
the commands layer.

## Inputs and Outputs
- Inputs: preprocessed image tensors, batch iterables, and backbone-specific runtime options.
- Outputs: embedding tensors or `EmbeddingVector` objects that can be serialized into files such as
`features.npy` and `metadata.csv`.

## Dependencies
- Internal: [`models`](../README.md), [`preprocess`](../../preprocess/README.md),
[`analysis`](../../analysis/README.md), [`features`](../../features/README.md).
- External: `torch`, `torchvision`, `numpy`.

## Extension and Maintenance Notes
- Feature dimensionality and preprocessing assumptions are part of the practical contract here;
changing them requires coordinating with extraction, analysis, and baseline code.
- Add new backbones through the same factory-style interface so command modules can stay declarative
about architecture choice.
- When serializing embeddings, keep metadata rich enough to support path- and accession-based lookup
in `data/dataset.py`.

## Related Directories
- [`models`](../README.md): Trainable model architectures and builders for density and cancer-
related mammography tasks.
- [`preprocess`](../../preprocess/README.md): Image preprocessing abstractions for mammography data.
- [`features`](../../features/README.md): Small feature-extraction package centered on a
ResNet50-based extractor.
- [`analysis`](../../analysis/README.md): Numerical post-processing helpers for embedding
exploration.
