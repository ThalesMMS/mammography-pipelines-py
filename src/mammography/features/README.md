# features

## Purpose
Small feature-extraction package centered on a ResNet50-based extractor. It appears to represent an
older or narrower abstraction than the multi-backbone extractors in `models/embeddings`.

## Entry Points and Key Modules
- No direct CLI entrypoint exists here; feature extraction commands usually call the richer
extractor stack under `models/embeddings`.

### Key Files
- `extractor.py`: ResNet50-based feature extractor class used to generate embedding vectors from
image batches.

## How It Fits into the Pipeline
- Offers a lightweight, self-contained feature extractor abstraction for experiments that do not
need the full embedding module hierarchy.
- Can serve as a simpler integration point for scripts or tests that only need one backbone.
- Sits conceptually between preprocessing and downstream analysis.

## Inputs and Outputs
- Inputs: image tensors or batches compatible with torchvision backbone preprocessing.
- Outputs: embedding vectors suitable for clustering, visualization, or classical baselines.

## Dependencies
- Internal: [`models/embeddings`](../models/embeddings/README.md),
[`preprocess`](../preprocess/README.md).
- External: `torch`, `torchvision`, `numpy`.

## Extension and Maintenance Notes
- Before adding another extractor here, decide whether it belongs in this lightweight layer or
should be added to `models/embeddings` where the main embedding workflows already live.
- If both packages remain active, keep naming and output-shape conventions aligned so downstream
analysis code can consume either path.
- Avoid duplicating preprocessing rules in this package; rely on shared preprocessing contracts
where possible.

## Related Directories
- [`models/embeddings`](../models/embeddings/README.md): Backbone-specific embedding extractors and
typed vector helpers.
- [`preprocess`](../preprocess/README.md): Image preprocessing abstractions for mammography data.
- [`analysis`](../analysis/README.md): Numerical post-processing helpers for embedding exploration.
- [`commands`](../commands/README.md): Internal command handlers behind the top-level `mammography`
CLI.
