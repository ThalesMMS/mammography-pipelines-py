# preprocess

## Purpose
Image preprocessing abstractions for mammography data. This package standardizes resize,
normalization, and tensor packaging so later stages can consume consistent image inputs.

## Entry Points and Key Modules
- Preprocessing commands and embedding/model pipelines import these helpers rather than executing
this directory directly.

### Key Files
- `image_preprocessor.py`: Image preprocessing module for mammography DICOM data.
- `preprocessed_tensor.py`: PreprocessedTensor model for standardized image data representation.

## How It Fits into the Pipeline
- Normalizes raw imaging data into tensors or structured preprocessed objects.
- Sits between low-level I/O and model-facing code, keeping image-shaping policy centralized.
- Provides reusable preprocessing logic for both embedding extraction and supervised
training/inference.

## Inputs and Outputs
- Inputs: DICOM paths or arrays plus preprocessing configuration such as target size, windowing, and
normalization settings.
- Outputs: preprocessed tensors and metadata structures that downstream models or extractors can
consume directly.

## Dependencies
- Internal: [`io`](../io/README.md), [`data`](../data/README.md),
[`models/embeddings`](../models/embeddings/README.md).
- External: `torch`, `pydicom`, `cv2`, `scipy`, `scikit-image`.

## Extension and Maintenance Notes
- Keep resize and normalization logic consistent across training and embedding workflows; silent
differences here are difficult to diagnose later.
- If preprocessing becomes architecture-specific, keep the shared baseline in this package and layer
model-specific adapters on top.
- Treat output tensor shape and intensity conventions as stable contracts because many downstream
modules assume them implicitly.

## Related Directories
- [`io`](../io/README.md): Low-level image I/O helpers, especially for DICOM handling.
- [`data`](../data/README.md): Source of truth for dataset ingestion.
- [`models/embeddings`](../models/embeddings/README.md): Backbone-specific embedding extractors and
typed vector helpers.
- [`training`](../training/README.md): Core training and validation logic for supervised mammography
models.
