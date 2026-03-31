# io

## Purpose
Low-level image I/O helpers, especially for DICOM handling. This package is responsible for reading,
windowing, caching, and lazily loading imaging data in forms the rest of the pipeline can use.

## Entry Points and Key Modules
- This package is imported throughout the repository; it is infrastructure rather than a standalone
workflow.

### Key Files
- `dicom.py`: DICOM I/O utilities and data models for the mammography pipelines.
- `dicom_cache.py`: LRU cache for DICOM datasets.
- `lazy_dicom.py`: Lazy loading wrapper for DICOM datasets.

## How It Fits into the Pipeline
- Provides the canonical DICOM reading and windowing path for data loading, preprocessing,
visualization, and apps.
- Introduces caching and lazy-loading helpers to keep repeated file access manageable.
- Contains the image-level abstractions that higher-level packages should reuse instead of
reimplementing DICOM logic.

## Inputs and Outputs
- Inputs: DICOM file paths, pydicom datasets, and image loading requests from datasets, apps, or
visualization code.
- Outputs: normalized image arrays, PIL-friendly representations, cached datasets, and typed
wrappers such as `MammographyImage`.

## Dependencies
- Internal: [`data`](../data/README.md), [`preprocess`](../preprocess/README.md),
[`apps`](../apps/README.md), [`vis`](../vis/README.md).
- External: `pydicom`, `numpy`, `Pillow`.

## Extension and Maintenance Notes
- Keep photometric interpretation, rescale, and windowing logic centralized here so training and UI
outputs remain consistent.
- Caching helpers should improve performance without masking corrupt-file behavior; explicit errors
are preferable to silent fallbacks in research pipelines.
- If a new image format needs first-class handling, define the contract here and let higher-level
loaders consume it rather than branching across the codebase.

## Related Directories
- [`data`](../data/README.md): Source of truth for dataset ingestion.
- [`preprocess`](../preprocess/README.md): Image preprocessing abstractions for mammography data.
- [`apps`](../apps/README.md): Umbrella package for operator-facing applications.
- [`vis`](../vis/README.md): Visualization and explainability package for the mammography pipelines.
