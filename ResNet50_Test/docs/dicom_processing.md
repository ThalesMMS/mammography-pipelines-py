# DICOM Processing in Mammography Analysis

**Educational documentation for the Breast Density Exploration pipeline**

**WARNING: This is an EDUCATIONAL RESEARCH project. It must NOT be used for clinical or medical diagnostic purposes. No medical decision should rely on these results.**

## Learning Objectives

This guide explains how mammography DICOM data is handled inside the project. After reading it you should:

1. Recall the structure of a DICOM file and the metadata required for breast imaging research.
2. Understand the clinical context for projection types, laterality, and acquisition parameters.
3. Follow each preprocessing step applied before feature extraction.
4. Recognise how technical choices support reproducibility and clinical relevance.
5. Implement or adapt the DICOM processing utilities in `src/io_dicom/` and `src/preprocess/`.

## Table of Contents

1. [DICOM Fundamentals](#dicom-fundamentals)
2. [Key Mammography Tags](#key-mammography-tags)
3. [Preprocessing Pipeline](#preprocessing-pipeline)
4. [Mathematical Details](#mathematical-details)
5. [Clinical Considerations](#clinical-considerations)
6. [Implementation Notes](#implementation-notes)
7. [Troubleshooting](#troubleshooting)

## DICOM Fundamentals

Digital Imaging and Communications in Medicine (DICOM) is the de‑facto standard for storing and transmitting medical images. Each file contains both pixel data and structured metadata, enabling interoperability between equipment vendors and software tools.

A mammography DICOM file comprises:

- **Header**: patient ID, study and series identifiers, acquisition parameters, device manufacturer, compression, and projection information.
- **Pixel Data**: a 12–16-bit grayscale matrix representing the breast image.

Research benefits of relying on DICOM:
- Consistent structure across modalities and vendors.
- Rich contextual metadata for grouping studies and enforcing patient-level splits.
- Built-in integrity checks (transfer syntax, pixel data length, checksum fields).
- Compatibility with imaging viewers and analysis pipelines.

## Key Mammography Tags

The table below lists the tags that must be present for the pipeline to process a study:

| Tag | Field | Description | Usage |
| --- | --- | --- | --- |
| `(0010,0020)` | PatientID | De-identified subject identifier | Prevent leakage and group images by patient |
| `(0020,000D)` | StudyInstanceUID | Unique study ID | Associates CC and MLO projections from same exam |
| `(0020,000E)` | SeriesInstanceUID | Series identifier | Distinguishes repeated acquisitions |
| `(0008,0018)` | SOPInstanceUID | Unique image ID | Deduplication and traceability |
| `(0018,5101)` | ViewPosition | Projection (CC, MLO) | Normalisation and analysis |
| `(0020,0020)` | PatientOrientation | Laterality | Separate left vs right breasts |
| `(0028,0030)` | PixelSpacing | mm/pixel spacing | Spatial measurements |
| `(0028,0101)` | BitsStored | Effective bit depth | Correct intensity scaling |
| `(0008,0070)` | Manufacturer | Device vendor | Bias audits |

Projection definitions:
- **CC (Craniocaudal)**: top-down view used to inspect central tissue; typically labelled “CC”.
- **MLO (Mediolateral Oblique)**: angled lateral view including the axillary tail; typically labelled “MLO”.

Laterality is encoded in `ImageLaterality` or `PatientOrientation`; ensure left (`L`) and right (`R`) breasts remain separated during splitting.

## Preprocessing Pipeline

Processing converts raw DICOM files into normalised tensors suitable for ResNet-50 inference. The canonical pipeline is:

```
DICOM → validation → pixel array → border removal → orientation checks → intensity normalisation → resizing → tensor packaging
```

### 1. Validation

- Confirm that the file is readable by `pydicom` and that mandatory tags are present.
- Verify that the pixel data is 2D grayscale and that bit depth matches expectations.
- Log missing or inconsistent metadata for later review.

```python
from pathlib import Path
import pydicom

REQUIRED_TAGS = [
    "PatientID", "StudyInstanceUID", "SeriesInstanceUID",
    "SOPInstanceUID", "ViewPosition", "PixelSpacing", "BitsStored",
]

def is_valid_mammography(path: Path) -> bool:
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
    except Exception:
        return False
    return all(hasattr(ds, tag) for tag in REQUIRED_TAGS)
```

### 2. Pixel Extraction

- Load the pixel array and convert it to a floating-point representation while preserving the original dynamic range (12–16 bits).
- Apply rescale slope and intercept when provided.
- Record intensity statistics (min/max/percentiles) for later quality checks.

### 3. Border and Artifact Removal

- Detect and crop black background, labels, or calibration markers using thresholding and morphological operations.
- Ensure the compressed breast region remains centred after cropping.

### 4. Orientation Normalisation

- Flip images horizontally so that left breasts appear on a consistent side.
- Align CC and MLO projections by checking `ViewPosition` and orientation tags.

### 5. Intensity Normalisation

- Use per-image z-score normalisation to match the expectations of ResNet-50.
- Keep high bit-depth information by performing operations in floating point before scaling to `[0, 1]` or `[−1, 1]` as required.

### 6. Resizing and Tensor Packaging

- Resize images to `512 × 512` with cubic interpolation, preserving aspect ratio via padding when necessary.
- Stack to 3 channels (grey duplication) before passing to the embedding extractor.
- Convert to `torch.Tensor` and attach metadata (patient, study, projection) for downstream auditing.

## Mathematical Details

- **Z-score normalisation**: `z = (x − μ) / σ`, applied per image to mitigate scanner-specific intensity offsets.
- **Otsu thresholding** or adaptive thresholding may be used for border removal.
- **Affine transforms**: flips and rotations preserve spatial relationships by applying homogeneous matrices.

## Clinical Considerations

- Preprocessing must never remove clinically relevant anatomy; retain the pectoral muscle region in MLO views for density estimation.
- Maintain consistent orientation to support side-by-side comparisons by radiologists.
- Keep metadata that could influence bias analysis (device, compression force, exposure time).
- Document any heuristic thresholds so radiologists can review the rationale.

## Implementation Notes

Relevant modules:
- `src/io_dicom/dicom_reader.py` – low-level file loading and validation.
- `src/io_dicom/mammography_image.py` – rich data structure combining pixel data and metadata.
- `src/preprocess/image_preprocessor.py` – orchestrates the pipeline described above.
- `src/preprocess/preprocessed_tensor.py` – final tensor container used across the project.

Configuration values live in `configs/base.yaml` and can be overridden via CLI arguments.

## Troubleshooting

- **Missing tags**: log the path and exclude the file; request updated exports from the imaging source.
- **Inverted pixels**: check `PhotometricInterpretation` (`MONOCHROME1` vs `MONOCHROME2`) and invert when required.
- **Unexpected bit depth**: confirm `BitsAllocated` and `BitsStored` and adjust scaling logic.
- **Artifacts after cropping**: visualise intermediate results using the visualisation utilities under `src/viz/`.

Accurate preprocessing is essential: any mistake propagates through embedding extraction, clustering, and eventual clinical interpretation.
