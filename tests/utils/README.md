# Test Utilities

## Purpose

`tests/utils/` is shared test infrastructure. Unlike the other `tests/*`
subfolders, it is not primarily a suite that you run directly. Instead, it
contains reusable helpers that keep test code consistent, shorter, and easier
to maintain across unit, integration, and specialized test folders.

## What Belongs Here

- Reusable assertion helpers with better failure messages
- Synthetic data builders for DICOMs, datasets, and embeddings
- Deterministic sampling utilities used by tests or test-support code

## What Does Not Belong Here

- One-off helpers that are only used by a single test file
- Workflow or end-to-end tests
- Folder-specific fixtures that are better placed in `../conftest.py`
- Production code that belongs under `src/mammography/`

## Notable Contents

- `assertions.py`
  Assertion helpers for tensors, checkpoints, configs, DICOM metadata, arrays,
  files, directories, and model outputs.
- `mock_data.py`
  Synthetic data generators for DICOM datasets, dataset metadata, image files,
  and embedding artifacts.
- `dataset_sampling.py`
  Small deterministic helpers for sampling sequences, paths, and data frames.

These modules are intended to reduce copy-paste setup code and make tests more
descriptive.

## Using These Helpers

Common import patterns:

```python
from tests.utils.assertions import assert_tensor_shape, assert_valid_checkpoint
from tests.utils.mock_data import generate_mock_dicom, generate_mock_dataset
from tests.utils.dataset_sampling import sample_sequence, sample_paths_by_extension
```

Example usage:

```python
dataset = generate_mock_dicom(seed=42)
sampled = sample_sequence(["a", "b", "c", "d"], ratio=0.5, seed=42)
assert len(sampled) == 2
```

When a helper becomes widely useful across test files, this folder is the right
home for it.

## Runtime Expectations

- These modules are imported by tests; they are not the main execution target.
- Optional dependencies vary by helper. For example, some functions expect
  `torch`, `pydicom`, `numpy`, or `pandas`.
- Shared fixtures still belong in `../conftest.py` when the main abstraction is
  a pytest fixture rather than a plain helper function.

## Adding New Helpers

- Add a helper here only when it is reusable across multiple test files or
  folders.
- Keep APIs small and predictable.
- Prefer deterministic defaults such as explicit seeds.
- When adding a new helper, also add or update tests that exercise it through
  the relevant consuming suite.

If a helper is only used once, keep it close to the test that owns it instead of
growing this folder with one-off abstractions.

## Related References

- [Top-level testing guide](../../docs/TESTING.md)
- [Shared fixtures](../conftest.py)
- [Unit tests](../unit/README.md)
