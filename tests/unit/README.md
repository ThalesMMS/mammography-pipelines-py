# Unit Tests

## Purpose

`tests/unit/` is the main fast-feedback test surface for the repository. It
contains isolated tests for individual functions, classes, configuration
objects, and command modules without relying on full end-to-end workflows.

Use this folder when you want to validate local behavior quickly, especially
before moving to slower workflow tests in `tests/integration/`.

## What Belongs Here

- Isolated tests for a single module, class, or function
- Tests that use shared fixtures from `../conftest.py` or synthetic data
- Validation of configuration parsing, DICOM helpers, model factories, CLI
  command wiring, and utility behavior
- Edge-case and regression coverage that does not need a full pipeline run

## What Does Not Belong Here

- Multi-step workflow validation across several subsystems
- Performance or scalability benchmarks
- Implementation-agnostic interface/schema tests
- Helper utilities that should live in `../utils/`

## Notable Contents

The folder is intentionally broad. Current files cluster into a few stable
themes:

- Command and CLI tests:
  `test_cli.py`, `test_commands_*.py`, `test_preprocess_command.py`,
  `test_inference_command.py`, `test_train_registry.py`,
  `test_tune_registry.py`
- Config, data, DICOM, and I/O tests:
  `test_config.py`, `test_data.py`, `test_dataset_transforms.py`,
  `test_dicom.py`, `test_dicom_validation.py`, `test_lazy_dicom.py`,
  `test_io_*.py`, `test_csv_mapping.py`, `test_format_detection.py`
- Models, training, tuning, and optimization tests:
  `test_models.py`, `test_train.py`, `test_training_comprehensive.py`,
  `test_lr_finder.py`, `test_lr_schedulers.py`, `test_optuna_tuner.py`,
  `test_cancer_*.py`, `test_vit_*.py`, `test_resnet_*.py`
- Embeddings, clustering, features, and evaluation tests:
  `test_embedding_registry.py`, `test_embeddings_*.py`,
  `test_clustering_*.py`, `test_features*.py`, `test_evaluation_*.py`,
  `test_eval_export*.py`
- Reporting, visualization, and workflow helpers:
  `test_report_pack*.py`, `test_visualization_*.py`,
  `test_smart_defaults.py`, `test_wizard.py`, `test_utils_comprehensive.py`

Specialized documentation for error-path coverage already exists in
[`ERROR_HANDLING_TESTS_README.md`](./ERROR_HANDLING_TESTS_README.md). This
README is the entry point for the folder as a whole and should not duplicate
that file's test-by-test inventory.

## Running This Suite

Path-based selection is the most reliable way to run this folder because not
every unit test file is explicitly marked with `@pytest.mark.unit`.

```bash
pytest tests/unit
pytest tests/unit/test_dicom_validation.py
pytest tests/unit/test_train.py -v
pytest -k "dicom and not integration" tests/unit -v
```

Use the top-level fast path when you want a broader signal across the
repository:

```bash
pytest -m "not slow and not gpu"
```

## Dependencies and Runtime Expectations

- Most tests are expected to run on CPU and complete quickly.
- Many files rely on optional imports guarded with `pytest.importorskip()`,
  especially for `torch`, `pydicom`, `numpy`, `pandas`, and image tooling.
- Shared fixtures come from `../conftest.py`.
- Reusable assertions and synthetic data helpers live in `../utils/`.

Because this folder is large, failures here are often the earliest indication
that a local refactor changed a module contract or a default value.

## Adding New Tests

- Put new tests here when they exercise one module or a tightly scoped unit of
  behavior.
- Prefer synthetic inputs and fixtures over real datasets.
- Keep tests deterministic and independent of execution order.
- Group new files by subject area instead of creating generic names such as
  `test_misc.py`.
- If a helper is broadly reusable, move it to `../utils/` or `../conftest.py`
  instead of duplicating setup code.

## Related References

- [Top-level testing guide](../../docs/TESTING.md)
- [Shared fixtures](../conftest.py)
- [Test utilities](../utils/README.md)
- [Integration tests](../integration/README.md)
