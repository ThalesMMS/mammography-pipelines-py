# Integration Tests

## Purpose

`tests/integration/` validates behavior across multiple subsystems at once. The
tests in this folder focus on workflow-level confidence: CLI routing,
configuration loading, checkpoint handling, preprocessing flows, training
workflows, and other end-to-end or near-end-to-end scenarios.

Use this folder after unit coverage is in place and you need to confirm that
the pieces still work together.

## What Belongs Here

- Multi-module workflows that cross package boundaries
- CLI and command integration that exercises real argument handling
- Smoke tests for critical user-facing flows
- Recovery, checkpoint, and configuration behavior that only makes sense when
  several components interact

## What Does Not Belong Here

- Single-function or single-class behavior that can be isolated in
  `tests/unit/`
- Pure interface/schema contract checks
- Performance benchmarking or reproducibility studies
- General-purpose helper code that should live in `tests/utils/`

## Notable Contents

Representative groups in the current folder:

- Smoke and fast workflow validation:
  `test_smoke_workflows.py`, `test_cli_integration.py`,
  `test_cli_comprehensive.py`
- End-to-end pipeline coverage:
  `test_full_pipeline.py`, `test_batch_inference_e2e.py`,
  `test_cancer_pipeline.py`, `test_parallel_pipeline.py`,
  `test_clustering_pipeline.py`, `test_preprocessing_pipeline.py`
- Robustness and workflow state handling:
  `test_checkpoint_workflows.py`, `test_failure_recovery.py`,
  `test_config_loading.py`, `test_dataset_presets.py`,
  `test_flexible_splits.py`, `test_wizard_smart_defaults.py`
- Task-specific flows:
  `test_automl_command.py`, `test_tune_workflow.py`,
  `test_compare_models_workflow.py`, `test_view_specific_training.py`,
  `test_embedding_extraction.py`, `test_explain_command.py`

## Running This Suite

This folder supports both path-based execution and marker-based narrowing.
Path-based selection is the safest default. Marker filters are useful here
because this folder contains most of the explicit `integration`, `slow`, `cpu`,
and `gpu` marker usage in the repository.

```bash
pytest tests/integration
pytest tests/integration/test_smoke_workflows.py
pytest tests/integration -m "integration and not slow and not gpu"
pytest tests/integration -m gpu -v
```

A good local progression is:

1. `pytest tests/integration/test_smoke_workflows.py`
2. `pytest tests/integration -m "integration and not slow and not gpu"`
3. `pytest tests/integration`

## Dependencies and Runtime Expectations

- Many tests use `pytest.importorskip()` for optional packages such as `torch`,
  `torchvision`, `sklearn`, and `pandas`.
- Some tests are CPU-only, some are GPU-aware, and some are explicitly slow.
- Compared with `tests/unit/`, these tests are more sensitive to configuration,
  CLI defaults, environment differences, and temporary filesystem behavior.

When a change affects command wiring, preprocessing flow composition, or model
workflow orchestration, expect this folder to be the decisive signal.

## Adding New Tests

- Add a test here when validating a user-visible workflow or a code path that
  spans multiple modules.
- Prefer a smoke-style version first when a full pipeline test would be too
  expensive.
- Mark tests accurately when they are slow or require GPU resources.
- Keep synthetic inputs small unless the specific purpose is to validate scale.
- Avoid re-testing low-level branch logic that already belongs in `tests/unit/`.

## Related References

- [Top-level testing guide](../../docs/TESTING.md)
- [Shared fixtures](../conftest.py)
- [Unit tests](../unit/README.md)
- [Performance tests](../performance/README.md)
