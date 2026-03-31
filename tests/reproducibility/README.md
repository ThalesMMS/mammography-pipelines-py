# Reproducibility Tests

## Purpose

`tests/reproducibility/` verifies deterministic behavior and repeatability. The
focus here is not raw accuracy or throughput, but whether seeded runs and
environment-dependent code paths produce stable, explainable results.

Use this folder when changing random seeds, data ordering, preprocessing
defaults, embedding extraction behavior, or any other logic that can affect
run-to-run consistency.

## What Belongs Here

- Seed handling and deterministic execution checks
- Cross-run consistency validation
- Tolerance-based comparisons for numeric outputs
- Environment capture that helps explain why a run is or is not reproducible

## What Does Not Belong Here

- General correctness tests that do not depend on determinism
- Performance benchmarks
- Interface-only schema checks
- Unbounded nondeterministic experiments with no documented tolerance

## Notable Contents

- `test_reproducibility_validation.py`
  Centralizes reproducibility validation for the current suite. It exercises
  seeded behavior across Python, NumPy, and PyTorch, captures environment
  information, and checks repeated pipeline-style operations for consistent
  outcomes within documented tolerances.

The module imports real project components such as preprocessing, embedding, and
clustering code, so it sits between pure unit testing and full integration.

## Running This Suite

```bash
pytest tests/reproducibility/test_reproducibility_validation.py -v
pytest tests/reproducibility -v
```

If you are debugging a suspected nondeterminism issue, run the folder on the
same machine before comparing results across environments.

## Dependencies and Runtime Expectations

- Optional dependencies include `numpy`, `psutil`, and `torch`.
- Results may vary by platform, CUDA availability, and library versions even
  when the code is correct.
- Assertions should use explicit tolerances where exact equality is unrealistic.

Treat failures here as either a real determinism regression or a sign that the
test needs a clearly justified environment-specific bound.

## Adding New Tests

- Set seeds explicitly for every random subsystem you rely on.
- Record tolerances in the test when comparing floating-point outputs.
- Capture enough environment context to explain cross-machine differences.
- Keep the distinction clear between "bitwise identical", "numerically close",
  and "functionally stable".

## Related References

- [Top-level testing guide](../../docs/TESTING.md)
- [Performance tests](../performance/README.md)
- [Unit tests](../unit/README.md)
