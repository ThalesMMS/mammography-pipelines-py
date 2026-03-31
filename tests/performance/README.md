# Performance Tests

## Purpose

`tests/performance/` contains benchmark-oriented tests for throughput, memory
usage, cache behavior, and scalability. These tests help validate that
optimizations around DICOM loading and large dataset processing produce
measurable benefits without regressing resource usage.

Use this folder selectively. It is not intended to be the default local test
loop for ordinary feature work.

## What Belongs Here

- Cache hit-rate and lazy-loading benchmarks
- Memory and processing-time checks for larger synthetic workloads
- Resource-oriented acceptance thresholds
- Tests whose main assertion is performance behavior rather than pure correctness

## What Does Not Belong Here

- Small correctness checks that belong in `tests/unit/`
- Interface/schema validation
- Determinism and seed behavior checks
- Generic workflow integration tests without a performance objective

## Notable Contents

- `test_cache_performance.py`
  Benchmarks DICOM cache behavior, lazy loading, cache hit rate, and loading
  time improvements using synthetic DICOM files.
- `test_large_datasets.py`
  Exercises larger synthetic datasets to track memory growth, processing time,
  and GPU-related resource usage during bulk processing scenarios.

Both files use monitoring helpers and acceptance-style thresholds rather than
micro-benchmarks tied to exact machine timing.

## Running This Suite

Run these tests intentionally, usually when changing I/O, caching, or
throughput-sensitive code.

```bash
pytest tests/performance/test_cache_performance.py -v
pytest tests/performance/test_large_datasets.py -v
pytest tests/performance -v
```

For routine local development, prefer `tests/unit/` or smoke-level integration
tests first.

## Dependencies and Runtime Expectations

- Optional dependencies include `numpy`, `psutil`, `pydicom`, and in some cases
  `torch`.
- Tests create temporary synthetic files rather than relying on real datasets.
- Resource usage depends on local hardware, available memory, and CUDA
  availability.
- Some assertions are threshold-based, so avoid tightening limits without a
  clear reason and representative measurements.

## Adding New Tests

- Add a performance test only when the primary requirement is resource or
  latency behavior.
- Use synthetic workloads that are large enough to reveal regressions but still
  bounded enough for repeatable execution.
- Prefer thresholds and relative improvements over brittle exact timings.
- Document why a threshold matters when adding or changing one.

## Related References

- [Top-level testing guide](../../docs/TESTING.md)
- [Integration tests](../integration/README.md)
- [Reproducibility tests](../reproducibility/README.md)
