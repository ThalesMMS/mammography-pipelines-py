# Coverage Tests

## Purpose

`tests/coverage/` is responsible for coverage-analysis logic that lives inside
the repository's test suite. This folder does not replace `pytest --cov`.
Instead, it validates coverage-related helpers, targets, and categorization
logic used to reason about how well the codebase is exercised.

Use this folder when changing coverage tooling, target thresholds, or the way
source modules are grouped for reporting.

## What Belongs Here

- Tests and helpers for coverage-analysis behavior
- Coverage target definitions and category bookkeeping
- Reporting logic that interprets coverage results
- Recommendations or summaries derived from measured coverage data

## What Does Not Belong Here

- Normal unit coverage for source modules
- General documentation checks
- Performance or reproducibility validation
- CI-only policy text with no executable behavior

## Notable Contents

- `test_coverage_analysis.py`
  Defines the current coverage analyzer, coverage target categories, and report
  generation behavior. It is the place to update when source layout changes make
  existing category mappings or target expectations obsolete.

The current analyzer distinguishes category-level targets such as core
preprocessing, utility modules, and overall coverage behavior.

## Running This Suite

Run the folder test when you are changing the repository's coverage-analysis
logic:

```bash
pytest tests/coverage/test_coverage_analysis.py -v
```

Run project-wide coverage separately when you need the actual coverage report:

```bash
pytest --cov=mammography --cov-report=term-missing --cov-fail-under=50
```

Those commands serve different purposes and should not be treated as
interchangeable.

## Dependencies and Runtime Expectations

- Requires the `coverage` package.
- Coverage analysis code may spawn `pytest` or inspect measured source files as
  part of report generation.
- Folder-local targets should stay aligned with project policy documented in the
  top-level testing guide and CI configuration.

## Adding New Tests

- Update this folder when coverage categories or thresholds change.
- Keep category names and module mappings understandable to contributors.
- Avoid hard-coding source paths that are likely to drift unless the test is
  specifically meant to catch that drift.
- When policy changes, update both executable tests and the human-facing docs.

## Related References

- [Top-level testing guide](../../docs/TESTING.md)
- [Project root README](../../README.md)
- [Unit tests](../unit/README.md)
