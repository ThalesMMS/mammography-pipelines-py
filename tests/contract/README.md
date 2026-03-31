# Contract Tests

## Purpose

`tests/contract/` defines expected interfaces without depending on a concrete
implementation. These tests act as living documentation for request/response
shapes, configuration payloads, and stable invariants that downstream code is
expected to honor.

Use this folder when the important question is "what must the interface look
like?" rather than "how is it implemented?"

## What Belongs Here

- Schema and payload shape validation
- Interface stability checks for request and response objects
- Invariants that should remain true across refactors
- Implementation-agnostic examples of supported API structures

## What Does Not Belong Here

- Full workflow execution
- Performance or reproducibility validation
- Detailed low-level logic testing for individual helpers
- Generic CLI smoke coverage

## Notable Contents

The current contract coverage is organized around three areas:

- `test_embedding_api.py`
  Defines expected embedding request and response structures, including batch
  payloads, metadata alignment, cache payloads, and embedding dimensionality.
- `test_preprocessing_api.py`
  Captures expected preprocessing input, configuration, and output contracts.
- `test_clustering_api.py`
  Describes clustering request/response shapes, result metadata, and grouping
  invariants.

These tests are especially useful when refactoring modules that exchange nested
dict-like payloads or when planning a future service boundary.

## Running This Suite

```bash
pytest tests/contract
pytest tests/contract/test_embedding_api.py -v
pytest tests/contract -k preprocessing -v
```

Because these tests are mostly structural, they should stay lightweight and be
safe to run frequently.

## Dependencies and Runtime Expectations

- Contract tests should remain lightweight and mostly data-structure driven.
- Avoid unnecessary filesystem, GPU, or large dataset setup.
- If a contract test needs implementation objects at all, keep that dependency
  minimal and focused on validating the interface.

## Adding New Tests

- Add a contract test when you need to lock down an external or cross-module
  payload shape.
- Express expectations clearly with small representative examples.
- Prefer explicit assertions for required keys, types, dimensions, and allowed
  values.
- Keep the tests implementation-agnostic so they remain valid during internal
  refactors.

## Related References

- [Top-level testing guide](../../docs/TESTING.md)
- [Integration tests](../integration/README.md)
- [Unit tests](../unit/README.md)
