# Code Quality and Development Tooling

**WARNING: This is an EDUCATIONAL RESEARCH project. It must NOT be used for clinical or medical diagnostic purposes.**

## Overview

This document outlines the code-quality practices used by ResNet50_Test to keep the project consistent, maintainable, and production-ready for research.

## Tooling Stack

### 1. Black – Opinionated Formatting
- Enforces a single formatting style across the codebase
- Integrates with pre-commit and CI pipelines
- Run `black src tests`

Configuration lives in `pyproject.toml`:
```toml
[tool.black]
line-length = 88
target-version = ["py311"]
```

### 2. Ruff – Fast Linting and Auto-fixes
- Consolidates PyFlakes, isort, and additional rule sets
- Enforces import order, bug detection, complexity limits, and quote styles
- Run `ruff check src tests` (use `--fix` for auto-fixes)

Key configuration in `pyproject.toml`:
```toml
[tool.ruff]
select = ["E", "F", "I", "B", "SIM", "TCH", "Q", "RUF"]
ignore = ["E501"]
line-length = 88
```

### 3. MyPy – Static Type Checking
- Ensures type safety and early bug detection
- Run `mypy src`

`mypy.ini` configures namespace packages, ignored modules, and per-module options.

### 4. Pytest – Automated Tests
- Unit, integration, contract, performance, and reproducibility suites
- Run `pytest tests -v --cov=src`
- Coverage report generated in `htmlcov/`

`pytest.ini` manages pytest markers, warning filters, and logging.

## Recommended Workflow

1. Enable pre-commit hooks to run Black, Ruff, and MyPy before every commit.
2. Use `make quality-fix` to apply formatting, lint fixes, type checks, and tests sequentially.
3. Use `make quality-check` in CI for a non-destructive verification run.

## Quality Metrics

- **Coverage target:** ≥85% on critical modules (preprocessing, embedding, clustering)
- **Ruff runtime:** ~10× faster than pylint, enabling frequent feedback
- **Deterministic linting:** Ruff respects Black configuration to avoid conflicts
- **Incremental typing:** MyPy caches results for faster re-runs

## Troubleshooting

- **Formatting conflicts** → run `black .` before `ruff --fix`
- **Missing optional dependencies** → add to `pyproject.toml` and `requirements.txt`
- **Unrecognised types** → update `mypy.ini` with stubs or `ignore_missing_imports = True`
- **Flaky tests** → use pytest markers and fixtures to isolate stateful components

## Best Practices

- Keep modules cohesive and small; prefer dependency injection for testability.
- Write descriptive docstrings explaining intent, inputs, and failure modes.
- Catch specific exceptions and provide actionable error messages.
- Add regression tests when bugs are fixed to prevent regressions.

## Roadmap Enhancements

1. Integrate SonarQube or CodeQL for deeper static analysis.
2. Add CodeClimate or similar tooling to monitor complexity.
3. Adopt semantic versioning once the project matures beyond the research phase.

Quality is non-negotiable—well-tested and well-documented code accelerates reproducible research and safer collaboration.
