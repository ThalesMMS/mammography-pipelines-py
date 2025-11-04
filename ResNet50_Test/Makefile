# Makefile for ResNet50_Test – development and quality tooling

.PHONY: help format lint lint-fix type-check test test-unit test-integration         quality-check quality-fix ci-check clean install dev pre-commit         status dev-setup lint-report coverage-report docs         data-check pipeline-test security-check perf-test repro-check all

PYTHON := python
PIP := pip
BLACK := black
RUFF := ruff
MYPY := mypy
PYTEST := pytest

SRC_DIR := src
SCRIPTS_DIR := scripts
TESTS_DIR := tests

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "[36m%-20s[0m %s
", $$1, $$2}'

install: ## Install runtime dependencies
	$(PIP) install -r requirements.txt

dev: ## Install development dependencies
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev]"

format: ## Format code with Black
	@echo "🔧 Running Black..."
	$(BLACK) $(SRC_DIR)/ $(SCRIPTS_DIR)/ $(TESTS_DIR)/
	@echo "✅ Formatting complete"

lint: ## Run Ruff lint checks
	@echo "🔍 Running Ruff..."
	$(RUFF) check $(SRC_DIR)/ $(SCRIPTS_DIR)/ $(TESTS_DIR)/
	@echo "✅ Linting complete"

lint-fix: ## Auto-fix lint issues with Ruff
	@echo "🔧 Applying Ruff fixes..."
	$(RUFF) check $(SRC_DIR)/ $(SCRIPTS_DIR)/ $(TESTS_DIR)/ --fix
	@echo "✅ Lint fixes applied"

type-check: ## Run MyPy type checks
	@echo "🔍 Running MyPy..."
	$(MYPY) $(SRC_DIR)/ $(SCRIPTS_DIR)/
	@echo "✅ Type checking complete"

test: ## Run full pytest suite with coverage report
	@echo "🧪 Running pytest with coverage..."
	$(PYTEST) $(TESTS_DIR)/ -v --cov=$(SRC_DIR) --cov-report=html --cov-report=term-missing
	@echo "✅ Tests finished"

test-unit: ## Run unit tests only
	@echo "🧪 Running unit tests..."
	$(PYTEST) $(TESTS_DIR)/unit/ -v
	@echo "✅ Unit tests finished"

test-integration: ## Run integration tests only
	@echo "🧪 Running integration tests..."
	$(PYTEST) $(TESTS_DIR)/integration/ -v
	@echo "✅ Integration tests finished"

quality-check: format lint type-check ## Run formatting, linting, and typing checks
	@echo "🎯 Quality checks complete"

quality-fix: format lint-fix ## Format and auto-fix issues
	@echo "🔧 Quality fixes applied"

ci-check: ## Run quality checks for CI (non-destructive)
	@echo "🔍 Running CI checks..."
	$(BLACK) --check $(SRC_DIR)/ $(SCRIPTS_DIR)/ $(TESTS_DIR)/
	$(RUFF) check $(SRC_DIR)/ $(SCRIPTS_DIR)/ $(TESTS_DIR)/
	$(MYPY) $(SRC_DIR)/ $(SCRIPTS_DIR)/
	$(PYTEST) $(TESTS_DIR)/ -v --cov=$(SRC_DIR) --cov-report=xml
	@echo "✅ CI checks complete"

clean: ## Remove temporary files and caches
	@echo "🧹 Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	@echo "✅ Cleanup complete"

pre-commit: quality-fix test ## Prepare code for commit
	@echo "🚀 Code ready for commit"

status: ## Print tool versions
	@echo "📊 Toolchain status:"
	@echo "  - Python: $(shell $(PYTHON) --version)"
	@echo "  - Black: $(shell $(BLACK) --version)"
	@echo "  - Ruff: $(shell $(RUFF) --version)"
	@echo "  - MyPy: $(shell $(MYPY) --version)"
	@echo "  - Pytest: $(shell $(PYTEST) --version)"

dev-setup: install dev ## Configure full development environment
	@echo "🚀 Development environment ready"

lint-report: ## Generate Ruff JSON report
	@echo "📊 Generating Ruff report..."
	$(RUFF) check $(SRC_DIR)/ $(SCRIPTS_DIR)/ $(TESTS_DIR)/ --output-format=json > ruff-report.json
	@echo "✅ Report saved to ruff-report.json"

coverage-report: ## Generate coverage HTML and XML reports
	@echo "📊 Generating coverage reports..."
	$(PYTEST) $(TESTS_DIR)/ --cov=$(SRC_DIR) --cov-report=html --cov-report=xml
	@echo "✅ Coverage reports ready"

docs: ## Generate documentation (placeholder)
	@echo "📚 Documentation available in docs/"

data-check: ## Run medical data safety checks
	@echo "🔍 Checking data safety..."
	$(PYTHON) scripts/check_medical_data.py
	@echo "✅ Data check complete"

pipeline-test: ## Execute the project status check
	@echo "🔄 Running project status check..."
	$(PYTHON) scripts/check_status.py
	@echo "✅ Status check complete"

security-check: ## Run dependency and static security scanners
	@echo "🔒 Running security checks..."
	$(PIP) install safety bandit
	safety check -r requirements.txt
	bandit -r $(SRC_DIR)/ $(SCRIPTS_DIR)/ -f json -o security-report.json
	@echo "✅ Security checks complete"

perf-test: ## Execute performance tests
	@echo "⚡ Running performance tests..."
	$(PYTEST) $(TESTS_DIR)/performance/ -v
	@echo "✅ Performance tests finished"

repro-check: ## Execute reproducibility tests
	@echo "🔄 Running reproducibility tests..."
	$(PYTEST) $(TESTS_DIR)/reproducibility/ -v
	@echo "✅ Reproducibility checks finished"

all: quality-check test ## Run the full quality pipeline
	@echo "🎉 All checks completed successfully"
