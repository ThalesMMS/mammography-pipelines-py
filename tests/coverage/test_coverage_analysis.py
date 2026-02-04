"""
Code coverage analysis and improvement for mammography analysis pipeline.

This module provides comprehensive code coverage analysis, reporting,
and improvement tools for the breast density exploration pipeline.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
No medical decision should be based on these results.

Educational Context:
- Code coverage analysis ensures comprehensive testing of all code paths
- Coverage reporting provides visibility into testing completeness
- Coverage improvement identifies areas needing additional testing
- High coverage ensures reliability and maintainability of research code

Author: Research Team
Version: 1.0.0
"""

import json
import logging
from pathlib import Path
import subprocess
from typing import Any, Dict, List

import coverage

# Configure logging
logger = logging.getLogger(__name__)

# Coverage targets
COVERAGE_TARGETS = {
    "core_preprocessing": 85,  # >85% for core preprocessing modules
    "utility_modules": 50,  # >50% for utility modules
    "overall": 70,  # >70% overall coverage
}

# Module categories for coverage analysis
MODULE_CATEGORIES = {
    "core_preprocessing": [
        "mammography/io/dicom.py",
        "src.preprocess.image_preprocessor",
        "src.models.embeddings.resnet50_extractor",
        "src.clustering.clustering_algorithms",
        "src.eval.clustering_evaluator",
    ],
    "utility_modules": [
        "src.utils.patient",
        "src.config.config_models",
        "src.viz.cluster_visualizer",
    ],
    "pipeline_modules": [
        "src.pipeline.mammography_pipeline",
        "src.cli.preprocess_cli",
        "src.cli.embed_cli",
        "src.cli.cluster_cli",
        "src.cli.analyze_cli",
    ],
}


class CoverageAnalyzer:
    """
    Code coverage analyzer for mammography analysis pipeline.

    This class provides comprehensive code coverage analysis, reporting,
    and improvement tools for the research pipeline.

    Educational Notes:
    - Coverage analysis ensures comprehensive testing of all code paths
    - Coverage reporting provides visibility into testing completeness
    - Coverage improvement identifies areas needing additional testing
    - High coverage ensures reliability and maintainability of research code
    """

    def __init__(self, source_dir: str = "src", test_dir: str = "tests"):
        """Initialize coverage analyzer."""
        self.source_dir = Path(source_dir)
        self.test_dir = Path(test_dir)
        self.coverage_data = None
        self.coverage_report = None

    def run_coverage_analysis(self, test_command: str = "pytest") -> Dict[str, Any]:
        """Run comprehensive coverage analysis."""
        logger.info("Starting coverage analysis...")

        # Initialize coverage
        cov = coverage.Coverage(
            source=[str(self.source_dir)],
            omit=["*/tests/*", "*/test_*", "*/__pycache__/*", "*/venv/*", "*/env/*"],
        )

        # Start coverage
        cov.start()

        try:
            # Run tests
            result = subprocess.run(
                test_command.split(), capture_output=True, text=True, cwd=Path.cwd()
            )

            # Stop coverage
            cov.stop()
            cov.save()

            # Generate coverage report
            self.coverage_data = self._generate_coverage_report(cov)

            logger.info(f"Coverage analysis completed. Exit code: {result.returncode}")
            return self.coverage_data

        except Exception as e:
            logger.error(f"Coverage analysis failed: {e}")
            cov.stop()
            raise

    def _generate_coverage_report(self, cov: coverage.Coverage) -> Dict[str, Any]:
        """Generate comprehensive coverage report."""
        # Get coverage data
        total_lines = cov.total()
        covered_lines = cov.covered()
        missing_lines = cov.missing()

        # Calculate overall coverage
        overall_coverage = (covered_lines / total_lines * 100) if total_lines > 0 else 0

        # Get coverage by module
        module_coverage = {}
        for module_name in cov.get_data().measured_files():
            if module_name.startswith(str(self.source_dir)):
                # Get module coverage
                module_cov = cov.analysis(module_name)
                if module_cov:
                    module_total = len(module_cov[1]) + len(module_cov[2])
                    module_covered = len(module_cov[1])
                    module_coverage[module_name] = {
                        "total_lines": module_total,
                        "covered_lines": module_covered,
                        "missing_lines": len(module_cov[2]),
                        "coverage_percent": (
                            (module_covered / module_total * 100)
                            if module_total > 0
                            else 0
                        ),
                    }

        # Categorize modules
        categorized_coverage = self._categorize_module_coverage(module_coverage)

        # Generate report
        report = {
            "overall": {
                "total_lines": total_lines,
                "covered_lines": covered_lines,
                "missing_lines": missing_lines,
                "coverage_percent": overall_coverage,
            },
            "by_module": module_coverage,
            "by_category": categorized_coverage,
            "targets_met": self._check_coverage_targets(categorized_coverage),
            "recommendations": self._generate_recommendations(categorized_coverage),
        }

        return report

    def _categorize_module_coverage(
        self, module_coverage: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Categorize modules by type and calculate category coverage."""
        categorized = {
            "core_preprocessing": {
                "modules": [],
                "total_coverage": 0,
                "module_count": 0,
            },
            "utility_modules": {"modules": [], "total_coverage": 0, "module_count": 0},
            "pipeline_modules": {"modules": [], "total_coverage": 0, "module_count": 0},
        }

        for module_name, coverage_data in module_coverage.items():
            # Determine category
            category = None
            for cat_name, cat_modules in MODULE_CATEGORIES.items():
                if any(cat_module in module_name for cat_module in cat_modules):
                    category = cat_name
                    break

            if category:
                categorized[category]["modules"].append(
                    {
                        "name": module_name,
                        "coverage_percent": coverage_data["coverage_percent"],
                    }
                )
                categorized[category]["total_coverage"] += coverage_data[
                    "coverage_percent"
                ]
                categorized[category]["module_count"] += 1

        # Calculate average coverage for each category
        for category in categorized:
            if categorized[category]["module_count"] > 0:
                categorized[category]["average_coverage"] = (
                    categorized[category]["total_coverage"]
                    / categorized[category]["module_count"]
                )
            else:
                categorized[category]["average_coverage"] = 0

        return categorized

    def _check_coverage_targets(
        self, categorized_coverage: Dict[str, Any]
    ) -> Dict[str, bool]:
        """Check if coverage targets are met."""
        targets_met = {}

        for category, target in COVERAGE_TARGETS.items():
            if category in categorized_coverage:
                actual_coverage = categorized_coverage[category]["average_coverage"]
                targets_met[category] = actual_coverage >= target
            else:
                targets_met[category] = False

        return targets_met

    def _generate_recommendations(
        self, categorized_coverage: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations for improving coverage."""
        recommendations = []

        for category, target in COVERAGE_TARGETS.items():
            if category in categorized_coverage:
                actual_coverage = categorized_coverage[category]["average_coverage"]
                if actual_coverage < target:
                    recommendations.append(
                        f"{category}: Current coverage {actual_coverage:.1f}% "
                        f"is below target {target}%. Add more tests."
                    )

        # Check for modules with very low coverage
        for category, data in categorized_coverage.items():
            for module in data["modules"]:
                if module["coverage_percent"] < 30:
                    recommendations.append(
                        f"Module {module['name']} has very low coverage "
                        f"({module['coverage_percent']:.1f}%). Consider adding comprehensive tests."
                    )

        return recommendations

    def generate_html_report(self, output_dir: str = "coverage_html") -> str:
        """Generate HTML coverage report."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Run coverage html command
        result = subprocess.run(
            ["coverage", "html", "-d", str(output_path)], capture_output=True, text=True
        )

        if result.returncode == 0:
            logger.info(f"HTML coverage report generated in {output_path}")
            return str(output_path)
        else:
            logger.error(f"Failed to generate HTML report: {result.stderr}")
            raise RuntimeError(f"HTML report generation failed: {result.stderr}")

    def generate_json_report(self, output_file: str = "coverage_report.json") -> str:
        """Generate JSON coverage report."""
        if self.coverage_data is None:
            raise RuntimeError(
                "No coverage data available. Run coverage analysis first."
            )

        output_path = Path(output_file)
        with open(output_path, "w") as f:
            json.dump(self.coverage_data, f, indent=2)

        logger.info(f"JSON coverage report generated: {output_path}")
        return str(output_path)


class CoverageImprovement:
    """
    Coverage improvement tools for mammography analysis pipeline.

    This class provides tools for identifying and implementing coverage
    improvements, including missing test cases and edge cases.

    Educational Notes:
    - Coverage improvement ensures comprehensive testing of all code paths
    - Missing test cases identify areas needing additional testing
    - Edge case testing ensures robust error handling
    - Coverage improvement enhances code reliability and maintainability
    """

    def __init__(self, coverage_analyzer: CoverageAnalyzer):
        """Initialize coverage improvement tools."""
        self.analyzer = coverage_analyzer
        self.improvement_plan = None

    def analyze_missing_coverage(self) -> Dict[str, Any]:
        """Analyze missing coverage and generate improvement plan."""
        if self.analyzer.coverage_data is None:
            raise RuntimeError(
                "No coverage data available. Run coverage analysis first."
            )

        missing_coverage = {
            "low_coverage_modules": [],
            "missing_test_cases": [],
            "edge_cases_needed": [],
            "improvement_priorities": [],
        }

        # Identify modules with low coverage
        for module_name, coverage_data in self.analyzer.coverage_data[
            "by_module"
        ].items():
            if coverage_data["coverage_percent"] < 70:
                missing_coverage["low_coverage_modules"].append(
                    {
                        "module": module_name,
                        "coverage_percent": coverage_data["coverage_percent"],
                        "missing_lines": coverage_data["missing_lines"],
                    }
                )

        # Identify missing test cases
        missing_coverage["missing_test_cases"] = self._identify_missing_test_cases()

        # Identify edge cases needed
        missing_coverage["edge_cases_needed"] = self._identify_edge_cases()

        # Generate improvement priorities
        missing_coverage["improvement_priorities"] = (
            self._generate_improvement_priorities(missing_coverage)
        )

        self.improvement_plan = missing_coverage
        return missing_coverage

    def _identify_missing_test_cases(self) -> List[Dict[str, Any]]:
        """Identify missing test cases."""
        missing_test_cases = []

        # Check for missing test files
        test_files = list(self.analyzer.test_dir.rglob("test_*.py"))
        source_files = list(self.analyzer.source_dir.rglob("*.py"))

        for source_file in source_files:
            if source_file.name == "__init__.py":
                continue

            # Check if corresponding test file exists
            test_file_name = f"test_{source_file.name}"
            test_file_path = self.analyzer.test_dir / test_file_name

            if not test_file_path.exists():
                missing_test_cases.append(
                    {
                        "type": "missing_test_file",
                        "source_file": str(source_file),
                        "expected_test_file": str(test_file_path),
                        "priority": "high",
                    }
                )

        return missing_test_cases

    def _identify_edge_cases(self) -> List[Dict[str, Any]]:
        """Identify edge cases that need testing."""
        edge_cases = []

        # Common edge cases for medical imaging
        edge_cases.extend(
            [
                {
                    "type": "invalid_input",
                    "description": "Test with invalid DICOM files",
                    "priority": "high",
                    "module": "dicom_reader",
                },
                {
                    "type": "empty_data",
                    "description": "Test with empty datasets",
                    "priority": "medium",
                    "module": "clustering_algorithms",
                },
                {
                    "type": "memory_limits",
                    "description": "Test with memory constraints",
                    "priority": "high",
                    "module": "image_preprocessor",
                },
                {
                    "type": "gpu_unavailable",
                    "description": "Test when GPU is not available",
                    "priority": "medium",
                    "module": "resnet50_extractor",
                },
                {
                    "type": "network_failure",
                    "description": "Test with network failures",
                    "priority": "low",
                    "module": "pipeline",
                },
            ]
        )

        return edge_cases

    def _generate_improvement_priorities(
        self, missing_coverage: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate improvement priorities."""
        priorities = []

        # High priority: Core modules with low coverage
        for module in missing_coverage["low_coverage_modules"]:
            if module["coverage_percent"] < 50:
                priorities.append(
                    {
                        "priority": "high",
                        "type": "module_coverage",
                        "module": module["module"],
                        "action": f"Increase coverage from {module['coverage_percent']:.1f}% to >70%",
                        "estimated_effort": "high",
                    }
                )

        # Medium priority: Missing test files
        for test_case in missing_coverage["missing_test_cases"]:
            if test_case["type"] == "missing_test_file":
                priorities.append(
                    {
                        "priority": "medium",
                        "type": "missing_test_file",
                        "file": test_case["source_file"],
                        "action": f"Create test file: {test_case['expected_test_file']}",
                        "estimated_effort": "medium",
                    }
                )

        # Low priority: Edge cases
        for edge_case in missing_coverage["edge_cases_needed"]:
            if edge_case["priority"] == "high":
                priorities.append(
                    {
                        "priority": "high",
                        "type": "edge_case",
                        "description": edge_case["description"],
                        "action": f"Add edge case testing for {edge_case['module']}",
                        "estimated_effort": "medium",
                    }
                )

        return priorities

    def generate_improvement_report(
        self, output_file: str = "coverage_improvement_report.json"
    ) -> str:
        """Generate coverage improvement report."""
        if self.improvement_plan is None:
            self.analyze_missing_coverage()

        output_path = Path(output_file)
        with open(output_path, "w") as f:
            json.dump(self.improvement_plan, f, indent=2)

        logger.info(f"Coverage improvement report generated: {output_path}")
        return str(output_path)


class TestCoverageValidation:
    """
    Test coverage validation for mammography analysis pipeline.

    This class provides comprehensive coverage validation testing
    to ensure coverage targets are met and maintained.

    Educational Notes:
    - Coverage validation ensures testing completeness
    - Coverage targets provide measurable goals for testing
    - Coverage validation maintains code quality standards
    - Coverage validation supports research code reliability
    """

    def __init__(self):
        """Initialize coverage validation."""
        self.analyzer = CoverageAnalyzer()
        self.improvement = None

    def test_coverage_targets_met(self):
        """Test that coverage targets are met."""
        # Run coverage analysis
        coverage_data = self.analyzer.run_coverage_analysis()

        # Check targets
        targets_met = coverage_data["targets_met"]

        # Validate core preprocessing coverage
        assert targets_met.get("core_preprocessing", False), (
            f"Core preprocessing coverage target not met. "
            f"Current: {coverage_data['by_category']['core_preprocessing']['average_coverage']:.1f}%, "
            f"Target: {COVERAGE_TARGETS['core_preprocessing']}%"
        )

        # Validate utility modules coverage
        assert targets_met.get("utility_modules", False), (
            f"Utility modules coverage target not met. "
            f"Current: {coverage_data['by_category']['utility_modules']['average_coverage']:.1f}%, "
            f"Target: {COVERAGE_TARGETS['utility_modules']}%"
        )

        # Validate overall coverage
        overall_coverage = coverage_data["overall"]["coverage_percent"]
        assert overall_coverage >= COVERAGE_TARGETS["overall"], (
            f"Overall coverage target not met. "
            f"Current: {overall_coverage:.1f}%, Target: {COVERAGE_TARGETS['overall']}%"
        )

        logger.info("All coverage targets met!")

    def test_coverage_improvement(self):
        """Test coverage improvement capabilities."""
        # Initialize improvement tools
        self.improvement = CoverageImprovement(self.analyzer)

        # Analyze missing coverage
        improvement_plan = self.improvement.analyze_missing_coverage()

        # Validate improvement plan
        assert (
            "low_coverage_modules" in improvement_plan
        ), "Improvement plan missing low coverage modules"
        assert (
            "missing_test_cases" in improvement_plan
        ), "Improvement plan missing test cases"
        assert (
            "edge_cases_needed" in improvement_plan
        ), "Improvement plan missing edge cases"
        assert (
            "improvement_priorities" in improvement_plan
        ), "Improvement plan missing priorities"

        # Generate improvement report
        report_path = self.improvement.generate_improvement_report()
        assert Path(report_path).exists(), "Improvement report not generated"

        logger.info("Coverage improvement analysis completed!")

    def test_coverage_reporting(self):
        """Test coverage reporting capabilities."""
        # Run coverage analysis
        coverage_data = self.analyzer.run_coverage_analysis()

        # Generate HTML report
        html_report_path = self.analyzer.generate_html_report()
        assert Path(html_report_path).exists(), "HTML report not generated"

        # Generate JSON report
        json_report_path = self.analyzer.generate_json_report()
        assert Path(json_report_path).exists(), "JSON report not generated"

        # Validate report content
        with open(json_report_path, "r") as f:
            report_data = json.load(f)

        assert "overall" in report_data, "Report missing overall coverage"
        assert "by_module" in report_data, "Report missing module coverage"
        assert "by_category" in report_data, "Report missing category coverage"
        assert "targets_met" in report_data, "Report missing targets status"

        logger.info("Coverage reporting test completed!")


def run_coverage_analysis():
    """Run comprehensive coverage analysis."""
    analyzer = CoverageAnalyzer()

    # Run coverage analysis
    coverage_data = analyzer.run_coverage_analysis()

    # Generate reports
    html_report_path = analyzer.generate_html_report()
    json_report_path = analyzer.generate_json_report()

    # Print summary
    print("\n" + "=" * 60)
    print("COVERAGE ANALYSIS SUMMARY")
    print("=" * 60)

    print(f"Overall Coverage: {coverage_data['overall']['coverage_percent']:.1f}%")
    print(f"Target: {COVERAGE_TARGETS['overall']}%")
    print(
        f"Status: {'PASS' if coverage_data['overall']['coverage_percent'] >= COVERAGE_TARGETS['overall'] else 'FAIL'}"
    )

    print("\nCategory Coverage:")
    for category, data in coverage_data["by_category"].items():
        target = COVERAGE_TARGETS.get(category, 0)
        status = "PASS" if data["average_coverage"] >= target else "FAIL"
        print(
            f"  {category}: {data['average_coverage']:.1f}% (target: {target}%) - {status}"
        )

    print("\nReports generated:")
    print(f"  HTML: {html_report_path}")
    print(f"  JSON: {json_report_path}")

    return coverage_data


if __name__ == "__main__":
    # Run coverage analysis
    run_coverage_analysis()
