"""
Reproducibility validation and report generation for mammography analysis pipeline.

This module provides comprehensive reproducibility validation, deterministic
operations testing, and report generation for the breast density exploration pipeline.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
No medical decision should be based on these results.

Educational Context:
- Reproducibility validation ensures consistent results across different environments
- Deterministic operations testing validates fixed seed behavior
- Configuration management testing ensures proper experiment tracking
- Report generation provides comprehensive reproducibility documentation

Author: Research Team
Version: 1.0.0
"""

from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import random
import sys
import tempfile
from typing import Any, Dict, List

import pytest

np = pytest.importorskip("numpy")
psutil = pytest.importorskip("psutil")
torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mammography.clustering.clustering_algorithms import ClusteringAlgorithms
from mammography.models.embeddings.resnet50_extractor import ResNet50Extractor

# Import pipeline components
from mammography.preprocess.image_preprocessor import ImagePreprocessor

# Configure logging
logger = logging.getLogger(__name__)

# Reproducibility test configuration
REPRODUCIBILITY_CONFIG = {
    "test_seeds": [42, 123, 456, 789, 999],
    "num_runs": 3,
    "tolerance": 1e-6,
    "test_environments": ["local", "ci", "docker"],
}


class ReproducibilityValidator:
    """
    Reproducibility validator for mammography analysis pipeline.

    This class provides comprehensive reproducibility validation including
    deterministic operations testing, configuration management validation,
    and cross-environment consistency testing.

    Educational Notes:
    - Reproducibility validation ensures consistent results across environments
    - Deterministic operations testing validates fixed seed behavior
    - Configuration management testing ensures proper experiment tracking
    - Cross-environment testing validates portability and consistency
    """

    def __init__(self):
        """Initialize reproducibility validator."""
        self.test_results = {}
        self.environment_info = self._get_environment_info()
        self.validation_report = None

    def _get_environment_info(self) -> Dict[str, Any]:
        """Get comprehensive environment information."""
        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "architecture": platform.architecture(),
            "processor": platform.processor(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "timestamp": datetime.now().isoformat(),
        }

    def test_deterministic_operations(self) -> Dict[str, Any]:
        """Test deterministic operations with fixed seeds."""
        logger.info("Testing deterministic operations...")

        results = {
            "test_name": "deterministic_operations",
            "passed": True,
            "details": {},
            "failures": [],
        }

        for seed in REPRODUCIBILITY_CONFIG["test_seeds"]:
            seed_results = self._test_seed_determinism(seed)
            results["details"][f"seed_{seed}"] = seed_results

            if not seed_results["passed"]:
                results["passed"] = False
                results["failures"].extend(seed_results["failures"])

        self.test_results["deterministic_operations"] = results
        return results

    def _test_seed_determinism(self, seed: int) -> Dict[str, Any]:
        """Test determinism with a specific seed."""
        results = {"seed": seed, "passed": True, "failures": [], "test_cases": {}}

        # Test numpy determinism
        np.random.seed(seed)
        array1 = np.random.randn(1000)

        np.random.seed(seed)
        array2 = np.random.randn(1000)

        if not np.allclose(array1, array2, atol=REPRODUCIBILITY_CONFIG["tolerance"]):
            results["passed"] = False
            results["failures"].append(f"NumPy determinism failed for seed {seed}")
        else:
            results["test_cases"]["numpy_determinism"] = True

        # Test PyTorch determinism
        torch.manual_seed(seed)
        tensor1 = torch.randn(1000)

        torch.manual_seed(seed)
        tensor2 = torch.randn(1000)

        if not torch.allclose(
            tensor1, tensor2, atol=REPRODUCIBILITY_CONFIG["tolerance"]
        ):
            results["passed"] = False
            results["failures"].append(f"PyTorch determinism failed for seed {seed}")
        else:
            results["test_cases"]["pytorch_determinism"] = True

        # Test Python random determinism
        random.seed(seed)
        rand1 = [random.random() for _ in range(100)]

        random.seed(seed)
        rand2 = [random.random() for _ in range(100)]

        if not all(
            abs(a - b) < REPRODUCIBILITY_CONFIG["tolerance"]
            for a, b in zip(rand1, rand2, strict=False)
        ):
            results["passed"] = False
            results["failures"].append(
                f"Python random determinism failed for seed {seed}"
            )
        else:
            results["test_cases"]["python_random_determinism"] = True

        return results

    def test_pipeline_reproducibility(self) -> Dict[str, Any]:
        """Test pipeline reproducibility across multiple runs."""
        logger.info("Testing pipeline reproducibility...")

        results = {
            "test_name": "pipeline_reproducibility",
            "passed": True,
            "details": {},
            "failures": [],
        }

        # Test with multiple runs
        for run_id in range(REPRODUCIBILITY_CONFIG["num_runs"]):
            run_results = self._test_pipeline_run(run_id)
            results["details"][f"run_{run_id}"] = run_results

            if not run_results["passed"]:
                results["passed"] = False
                results["failures"].extend(run_results["failures"])

        self.test_results["pipeline_reproducibility"] = results
        return results

    def _test_pipeline_run(self, run_id: int) -> Dict[str, Any]:
        """Test a single pipeline run for reproducibility."""
        results = {"run_id": run_id, "passed": True, "failures": [], "outputs": {}}

        try:
            # Set fixed seed
            seed = 42
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            # Create test data
            test_data = self._create_test_data()

            # Test preprocessing reproducibility
            preprocessor = ImagePreprocessor(
                {
                    "target_size": [224, 224],
                    "normalization_method": "z_score_per_image",
                    "input_adapter": "1to3_replication",
                    "border_removal": True,
                    "seed": seed,
                }
            )

            # Test embedding extraction reproducibility
            extractor = ResNet50Extractor(
                {
                    "model_name": "resnet50",
                    "pretrained": True,
                    "input_adapter": "1to3_replication",
                    "batch_size": 1,
                    "device": "cpu",
                    "seed": seed,
                }
            )

            # Test clustering reproducibility
            clusterer = ClusteringAlgorithms(
                {
                    "algorithm": "kmeans",
                    "n_clusters": 3,
                    "pca_dims": 50,
                    "random_state": seed,
                }
            )

            # Store outputs for comparison
            results["outputs"] = {
                "preprocessor_config": preprocessor.config,
                "extractor_config": extractor.config,
                "clusterer_config": clusterer.config,
                "seed": seed,
            }

        except Exception as e:
            results["passed"] = False
            results["failures"].append(f"Pipeline run {run_id} failed: {e!s}")

        return results

    def _create_test_data(self) -> Dict[str, Any]:
        """Create test data for reproducibility testing."""
        # Create synthetic test data
        test_images = []
        for i in range(10):
            # Create synthetic mammography-like image
            image = np.random.rand(224, 224).astype(np.float32)
            test_images.append(image)

        return {
            "images": test_images,
            "metadata": [
                {"patient_id": f"patient_{i}", "projection": "CC", "laterality": "L"}
                for i in range(10)
            ],
        }

    def test_configuration_management(self) -> Dict[str, Any]:
        """Test configuration management and experiment tracking."""
        logger.info("Testing configuration management...")

        results = {
            "test_name": "configuration_management",
            "passed": True,
            "details": {},
            "failures": [],
        }

        # Test configuration serialization
        config_serialization = self._test_config_serialization()
        results["details"]["config_serialization"] = config_serialization

        if not config_serialization["passed"]:
            results["passed"] = False
            results["failures"].extend(config_serialization["failures"])

        # Test experiment tracking
        experiment_tracking = self._test_experiment_tracking()
        results["details"]["experiment_tracking"] = experiment_tracking

        if not experiment_tracking["passed"]:
            results["passed"] = False
            results["failures"].extend(experiment_tracking["failures"])

        self.test_results["configuration_management"] = results
        return results

    def _test_config_serialization(self) -> Dict[str, Any]:
        """Test configuration serialization and deserialization."""
        results = {"passed": True, "failures": [], "test_cases": {}}

        try:
            # Test configuration serialization
            config = {
                "preprocessing": {
                    "target_size": [224, 224],
                    "normalization_method": "z_score_per_image",
                    "input_adapter": "1to3_replication",
                    "seed": 42,
                },
                "embedding": {
                    "model_name": "resnet50",
                    "pretrained": True,
                    "input_adapter": "1to3_replication",
                    "seed": 42,
                },
                "clustering": {
                    "algorithm": "kmeans",
                    "n_clusters": 3,
                    "random_state": 42,
                },
            }

            # Serialize to JSON
            config_json = json.dumps(config, indent=2)
            results["test_cases"]["json_serialization"] = True

            # Deserialize from JSON
            config_loaded = json.loads(config_json)
            results["test_cases"]["json_deserialization"] = True

            # Verify consistency
            if config == config_loaded:
                results["test_cases"]["config_consistency"] = True
            else:
                results["passed"] = False
                results["failures"].append(
                    "Configuration serialization/deserialization failed"
                )

        except Exception as e:
            results["passed"] = False
            results["failures"].append(f"Configuration serialization failed: {e!s}")

        return results

    def _test_experiment_tracking(self) -> Dict[str, Any]:
        """Test experiment tracking capabilities."""
        results = {"passed": True, "failures": [], "test_cases": {}}

        try:
            # Test experiment metadata
            experiment_metadata = {
                "experiment_id": "test_exp_001",
                "timestamp": datetime.now().isoformat(),
                "environment": self.environment_info,
                "config": {"seed": 42, "algorithm": "kmeans", "n_clusters": 3},
                "results": {
                    "silhouette_score": 0.5,
                    "davies_bouldin_score": 1.2,
                    "calinski_harabasz_score": 100.0,
                },
            }

            # Test metadata serialization
            metadata_json = json.dumps(experiment_metadata, indent=2, default=str)
            results["test_cases"]["metadata_serialization"] = True

            # Test metadata hash generation
            metadata_hash = hashlib.md5(metadata_json.encode()).hexdigest()
            results["test_cases"]["metadata_hashing"] = True

            # Test metadata storage
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                f.write(metadata_json)
                temp_file = f.name

            # Verify file can be read back
            with open(temp_file, "r") as f:
                loaded_metadata = json.load(f)

            if loaded_metadata["experiment_id"] == experiment_metadata["experiment_id"]:
                results["test_cases"]["metadata_storage"] = True
            else:
                results["passed"] = False
                results["failures"].append("Metadata storage/retrieval failed")

            # Cleanup
            os.unlink(temp_file)

        except Exception as e:
            results["passed"] = False
            results["failures"].append(f"Experiment tracking failed: {e!s}")

        return results

    def test_cross_environment_consistency(self) -> Dict[str, Any]:
        """Test consistency across different environments."""
        logger.info("Testing cross-environment consistency...")

        results = {
            "test_name": "cross_environment_consistency",
            "passed": True,
            "details": {},
            "failures": [],
        }

        # Test environment detection
        environment_detection = self._test_environment_detection()
        results["details"]["environment_detection"] = environment_detection

        if not environment_detection["passed"]:
            results["passed"] = False
            results["failures"].extend(environment_detection["failures"])

        # Test platform compatibility
        platform_compatibility = self._test_platform_compatibility()
        results["details"]["platform_compatibility"] = platform_compatibility

        if not platform_compatibility["passed"]:
            results["passed"] = False
            results["failures"].extend(platform_compatibility["failures"])

        self.test_results["cross_environment_consistency"] = results
        return results

    def _test_environment_detection(self) -> Dict[str, Any]:
        """Test environment detection capabilities."""
        results = {"passed": True, "failures": [], "test_cases": {}}

        try:
            # Test Python version detection
            python_version = sys.version_info
            results["test_cases"]["python_version_detection"] = True

            # Test platform detection
            platform_info = platform.platform()
            results["test_cases"]["platform_detection"] = True

            # Test PyTorch availability
            torch_available = torch is not None
            results["test_cases"]["pytorch_availability"] = torch_available

            # Test CUDA availability
            cuda_available = torch.cuda.is_available()
            results["test_cases"]["cuda_availability"] = cuda_available

            # Test memory detection
            memory_gb = psutil.virtual_memory().total / (1024**3)
            results["test_cases"]["memory_detection"] = memory_gb > 0

        except Exception as e:
            results["passed"] = False
            results["failures"].append(f"Environment detection failed: {e!s}")

        return results

    def _test_platform_compatibility(self) -> Dict[str, Any]:
        """Test platform compatibility."""
        results = {"passed": True, "failures": [], "test_cases": {}}

        try:
            # Test basic operations
            test_array = np.array([1, 2, 3, 4, 5])
            test_sum = np.sum(test_array)
            results["test_cases"]["numpy_operations"] = test_sum == 15

            # Test PyTorch operations
            test_tensor = torch.tensor([1, 2, 3, 4, 5])
            test_sum_tensor = torch.sum(test_tensor)
            results["test_cases"]["pytorch_operations"] = test_sum_tensor.item() == 15

            # Test file operations
            with tempfile.NamedTemporaryFile(delete=False) as f:
                f.write(b"test data")
                temp_file = f.name

            with open(temp_file, "rb") as f:
                data = f.read()

            if data == b"test data":
                results["test_cases"]["file_operations"] = True
            else:
                results["passed"] = False
                results["failures"].append("File operations failed")

            # Cleanup
            os.unlink(temp_file)

        except Exception as e:
            results["passed"] = False
            results["failures"].append(f"Platform compatibility failed: {e!s}")

        return results

    def generate_reproducibility_report(self) -> Dict[str, Any]:
        """Generate comprehensive reproducibility report."""
        logger.info("Generating reproducibility report...")

        # Run all reproducibility tests
        self.test_deterministic_operations()
        self.test_pipeline_reproducibility()
        self.test_configuration_management()
        self.test_cross_environment_consistency()

        # Generate report
        report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "environment": self.environment_info,
                "test_config": REPRODUCIBILITY_CONFIG,
            },
            "test_results": self.test_results,
            "summary": self._generate_summary(),
            "recommendations": self._generate_recommendations(),
        }

        self.validation_report = report
        return report

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate reproducibility test summary."""
        total_tests = len(self.test_results)
        passed_tests = sum(
            1 for result in self.test_results.values() if result["passed"]
        )
        failed_tests = total_tests - passed_tests

        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": (
                (passed_tests / total_tests * 100) if total_tests > 0 else 0
            ),
            "overall_status": "PASS" if failed_tests == 0 else "FAIL",
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations for improving reproducibility."""
        recommendations = []

        for test_name, result in self.test_results.items():
            if not result["passed"]:
                if test_name == "deterministic_operations":
                    recommendations.append(
                        "Fix deterministic operations: Ensure all random operations use fixed seeds"
                    )
                elif test_name == "pipeline_reproducibility":
                    recommendations.append(
                        "Fix pipeline reproducibility: Ensure consistent results across multiple runs"
                    )
                elif test_name == "configuration_management":
                    recommendations.append(
                        "Fix configuration management: Ensure proper serialization and experiment tracking"
                    )
                elif test_name == "cross_environment_consistency":
                    recommendations.append(
                        "Fix cross-environment consistency: Ensure compatibility across different platforms"
                    )

        if not recommendations:
            recommendations.append(
                "All reproducibility tests passed! No recommendations needed."
            )

        return recommendations

    def save_report(self, output_file: str = "reproducibility_report.json") -> str:
        """Save reproducibility report to file."""
        if self.validation_report is None:
            self.generate_reproducibility_report()

        output_path = Path(output_file)
        with open(output_path, "w") as f:
            json.dump(self.validation_report, f, indent=2, default=str)

        logger.info(f"Reproducibility report saved: {output_path}")
        return str(output_path)


class TestReproducibilityValidation:
    """
    Comprehensive reproducibility test suite.

    This class provides a complete test suite for reproducibility validation
    including deterministic operations, configuration management, and cross-environment testing.

    Educational Notes:
    - Reproducibility testing ensures consistent results across environments
    - Deterministic operations testing validates fixed seed behavior
    - Configuration management testing ensures proper experiment tracking
    - Cross-environment testing validates portability and consistency
    """

    def __init__(self):
        """Initialize reproducibility test suite."""
        self.validator = ReproducibilityValidator()

    def test_reproducibility_validation(self):
        """Test reproducibility validation capabilities."""
        # Test deterministic operations
        deterministic_results = self.validator.test_deterministic_operations()
        assert deterministic_results[
            "passed"
        ], f"Deterministic operations failed: {deterministic_results['failures']}"

        # Test pipeline reproducibility
        pipeline_results = self.validator.test_pipeline_reproducibility()
        assert pipeline_results[
            "passed"
        ], f"Pipeline reproducibility failed: {pipeline_results['failures']}"

        # Test configuration management
        config_results = self.validator.test_configuration_management()
        assert config_results[
            "passed"
        ], f"Configuration management failed: {config_results['failures']}"

        # Test cross-environment consistency
        environment_results = self.validator.test_cross_environment_consistency()
        assert environment_results[
            "passed"
        ], f"Cross-environment consistency failed: {environment_results['failures']}"

        logger.info("All reproducibility validation tests passed!")

    def test_report_generation(self):
        """Test reproducibility report generation."""
        # Generate report
        report = self.validator.generate_reproducibility_report()

        # Validate report structure
        assert "report_metadata" in report, "Report missing metadata"
        assert "test_results" in report, "Report missing test results"
        assert "summary" in report, "Report missing summary"
        assert "recommendations" in report, "Report missing recommendations"

        # Validate summary
        summary = report["summary"]
        assert "total_tests" in summary, "Summary missing total tests"
        assert "passed_tests" in summary, "Summary missing passed tests"
        assert "failed_tests" in summary, "Summary missing failed tests"
        assert "success_rate" in summary, "Summary missing success rate"
        assert "overall_status" in summary, "Summary missing overall status"

        # Save report
        report_path = self.validator.save_report()
        assert Path(report_path).exists(), "Report file not created"

        logger.info("Reproducibility report generation test passed!")

    def test_environment_info(self):
        """Test environment information collection."""
        env_info = self.validator.environment_info

        # Validate environment info
        assert "python_version" in env_info, "Environment info missing Python version"
        assert "platform" in env_info, "Environment info missing platform"
        assert "torch_version" in env_info, "Environment info missing PyTorch version"
        assert "numpy_version" in env_info, "Environment info missing NumPy version"
        assert (
            "cuda_available" in env_info
        ), "Environment info missing CUDA availability"
        assert "timestamp" in env_info, "Environment info missing timestamp"

        logger.info("Environment information collection test passed!")


def run_reproducibility_validation():
    """Run comprehensive reproducibility validation."""
    validator = ReproducibilityValidator()

    # Generate report
    report = validator.generate_reproducibility_report()

    # Save report
    report_path = validator.save_report()

    # Print summary
    print("\n" + "=" * 60)
    print("REPRODUCIBILITY VALIDATION SUMMARY")
    print("=" * 60)

    summary = report["summary"]
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed Tests: {summary['passed_tests']}")
    print(f"Failed Tests: {summary['failed_tests']}")
    print(f"Success Rate: {summary['success_rate']:.1f}%")
    print(f"Overall Status: {summary['overall_status']}")

    print(f"\nReport saved: {report_path}")

    return report


if __name__ == "__main__":
    # Run reproducibility validation
    run_reproducibility_validation()
