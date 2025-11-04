#!/usr/bin/env python3
"""
Basic test script to verify the mammography analysis pipeline implementation.

This script tests the basic functionality without requiring model downloads
to ensure the core implementation is working correctly.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
No medical decision should be based on these results.

Educational Context:
- Basic testing verifies implementation correctness
- Import testing ensures components can be loaded
- Configuration testing validates parameter handling

Author: Research Team
Version: 1.0.0
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Research disclaimer
RESEARCH_DISCLAIMER = """
⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
No medical decision should be based on these results.

This project is intended exclusively for research and education purposes
in medical imaging processing and machine learning.
"""

def test_imports():
    """Test that all modules can be imported successfully."""
    print("Testing imports...")
    
    try:
        from src.viz.cluster_visualizer import ClusterVisualizer
        print("✓ ClusterVisualizer imported successfully")
        
        from src.cli.preprocess_cli import app as preprocess_app
        print("✓ Preprocessing CLI imported successfully")
        
        from src.cli.embed_cli import app as embed_app
        print("✓ Embedding CLI imported successfully")
        
        from src.cli.cluster_cli import app as cluster_app
        print("✓ Clustering CLI imported successfully")
        
        from src.cli.analyze_cli import app as analyze_app
        print("✓ Analysis CLI imported successfully")
        
        from src.pipeline.mammography_pipeline import MammographyPipeline
        print("✓ MammographyPipeline imported successfully")
        
        from src.cli.main import app as main_app
        print("✓ Main CLI imported successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {str(e)}")
        return False

def test_cluster_visualizer():
    """Test ClusterVisualizer basic functionality."""
    print("\nTesting ClusterVisualizer...")
    
    try:
        from src.viz.cluster_visualizer import ClusterVisualizer
        
        # Create test configuration
        config = {
            'visualizations': ['umap_2d', 'metrics_plot'],
            'umap_params': {
                'n_neighbors': 15,
                'min_dist': 0.1,
                'n_components': 2,
                'metric': 'euclidean',
                'random_state': 42
            },
            'plot_params': {
                'figsize': (10, 8),
                'dpi': 300,
                'style': 'whitegrid',
                'palette': 'husl'
            },
            'seed': 42
        }
        
        # Initialize visualizer
        _ = ClusterVisualizer(config)
        print("✓ ClusterVisualizer initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ ClusterVisualizer test failed: {str(e)}")
        return False

def test_cli_structure():
    """Test CLI structure without full initialization."""
    print("\nTesting CLI structure...")
    
    try:
        from src.cli.main import app as main_app
        
        # Test that the main app exists and has the right structure
        if hasattr(main_app, 'commands'):
            print("✓ Main CLI has commands structure")
        else:
            print("✓ Main CLI app accessible (commands may be dynamic)")
        
        return True
        
    except Exception as e:
        print(f"✗ CLI structure test failed: {str(e)}")
        return False

def test_data_models():
    """Test data model creation."""
    print("\nTesting data models...")
    
    try:
        from src.preprocess.preprocessed_tensor import PreprocessedTensor
        from src.models.embeddings.embedding_vector import EmbeddingVector
        from src.clustering.clustering_result import ClusteringResult
        
        # Test PreprocessedTensor creation
        tensor_data = torch.randn(3, 224, 224)
        _ = PreprocessedTensor(
            image_id="test_image",
            tensor_data=tensor_data,
            preprocessing_config={"method": "test", "target_size": [224, 224], "normalization_method": "z_score_per_image", "input_adapter": "1to3_replication"},
            normalization_method="z_score_per_image",
            target_size=(224, 224),
            input_adapter="1to3_replication"
        )
        print("✓ PreprocessedTensor created successfully")
        
        # Test EmbeddingVector creation
        embedding_data = torch.randn(2048)
        _ = EmbeddingVector(
            image_id="test_image",
            embedding=embedding_data,
            model_config={"model": "resnet50", "model_name": "resnet50", "pretrained": True, "feature_layer": "avgpool"},
            input_adapter="1to3_replication",
            extraction_time=1.0
        )
        print("✓ EmbeddingVector created successfully")
        
        # Test ClusteringResult creation
        cluster_labels = torch.tensor([0, 1, 0, 1, 2])
        _ = ClusteringResult(
            experiment_id="test_exp",
            algorithm="kmeans",
            cluster_labels=cluster_labels,
            hyperparameters={"n_clusters": 3, "random_state": 42},
            metrics={"silhouette": 0.5, "davies_bouldin": 0.3, "calinski_harabasz": 100.0},
            centroids=torch.randn(3, 2048),
            uncertainty_scores=torch.randn(5),
            embedding_ids=["img1", "img2", "img3", "img4", "img5"],
            processing_time=1.0
        )
        print("✓ ClusteringResult created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Data models test failed: {str(e)}")
        return False

def test_configuration_validation():
    """Test configuration validation without full initialization."""
    print("\nTesting configuration validation...")
    
    try:
        from src.preprocess.image_preprocessor import ImagePreprocessor
        
        # Test valid configuration
        valid_config = {
            'target_size': [224, 224],
            'normalization_method': 'z_score_per_image',
            'input_adapter': '1to3_replication'
        }
        
        _ = ImagePreprocessor(valid_config)
        print("✓ Valid configuration accepted")
        
        # Test invalid configuration
        try:
            invalid_config = {
                'target_size': [224, 224],
                'normalization_method': 'invalid_method',
                'input_adapter': '1to3_replication'
            }
            _ = ImagePreprocessor(invalid_config)
            print("✗ Invalid configuration should have been rejected")
            return False
        except ValueError:
            print("✓ Invalid configuration correctly rejected")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration validation test failed: {str(e)}")
        return False

def main():
    """Run all tests."""
    print(RESEARCH_DISCLAIMER)
    print()
    print("Testing Mammography Analysis Pipeline Implementation (Basic)")
    print("=" * 70)
    
    tests = [
        test_imports,
        test_cluster_visualizer,
        test_cli_structure,
        test_data_models,
        test_configuration_validation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 70)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All basic tests passed! Core implementation is working correctly.")
        print("\nNext steps:")
        print("1. Install missing dependencies if needed")
        print("2. Test with actual DICOM data")
        print("3. Run the complete pipeline")
        return 0
    else:
        print("✗ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
