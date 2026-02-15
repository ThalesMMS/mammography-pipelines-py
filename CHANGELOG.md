## [1.0.0] - 2026-02-04

### Added

- Smart wizard with contextual help and progress indicators for training workflows
- Dataset format auto-detection with support for CSV/TSV, DICOM, and image formats (PNG/JPG)
- Web-based UI dashboard with Streamlit including inference, experiments, training config, explainability, and hyperparameter tuning pages
- Explainable AI visualizations with GradCAM for CNN models and attention map support for Vision Transformers
- Experiment tracking dashboard with LocalTracker SQLite backend and multi-format export capabilities
- View-specific model training with ensemble evaluation and view comparison visualization
- Flexible train/validation/test split modes including three-way splits, k-fold cross-validation, and pre-defined CSV splits
- Automated hyperparameter tuning with Optuna integration and YAML-based search space configuration
- Parallel processing support for preprocessing and embedding pipeline steps (3-4x performance improvement)
- Efficient DICOM loading with LRU caching and lazy loading support for reduced memory footprint
- FP16 mixed-precision support to ResNet50Extractor
- Vision Transformer (ViT) model support
- EfficientNet embedding extraction
- Docker support and reproducibility enhancements
- Comprehensive API documentation with Sphinx configuration and docstring coverage enforcement (90% threshold)
- CLI wizard with PCA SVD solver selection
- Python 3.14 support

### Changed

- Consolidated DICOM loading and windowing utilities into io/dicom.py module, eliminating duplication across apps
- Refactored CLI to use internal command modules with mammography app consolidation
- Refactored EDA pipeline into modular architecture
- Updated experiment tracking to include per-epoch comprehensive metric logging and automatic hyperparameter logging
- Enhanced training pipeline with robust error handling and OOM safeguards
- Migrated pipeline documentation and removed legacy ResNet50_Test
- Updated training script main loop for view-specific training support
- Improved dataset validation with format detection and preprocessing suggestions

### Fixed

- Corrected explain command function signatures
- Fixed view column default from ImageLaterality to view
- Used StratifiedGroupKFold for seed-dependent k-fold splits ensuring proper stratification
- Eliminated double DICOM file reading in validation by passing dataset parameter through call chain
- Lazy loading validation compatibility

### Documentation

- Created comprehensive API reference documentation for core and supporting modules
- Added basic usage example notebook and custom model extension examples
- Established documentation build and docstring coverage workflows
- Updated function docstrings to document dataset parameter usage
- Updated CLI help text and documentation for dataset format auto-detection
- Updated documentation links to use explicit file references
- Added QUICKSTART guide and updated README for unified CLI

## 1.0.0 - Comprehensive ML Pipeline Modernization

### New Features

- Experiment tracking dashboard with SQLite backend for logging metrics, hyperparameters, and per-epoch data
- Automated hyperparameter tuning via Optuna with YAML-based search space configuration
- Web-based UI dashboard with Streamlit supporting inference, explainability, training configuration, and experiment monitoring
- Dataset format auto-detection with support for CSV/TSV, image (DICOM/PNG/JPG), and automatic schema inference
- View-specific model training for handling different medical imaging perspectives with ensemble prediction
- Flexible train/validation/test splits supporting three-way splits, k-fold cross-validation, and CSV-based split loading
- Explainable AI visualizations with GradCAM support for CNN models and attention map visualization for Vision Transformers
- Vision Transformer (ViT) model support alongside existing ResNet50 and EfficientNet architectures
- CLI export functionality for figures and experiment data in multiple formats
- FP16 mixed-precision support to ResNet50Extractor for improved training efficiency
- Lazy loading and LRU caching for DICOM files with disk persistence to reduce memory footprint
- Parallel processing pipeline steps with configurable worker counts for 3-4x performance improvement
- Automatic image normalization and validation utilities

### Improvements

- Consolidated DICOM loading and windowing functions to eliminate code duplication across multiple apps
- Refactored CLI to use internal command modules for better maintainability
- Enhanced training pipeline with robust error handling and OOM safeguards
- Optimized numpy-torch conversions in clustering and preprocessing modules
- Improved reproducibility with Docker support and unified configuration management
- Modular EDA pipeline architecture with improved structure and reusability
- Updated documentation with explicit file references and comprehensive usage examples
- End-to-end workflow verification and smoke test validation across features
- Configuration documentation and example configs demonstrating new capabilities

### Bug Fixes

- Corrected view column default from ImageLaterality to view in view-specific training
- Fixed explain command function signatures for proper CLI integration
- Resolved lazy loading validation compatibility issues
- Addressed environmental constraints in parallel processing QA verification
- Fixed StratifiedGroupKFold usage for seed-dependent k-fold splits

### Documentation

- Added subtask completion summaries and integration guides
- Updated .claude_settings.json and CLAUDE.md documentation files
- Documented performance improvements and lazy loading usage examples
- Added docstrings to split functions and updated function documentation for dataset parameters

---

## What's Changed

- feat: Add experiment tracking dashboard with SQLite backend by @Thales Matheus in c9714a3
- feat: Add comprehensive E2E reports, tests and utilities by @Thales Matheus in cef5bfb
- docs: Add subtask 5-1 completion summary by @Thales Matheus in aebe9bd
- feat: End-to-end workflow verification by @Thales Matheus in ff0beb7
- feat: Integrate Optuna study storage with LocalTracker by @Thales Matheus in 6f66b49
- feat: Create hyperparameter tuning Streamlit page by @Thales Matheus in 488d63f
- feat: Add CLI flag for automatic publication export during training by @Thales Matheus in 9f677bc
- feat: Add export buttons to Streamlit experiments page by @Thales Matheus in 745d06b
- feat: Create export module with multi-format figure export by @Thales Matheus in 71364dc
- feat: Add automatic hyperparameter logging by @Thales Matheus in ef5b94d
- feat: Add per-epoch comprehensive metric logging by @Thales Matheus in 4dca783
- feat: Integrate LocalTracker into train.py ExperimentTracker by @Thales Matheus in 789c18c
- feat: Create LocalTracker class with SQLite backend by @Thales Matheus in ac9985e
- feat: Update documentation and CLI help text by @Thales Matheus in 1409782
- feat: Create integration tests with real dataset patterns by @Thales Matheus in 6ac7005
- feat: Create unit tests for format detection by @Thales Matheus in 3f1eb51
- feat: Display warnings and suggestions in CLI by @Thales Matheus in b89b9d9
- feat: Add preprocessing suggestions based on format by @Thales Matheus in d204edf
- feat: Implement format validation warnings by @Thales Matheus in b965520
- feat: Update train CLI to support auto-detection by @Thales Matheus in f429df7
- feat: Update extract_features CLI to support auto-detect by @Thales Matheus in 1d40047
- feat: Integrate auto-detection into load_dataset_dataframe by @Thales Matheus in 8b917c0
- feat: Add image format detection (DICOM vs PNG/JPG) by @Thales Matheus in 35f30d5
- feat: Add CSV/TSV format detection and schema inference by @Thales Matheus in 31578b5
- feat: Create format detection module with directory structure analysis by @Thales Matheus in 868f07e
- feat: Add error handling and UI polish to web dashboard by @Thales Matheus in 9971d0a
- feat: Add shared state management across pages by @Thales Matheus in 6b39f5b
- feat: Add training job launch and progress monitoring by @Thales Matheus in 8b6c646
- feat: Create training config page with parameter inputs by @Thales Matheus in 72f1a2e
- feat: Add experiment comparison and metrics visualization by @Thales Matheus in e8f6d87
- feat: Create experiments page with MLflow integration by @Thales Matheus in d76a880
- feat: Add explainability options (target layer, colormap, alpha) by @Thales Matheus in 2d52f21
- feat: Create explainability page with GradCAM generation by @Thales Matheus in 6e507c5
- feat: Implement image upload, preprocessing, and inference by @Thales Matheus in 0739b94
- feat: Implement model selection and checkpoint loading in inference page by @Thales Matheus in d3baa0f
- feat: Create pages directory and inference page structure by @Thales Matheus in bb098c2
- feat: Create CLI web command that launches the Streamlit by @Thales Matheus in 68c408e
- feat: Create main Streamlit app landing page with navigation by @Thales Matheus in af7b8bf
- feat: Create web_ui app directory structure by @Thales Matheus in 90ac4d4
- feat: Add outputs/tests/docs and update core modules by @Thales Matheus in 3993f51
- feat: Expand article content and add validation tools by @Thales Matheus in ee64988
- fix: Correct explain command function signatures by @Thales Matheus in d38a3cd
- feat: Create integration test for explain CLI command by @Thales Matheus in 2450c0a
- feat: Create unit tests for explainability module by @Thales Matheus in f0da6bb
- feat: Update report_pack to include explanation grids in LaTeX output by @Thales Matheus in 30bbf73
- feat: Add attention map support to report_pack gradcam utilities by @Thales Matheus in beb45fd
- feat: Register explain command in CLI entry point by @Thales Matheus in d387056
- feat: Create explain command module by @Thales Matheus in cdea5c5
- feat: Update vis/__init__.py to export explainability functions by @Thales Matheus in 437f332
- feat: Add batch processing and export utilities to explainability module by @Thales Matheus in 96d776f
- feat: Add ViT attention map visualization to explainability module by @Thales Matheus in 8cc2847
- feat: Create explainability module with GradCAM class for CNN models by @Thales Matheus in 76e7516
- feat: Add Vision Transformer (ViT) model support and tests by @Thales Matheus in 6644641
- fix: Correct view column default from ImageLaterality to view by @Thales Matheus in b9309c4
- feat: End-to-end smoke test validation by @Thales Matheus in da954f4
- feat: Update documentation and example config by @Thales Matheus in 42c936f
- feat: Create integration test for view-specific training by @Thales Matheus in 835e10e
- feat: Create view comparison visualization by @Thales Matheus in 0ea5f37
- feat: Update metrics saving to include view identifier by @Thales Matheus in a2bc4e1
- feat: Add ensemble evaluation to training script by @Thales Matheus in a81c866
- feat: Create ensemble predictor class by @Thales Matheus in abca1bf
- feat: Update checkpoint saving for view-specific models by @Thales Matheus in c0eafe2
- feat: Modify train.py main loop for view-specific training by @Thales Matheus in af30473
- feat: Create view-specific model wrapper class by @Thales Matheus in 963808f
- feat: Add view filtering function to splits module by @Thales Matheus in 783579f
- feat: Add view extraction to CSV loader by @Thales Matheus in e62b17c
- feat: Add view-specific CLI arguments to train.py by @Thales Matheus in 17cb7e3
- feat: Add view-specific training config fields to TrainConfig by @Thales Matheus in c495d0e
- fix: Use StratifiedGroupKFold for seed-dependent k-fold splits by @Thales Matheus in 9363fb2
- feat: Create example config demonstrating new split modes by @Thales Matheus in 1400aee
- feat: Add docstrings to new split functions by @Thales Matheus in 884c1e1
- feat: Update existing tests to ensure backward compatibility by @Thales Matheus in f6121f4
- feat: Create integration test for end-to-end split workflow by @Thales Matheus in 85e59cc
- feat: Create unit tests for create_kfold_splits() by @Thales Matheus in 16ed151
- feat: Create unit tests for load_splits_from_csvs() and validate_split_overlap() by @Thales Matheus in 2a488a1
- feat: Create unit tests for create_three_way_split() by @Thales Matheus in 99cb21e
- feat: Add validate_split_overlap() function by @Thales Matheus in a386b10
- feat: Add load_multiple_csvs() function to csv_loader.py by @Thales Matheus in 442e50d
- feat: Update train.py main logic to use new split functions by @Thales Matheus in 16f0db3
- feat: Add CLI arguments to train.py for split configuration by @Thales Matheus in cdb81b0
- feat: Add split configuration section to density.yaml by @Thales Matheus in c848824
- feat: Add SplitConfig dataclass to encapsulate split configuration by @Thales Matheus in 4112f2b
- feat: Add create_kfold_splits() function for k-fold cross-validation by @Thales Matheus in c1414e8
- feat: Add load_splits_from_csvs() function to load pre-defined splits by @Thales Matheus in 74f804d
- feat: Add create_three_way_split() function for train/val/test splitting by @Thales Matheus in f574625
- feat: Add FP16 mixed-precision support to ResNet50Extractor by @Thales Matheus in c0f66ea
- fix: QA verification with environmental constraint by @Thales Matheus in 9599b52
- feat: Update pipeline configuration documentation by @Thales Matheus in 4f2da14
- feat: Run full test suite to ensure no regressions by @Thales Matheus in 9afde92
- feat: Run performance benchmark comparing sequential vs parallel by @Thales Matheus in 6afbdb0
- feat: Add integration test for parallel pipeline processing by @Thales Matheus in c318489
- feat: Refactor _run_embedding_step to use parallel processing by @Thales Matheus in 9b51379
- feat: Refactor _run_preprocessing_step to use parallel processing by @Thales Matheus in adb37fb
- feat: Add max_workers parameter to MammographyPipeline configuration by @Thales Matheus in 1ba9888
- feat: Optimize numpy-torch conversions in clustering and preprocessing by @Thales Matheus in 2c116f6
- fix: Lazy loading validation compatibility by @Thales Matheus in b7110dc
- feat: Document performance improvements and usage examples by @Thales Matheus in 6535d51
- feat: Create performance benchmarks for memory usage and loading time by @Thales Matheus in 756a5b4
- feat: Add integration tests for DicomReader with lazy loading by @Thales Matheus in 0863772
- feat: Update __init__.py exports for new modules by @Thales Matheus in acef851
- feat: Add lazy_load parameter to DicomReader class by @Thales Matheus in fda40b4
- feat: Add unit tests for LRU cache behavior by @Thales Matheus in 55737f2
- feat: Add disk persistence support for cache by @Thales Matheus in 7b5e676
- feat: Create DicomLRUCache class with configurable max_size by @Thales Matheus in 5b6c611
- feat: Add unit tests for lazy loading behavior by @Thales Matheus in 8482860
- feat: Create LazyDicomDataset class with deferred pixel_array loading by @Thales Matheus in 06dc3b2
- feat: Run full test suite to ensure no regressions by @Thales Matheus in 39a7326
- feat: Update function docstrings to document dataset parameters by @Thales Matheus in bfb8d92
- feat: Verify backward compatibility with legacy code by @Thales Matheus in c1d7f21
- feat: Run unit tests for DICOM validation by @Thales Matheus in 64e258a
- feat: Update DicomReader.read_dicom_file to pass dataset parameter by @Thales Matheus in a845283
- feat: Update create_mammography_image_from_dicom to accept optional dataset parameter by @Thales Matheus in 17ed1f2
- feat: Update validate_dicom_file to use optional dataset by @Thales Matheus in 13ee723
- feat: Add optional dataset parameter to MammographyImage by @Thales Matheus in ca4ca18
- feat: Add automatic image normalization and validation utilities by @Thales Matheus in 3915398
- feat: Add consolidation plan for mammography-pipelines by @Thales Matheus in a48b353
- feat: Run full test suite to verify no regressions by @Thales Matheus in 4a31469
- feat: Remove duplicated apply_windowing from app dicom_loader by @Thales Matheus in 0832318
- feat: Verify patch_marking app still works by @Thales Matheus in 4f0be89
- feat: Update patch_marking/dicom_loader.py to import from io/dicom by @Thales Matheus in 4fba8d6
- feat: Verify density_classifier app still works by @Thales Matheus in 0b6934e
- feat: Update density_classifier/dicom_loader.py to import from io/dicom by @Thales Matheus in 5912c89
- feat: Add extract_window_parameters helper function to io/dicom.py by @Thales Matheus in 7309dad
- feat: Add apply_windowing function to io/dicom.py by @Thales Matheus in 318fb14
- feat: Refactor EDA pipeline into modular architecture by @Thales Matheus in ab494ac
- fix: Address QA issues by @Thales Matheus in 87c249b
- feat: Add --dry-run flag for tune integration test by @Thales Matheus in 908b784
- feat: Register tune subcommand in cli.py by @Thales Matheus in 11f302a
- feat: Create tune.py command module by @Thales Matheus in 315aaca
- feat: Create optuna_tuner.py with trial objective wrapper by @Thales Matheus in 84ac9d6
- feat: Create tune.yaml config with default search space by @Thales Matheus in 611802c
- feat: Create search_space.py with YAML loader and validation by @Thales Matheus in c5cebc9
- feat: Add EfficientNet embedding extraction and integration tests by @Thales Matheus in 68d044e
- feat: Create tuning package structure by @Thales Matheus in 8d7c486
- feat: Add Optuna to pyproject.toml dependencies by @Thales Matheus in c8964ab
- feat: Add robust training, error handling, and integration test by @Thales Matheus in 07d9ad7
- feat: Add dataset failure and OOM safeguards to training by @Thales Matheus in 02ca819
- feat: Add Docker support, improve reproducibility, and enhance training pipeline by @Thales Matheus in 08c0c7c
- feat: Add Streamlit UIs and refactor CLI to in-process launch by @Thales Matheus in cf77ec6
- docs: Update documentation links to use explicit file references by @Thales Matheus in 68677f7
- docs: Add initial documentation for mammography pipelines by @Thales Matheus in ec189e5
- feat: Refactor CLI to use internal command modules by @Thales Matheus in 0144331
- feat: Improve test data generation and PCA usage in tests by @Thales Matheus in ff2cfa7
- feat: Refactor CLI: integrate data-audit, remove tools wrappers by @Thales Matheus in 0cf46c0
- feat: Add PCA SVD solver selection to wizard CLI by @Thales Matheus in ebc85f5
- refactor: Remove legacy outputs and code, update core applications by @Thales Matheus in b166b0f
- feat: Migrate pipeline and docs, remove legacy ResNet50_Test by @Thales Matheus in 023f686
- feat: Remove legacy unified_cli and update related code by @Thales Matheus in 42951e7
- feat: Refactor CLI, config, and wizard; add embedding support by @Thales Matheus in c5c9825
- feat: Remove legacy RL and wrapper scripts; add augment/inference by @Thales Matheus in a18d177
- feat: Add Python 3.14 support to requirements files by @Thales Matheus in e3af5dc
- feat: Archive legacy EDA script and add compatibility shim by @Thales Matheus in 2a8c5d3
- feat: Refactor CLI to consolidated mammography entrypoint by @Thales Matheus in f83d9f8
- feat: Add RL refinement compatibility shims and update paths by @Thales Matheus in 656b9d7
- feat: Archive legacy ResNet50 and unified CLI projects by @Thales Matheus in a19f357

## Thanks to all contributors

@Thales Matheus