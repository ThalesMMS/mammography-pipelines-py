"""
Clustering CLI for mammography embedding analysis.

This module provides command-line interface for clustering mammography
embeddings using various algorithms including K-means, GMM, and HDBSCAN
for the breast density exploration pipeline.

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
No medical decision should be based on these results.

Educational Context:
- CLI provides user-friendly interface for clustering operations
- Multiple algorithms enable comparison and robustness assessment
- Evaluation metrics assess clustering quality without ground truth
- Ablation studies help understand algorithm performance

Author: Research Team
Version: 1.0.0
"""

from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
import yaml

from ..clustering.clustering_algorithms import ClusteringAlgorithms
from ..clustering.clustering_result import ClusteringResult
from ..models.embeddings.embedding_vector import EmbeddingVector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Typer app
app = typer.Typer(
    name="mmg-cluster",
    help="Cluster mammography embeddings using various algorithms",
    add_completion=False,
)

# Research disclaimer
RESEARCH_DISCLAIMER = """
⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
No medical decision should be based on these results.
"""


@app.command()
def cluster(
    input_dir: Path = typer.Option(
        ..., "--input-dir", "-i", help="Directory containing embedding vectors"
    ),
    output_dir: Path = typer.Option(
        ..., "--output-dir", "-o", help="Directory to save clustering results"
    ),
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to clustering configuration file"
    ),
    algorithm: str = typer.Option(
        "kmeans",
        "--algorithm",
        "-a",
        help="Clustering algorithm (kmeans, gmm, hdbscan, agglomerative)",
    ),
    n_clusters: int = typer.Option(
        4,
        "--n-clusters",
        "-k",
        help="Number of clusters (for kmeans, gmm, agglomerative)",
    ),
    pca_dimensions: int = typer.Option(
        50,
        "--pca-dims",
        "-p",
        help="Number of PCA dimensions for dimensionality reduction",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
):
    """
    Cluster mammography embeddings using specified algorithm.

    This command applies clustering algorithms to mammography embeddings
    to discover patterns and group similar images together.

    Educational Note: Clustering discovers patterns in high-dimensional
    embeddings without requiring labeled training data.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Print research disclaimer
    typer.echo(RESEARCH_DISCLAIMER)
    typer.echo()

    try:
        # Load configuration
        config = _load_config(config_file, algorithm, n_clusters, pca_dimensions)

        # Validate input directory
        if not input_dir.exists():
            typer.echo(f"Error: Input directory {input_dir} does not exist", err=True)
            raise typer.Exit(1)

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize clustering algorithms
        clusterer = ClusteringAlgorithms(config)

        typer.echo(f"Starting clustering with {algorithm} algorithm")
        typer.echo(f"Configuration: {config}")

        # Load embeddings
        embedding_vectors = _load_embeddings(input_dir)
        if not embedding_vectors:
            typer.echo("No embedding vectors found!", err=True)
            raise typer.Exit(1)

        typer.echo(f"Loaded {len(embedding_vectors)} embedding vectors")

        # Perform clustering
        clustering_result = clusterer.cluster_embeddings(embedding_vectors)
        if clustering_result is None:
            typer.echo("Clustering failed!", err=True)
            raise typer.Exit(1)

        # Save results
        _save_clustering_results(clustering_result, output_dir)

        # Print results summary
        _print_clustering_summary(clustering_result)

        typer.echo("Clustering completed successfully!")
        typer.echo(f"Results saved to {output_dir}")

    except Exception as e:
        logger.error(f"Error in clustering: {e!s}")
        typer.echo(f"Error: {e!s}", err=True)
        raise typer.Exit(1)


@app.command()
def compare(
    input_dir: Path = typer.Option(
        ..., "--input-dir", "-i", help="Directory containing embedding vectors"
    ),
    output_dir: Path = typer.Option(
        ..., "--output-dir", "-o", help="Directory to save comparison results"
    ),
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to clustering configuration file"
    ),
    algorithms: str = typer.Option(
        "kmeans,gmm,hdbscan",
        "--algorithms",
        "-a",
        help="Comma-separated list of algorithms to compare",
    ),
    n_clusters: int = typer.Option(
        4,
        "--n-clusters",
        "-k",
        help="Number of clusters (for algorithms that require it)",
    ),
    pca_dimensions: int = typer.Option(
        50,
        "--pca-dims",
        "-p",
        help="Number of PCA dimensions for dimensionality reduction",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
):
    """
    Compare multiple clustering algorithms on the same dataset.

    This command runs multiple clustering algorithms and compares
    their performance using evaluation metrics.

    Educational Note: Algorithm comparison helps identify the most
    suitable clustering approach for the specific dataset.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Print research disclaimer
    typer.echo(RESEARCH_DISCLAIMER)
    typer.echo()

    try:
        # Parse algorithms
        algorithm_list = [alg.strip() for alg in algorithms.split(",")]

        # Validate input directory
        if not input_dir.exists():
            typer.echo(f"Error: Input directory {input_dir} does not exist", err=True)
            raise typer.Exit(1)

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load embeddings
        embedding_vectors = _load_embeddings(input_dir)
        if not embedding_vectors:
            typer.echo("No embedding vectors found!", err=True)
            raise typer.Exit(1)

        typer.echo(f"Loaded {len(embedding_vectors)} embedding vectors")
        typer.echo(f"Comparing algorithms: {algorithm_list}")

        # Compare algorithms
        comparison_results = _compare_algorithms(
            embedding_vectors, algorithm_list, n_clusters, pca_dimensions, config_file
        )

        # Save comparison results
        _save_comparison_results(comparison_results, output_dir)

        # Print comparison summary
        _print_comparison_summary(comparison_results)

        typer.echo("Algorithm comparison completed successfully!")
        typer.echo(f"Results saved to {output_dir}")

    except Exception as e:
        logger.error(f"Error in algorithm comparison: {e!s}")
        typer.echo(f"Error: {e!s}", err=True)
        raise typer.Exit(1)


@app.command()
def evaluate(
    clustering_result_file: Path = typer.Option(
        ..., "--result-file", "-r", help="Path to clustering result file"
    ),
    input_dir: Path = typer.Option(
        ..., "--input-dir", "-i", help="Directory containing embedding vectors"
    ),
    output_dir: Path = typer.Option(
        ..., "--output-dir", "-o", help="Directory to save evaluation results"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
):
    """
    Evaluate clustering results using various metrics.

    This command computes evaluation metrics for existing clustering
    results to assess clustering quality.

    Educational Note: Evaluation metrics help assess clustering quality
    without requiring ground truth labels.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Print research disclaimer
    typer.echo(RESEARCH_DISCLAIMER)
    typer.echo()

    try:
        # Validate input files
        if not clustering_result_file.exists():
            typer.echo(
                f"Error: Clustering result file {clustering_result_file} does not exist",
                err=True,
            )
            raise typer.Exit(1)

        if not input_dir.exists():
            typer.echo(f"Error: Input directory {input_dir} does not exist", err=True)
            raise typer.Exit(1)

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load clustering result
        clustering_result = _load_clustering_result(clustering_result_file)
        if clustering_result is None:
            typer.echo("Failed to load clustering result!", err=True)
            raise typer.Exit(1)

        # Load embeddings
        embedding_vectors = _load_embeddings(input_dir)
        if not embedding_vectors:
            typer.echo("No embedding vectors found!", err=True)
            raise typer.Exit(1)

        typer.echo(f"Evaluating clustering result: {clustering_result.algorithm}")
        typer.echo(f"Loaded {len(embedding_vectors)} embedding vectors")

        # Evaluate clustering
        evaluation_results = _evaluate_clustering(clustering_result, embedding_vectors)

        # Save evaluation results
        _save_evaluation_results(evaluation_results, output_dir)

        # Print evaluation summary
        _print_evaluation_summary(evaluation_results)

        typer.echo("Clustering evaluation completed successfully!")
        typer.echo(f"Results saved to {output_dir}")

    except Exception as e:
        logger.error(f"Error in clustering evaluation: {e!s}")
        typer.echo(f"Error: {e!s}", err=True)
        raise typer.Exit(1)


@app.command()
def config_template(
    output_file: Path = typer.Option(
        "clustering_config.yaml",
        "--output",
        "-o",
        help="Output file for configuration template",
    )
):
    """
    Generate a clustering configuration template.

    This command creates a template configuration file with
    default values for clustering parameters.

    Educational Note: Configuration templates help users understand
    available parameters and their effects on clustering.
    """
    # Print research disclaimer
    typer.echo(RESEARCH_DISCLAIMER)
    typer.echo()

    try:
        # Generate template configuration
        template_config = _generate_config_template()

        # Save template
        with open(output_file, "w") as f:
            yaml.dump(template_config, f, default_flow_style=False, indent=2)

        typer.echo(f"Configuration template saved to {output_file}")
        typer.echo("Edit the configuration file to customize clustering parameters")

    except Exception as e:
        logger.error(f"Error generating config template: {e!s}")
        typer.echo(f"Error: {e!s}", err=True)
        raise typer.Exit(1)


def _load_config(
    config_file: Optional[Path], algorithm: str, n_clusters: int, pca_dimensions: int
) -> Dict[str, Any]:
    """
    Load clustering configuration.

    Educational Note: Configuration loading enables reproducible
    experiments and parameter management.

    Args:
        config_file: Optional path to configuration file
        algorithm: Clustering algorithm name
        n_clusters: Number of clusters
        pca_dimensions: Number of PCA dimensions

    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    if config_file and config_file.exists():
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_file}")
    else:
        # Use default configuration
        config = _get_default_config()
        logger.info("Using default configuration")

    # Override with command line parameters
    config["algorithm"] = algorithm
    config["n_clusters"] = n_clusters
    config["pca_dimensions"] = pca_dimensions

    return config


def _get_default_config() -> Dict[str, Any]:
    """
    Get default clustering configuration.

    Educational Note: Default configurations provide sensible
    starting points for clustering parameters.

    Returns:
        Dict[str, Any]: Default configuration dictionary
    """
    return {
        "algorithm": "kmeans",
        "n_clusters": 4,
        "pca_dimensions": 50,
        "evaluation_metrics": ["silhouette", "davies_bouldin", "calinski_harabasz"],
        "seed": 42,
        "hyperparameters": {},
    }


def _load_embeddings(input_dir: Path) -> List[EmbeddingVector]:
    """
    Load embedding vectors from directory.

    Educational Note: Efficient embedding loading enables
    fast processing for clustering operations.

    Args:
        input_dir: Input directory path

    Returns:
        List[EmbeddingVector]: List of loaded embedding vectors
    """
    embedding_vectors = []

    try:
        # Find embedding files
        embedding_files = list(input_dir.rglob("*.embedding.pt"))

        for file_path in embedding_files:
            try:
                # Load embedding data
                embedding_data = torch.load(file_path, map_location="cpu")

                # Load metadata
                metadata_path = file_path.with_suffix(".metadata.yaml")
                if metadata_path.exists():
                    with open(metadata_path, "r") as f:
                        metadata = yaml.safe_load(f)
                else:
                    metadata = {}

                # Create EmbeddingVector instance
                embedding_vector = EmbeddingVector(
                    image_id=metadata.get("image_id", file_path.stem),
                    patient_id=metadata.get("patient_id", "unknown"),
                    embedding=embedding_data,
                    extraction_config=metadata.get("extraction_config", {}),
                    timestamp=datetime.fromisoformat(
                        metadata.get("timestamp", datetime.now().isoformat())
                    ),
                )

                embedding_vectors.append(embedding_vector)

            except Exception as e:
                logger.error(f"Error loading embedding from {file_path}: {e!s}")
                continue

        logger.info(f"Loaded {len(embedding_vectors)} embedding vectors")

    except Exception as e:
        logger.error(f"Error loading embeddings: {e!s}")

    return embedding_vectors


def _compare_algorithms(
    embedding_vectors: List[EmbeddingVector],
    algorithm_list: List[str],
    n_clusters: int,
    pca_dimensions: int,
    config_file: Optional[Path],
) -> Dict[str, Any]:
    """
    Compare multiple clustering algorithms.

    Educational Note: Algorithm comparison helps identify the most
    suitable clustering approach for the specific dataset.

    Args:
        embedding_vectors: List of EmbeddingVector instances
        algorithm_list: List of algorithm names to compare
        n_clusters: Number of clusters
        pca_dimensions: Number of PCA dimensions
        config_file: Optional configuration file

    Returns:
        Dict[str, Any]: Comparison results
    """
    comparison_results = {
        "algorithms": algorithm_list,
        "results": {},
        "comparison_metrics": {},
        "timestamp": datetime.now().isoformat(),
    }

    for algorithm in algorithm_list:
        try:
            typer.echo(f"Running {algorithm} clustering...")

            # Create configuration for this algorithm
            config = _load_config(config_file, algorithm, n_clusters, pca_dimensions)

            # Initialize clusterer
            clusterer = ClusteringAlgorithms(config)

            # Perform clustering
            clustering_result = clusterer.cluster_embeddings(embedding_vectors)

            if clustering_result is not None:
                comparison_results["results"][algorithm] = clustering_result

                # Extract metrics for comparison
                comparison_results["comparison_metrics"][algorithm] = {
                    "silhouette": clustering_result.metrics.get("silhouette", -1.0),
                    "davies_bouldin": clustering_result.metrics.get(
                        "davies_bouldin", float("inf")
                    ),
                    "calinski_harabasz": clustering_result.metrics.get(
                        "calinski_harabasz", 0.0
                    ),
                    "n_clusters": len(torch.unique(clustering_result.cluster_labels)),
                    "processing_time": clustering_result.processing_time,
                }

                typer.echo(
                    f"  {algorithm}: Silhouette={clustering_result.metrics.get('silhouette', -1.0):.3f}"
                )
            else:
                typer.echo(f"  {algorithm}: Failed")
                comparison_results["results"][algorithm] = None

        except Exception as e:
            logger.error(f"Error running {algorithm}: {e!s}")
            comparison_results["results"][algorithm] = None

    return comparison_results


def _evaluate_clustering(
    clustering_result: ClusteringResult, embedding_vectors: List[EmbeddingVector]
) -> Dict[str, Any]:
    """
    Evaluate clustering results using various metrics.

    Educational Note: Evaluation metrics help assess clustering quality
    without requiring ground truth labels.

    Args:
        clustering_result: ClusteringResult instance
        embedding_vectors: List of EmbeddingVector instances

    Returns:
        Dict[str, Any]: Evaluation results
    """
    evaluation_results = {
        "clustering_result": clustering_result,
        "evaluation_timestamp": datetime.now().isoformat(),
        "metrics": clustering_result.metrics,
        "cluster_statistics": {},
    }

    try:
        # Compute cluster statistics
        cluster_labels = clustering_result.cluster_labels.numpy()
        unique_labels, counts = torch.unique(
            clustering_result.cluster_labels, return_counts=True
        )

        evaluation_results["cluster_statistics"] = {
            "n_clusters": len(unique_labels),
            "cluster_sizes": {
                f"cluster_{label.item()}": count.item()
                for label, count in zip(unique_labels, counts, strict=False)
            },
            "total_samples": len(cluster_labels),
        }

        # Compute additional metrics if needed
        if "silhouette" not in clustering_result.metrics:
            from sklearn.metrics import silhouette_score

            try:
                # Extract embedding matrix
                embedding_matrix = torch.stack(
                    [emb.embedding for emb in embedding_vectors]
                ).numpy()
                silhouette = silhouette_score(embedding_matrix, cluster_labels)
                evaluation_results["metrics"]["silhouette"] = silhouette
            except Exception as e:
                logger.warning(f"Could not compute silhouette score: {e!s}")

    except Exception as e:
        logger.error(f"Error in clustering evaluation: {e!s}")
        evaluation_results["error"] = str(e)

    return evaluation_results


def _save_clustering_results(
    clustering_result: ClusteringResult, output_dir: Path
) -> None:
    """
    Save clustering results to files.

    Educational Note: Results saving enables reproducibility
    and further analysis of clustering outcomes.

    Args:
        clustering_result: ClusteringResult instance
        output_dir: Output directory path
    """
    try:
        # Save clustering result
        result_path = (
            output_dir / f"clustering_result_{clustering_result.experiment_id}.pt"
        )
        torch.save(clustering_result, result_path)

        # Save results summary
        summary = {
            "experiment_id": clustering_result.experiment_id,
            "algorithm": clustering_result.algorithm,
            "n_clusters": len(torch.unique(clustering_result.cluster_labels)),
            "n_samples": len(clustering_result.cluster_labels),
            "metrics": clustering_result.metrics,
            "processing_time": clustering_result.processing_time,
            "timestamp": clustering_result.timestamp.isoformat(),
        }

        summary_path = (
            output_dir / f"clustering_summary_{clustering_result.experiment_id}.yaml"
        )
        with open(summary_path, "w") as f:
            yaml.dump(summary, f, default_flow_style=False, indent=2)

        logger.info(f"Clustering results saved to {output_dir}")

    except Exception as e:
        logger.error(f"Error saving clustering results: {e!s}")


def _save_comparison_results(
    comparison_results: Dict[str, Any], output_dir: Path
) -> None:
    """
    Save algorithm comparison results.

    Educational Note: Comparison results enable analysis
    of different clustering approaches.

    Args:
        comparison_results: Comparison results dictionary
        output_dir: Output directory path
    """
    try:
        # Save comparison summary
        comparison_path = output_dir / "algorithm_comparison.yaml"
        with open(comparison_path, "w") as f:
            yaml.dump(comparison_results, f, default_flow_style=False, indent=2)

        # Save individual results
        for algorithm, result in comparison_results["results"].items():
            if result is not None:
                result_path = (
                    output_dir
                    / f"clustering_result_{algorithm}_{result.experiment_id}.pt"
                )
                torch.save(result, result_path)

        logger.info(f"Comparison results saved to {output_dir}")

    except Exception as e:
        logger.error(f"Error saving comparison results: {e!s}")


def _save_evaluation_results(
    evaluation_results: Dict[str, Any], output_dir: Path
) -> None:
    """
    Save clustering evaluation results.

    Educational Note: Evaluation results enable assessment
    of clustering quality and performance.

    Args:
        evaluation_results: Evaluation results dictionary
        output_dir: Output directory path
    """
    try:
        # Save evaluation results
        evaluation_path = output_dir / "clustering_evaluation.yaml"
        with open(evaluation_path, "w") as f:
            yaml.dump(evaluation_results, f, default_flow_style=False, indent=2)

        logger.info(f"Evaluation results saved to {output_dir}")

    except Exception as e:
        logger.error(f"Error saving evaluation results: {e!s}")


def _load_clustering_result(result_file: Path) -> Optional[ClusteringResult]:
    """
    Load clustering result from file.

    Educational Note: Result loading enables evaluation
    of previously computed clustering results.

    Args:
        result_file: Path to clustering result file

    Returns:
        ClusteringResult: Loaded clustering result, None if failed
    """
    try:
        clustering_result = torch.load(result_file, map_location="cpu")
        return clustering_result
    except Exception as e:
        logger.error(f"Error loading clustering result from {result_file}: {e!s}")
        return None


def _print_clustering_summary(clustering_result: ClusteringResult) -> None:
    """
    Print clustering results summary.

    Educational Note: Clear reporting helps users understand
    clustering outcomes and performance.

    Args:
        clustering_result: ClusteringResult instance
    """
    typer.echo("Clustering Summary:")
    typer.echo(f"  Algorithm: {clustering_result.algorithm}")
    typer.echo(f"  Experiment ID: {clustering_result.experiment_id}")
    typer.echo(
        f"  Number of clusters: {len(torch.unique(clustering_result.cluster_labels))}"
    )
    typer.echo(f"  Number of samples: {len(clustering_result.cluster_labels)}")
    typer.echo(f"  Processing time: {clustering_result.processing_time:.3f}s")

    typer.echo("  Metrics:")
    for metric, value in clustering_result.metrics.items():
        typer.echo(f"    {metric}: {value:.3f}")


def _print_comparison_summary(comparison_results: Dict[str, Any]) -> None:
    """
    Print algorithm comparison summary.

    Educational Note: Comparison summaries help users identify
    the best performing algorithm for their dataset.

    Args:
        comparison_results: Comparison results dictionary
    """
    typer.echo("Algorithm Comparison Summary:")

    # Create comparison table
    algorithms = list(comparison_results["comparison_metrics"].keys())
    metrics = ["silhouette", "davies_bouldin", "calinski_harabasz", "processing_time"]

    # Print header
    header = f"{'Algorithm':<15} {'Silhouette':<12} {'Davies-Bouldin':<15} {'Calinski-Harabasz':<18} {'Time (s)':<10}"
    typer.echo(header)
    typer.echo("-" * len(header))

    # Print results for each algorithm
    for algorithm in algorithms:
        if algorithm in comparison_results["comparison_metrics"]:
            metrics_data = comparison_results["comparison_metrics"][algorithm]
            row = f"{algorithm:<15} {metrics_data.get('silhouette', -1.0):<12.3f} {metrics_data.get('davies_bouldin', float('inf')):<15.3f} {metrics_data.get('calinski_harabasz', 0.0):<18.3f} {metrics_data.get('processing_time', 0.0):<10.3f}"
            typer.echo(row)

    # Find best algorithm by silhouette score
    best_algorithm = None
    best_silhouette = -1.0

    for algorithm, metrics_data in comparison_results["comparison_metrics"].items():
        silhouette = metrics_data.get("silhouette", -1.0)
        if silhouette > best_silhouette:
            best_silhouette = silhouette
            best_algorithm = algorithm

    if best_algorithm:
        typer.echo(
            f"\nBest algorithm by silhouette score: {best_algorithm} ({best_silhouette:.3f})"
        )


def _print_evaluation_summary(evaluation_results: Dict[str, Any]) -> None:
    """
    Print clustering evaluation summary.

    Educational Note: Evaluation summaries help users understand
    clustering quality and performance.

    Args:
        evaluation_results: Evaluation results dictionary
    """
    typer.echo("Clustering Evaluation Summary:")

    if "clustering_result" in evaluation_results:
        clustering_result = evaluation_results["clustering_result"]
        typer.echo(f"  Algorithm: {clustering_result.algorithm}")
        typer.echo(f"  Experiment ID: {clustering_result.experiment_id}")

    if "cluster_statistics" in evaluation_results:
        stats = evaluation_results["cluster_statistics"]
        typer.echo(f"  Number of clusters: {stats.get('n_clusters', 'N/A')}")
        typer.echo(f"  Total samples: {stats.get('total_samples', 'N/A')}")

        if "cluster_sizes" in stats:
            typer.echo("  Cluster sizes:")
            for cluster, size in stats["cluster_sizes"].items():
                typer.echo(f"    {cluster}: {size}")

    if "metrics" in evaluation_results:
        typer.echo("  Metrics:")
        for metric, value in evaluation_results["metrics"].items():
            typer.echo(f"    {metric}: {value:.3f}")


def _generate_config_template() -> Dict[str, Any]:
    """
    Generate clustering configuration template.

    Educational Note: Configuration templates help users
    understand available parameters and their effects.

    Returns:
        Dict[str, Any]: Configuration template
    """
    return {
        "algorithm": "kmeans",  # Options: kmeans, gmm, hdbscan, agglomerative
        "n_clusters": 4,
        "pca_dimensions": 50,
        "evaluation_metrics": ["silhouette", "davies_bouldin", "calinski_harabasz"],
        "seed": 42,
        "hyperparameters": {
            # K-means specific
            "n_init": 10,
            "max_iter": 300,
            # GMM specific
            "covariance_type": "full",
            "max_iter": 100,
            # HDBSCAN specific
            "min_cluster_size": 10,
            "min_samples": 5,
            # Agglomerative specific
            "linkage": "ward",
        },
    }


if __name__ == "__main__":
    app()
