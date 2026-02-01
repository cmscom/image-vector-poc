"""Evaluation metrics for embedding quality assessment."""

from dataclasses import asdict, dataclass, field
from datetime import datetime

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import silhouette_score


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""

    model_name: str
    embedding_dim: int
    timestamp: str

    # Dataset info
    total_images: int
    categories: list[str]
    category_counts: dict[str, int]

    # 2D metrics (t-SNE)
    silhouette_2d: float
    trustworthiness_2d: float
    distance_ratio_2d: float

    # 3D metrics (t-SNE)
    silhouette_3d: float
    trustworthiness_3d: float
    distance_ratio_3d: float

    # PCA metrics (for comparison)
    pca_silhouette_2d: float = 0.0
    pca_silhouette_3d: float = 0.0
    pca_variance_ratio_2d: float = 0.0
    pca_variance_ratio_3d: float = 0.0

    # Performance
    processing_time_seconds: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


def compute_distance_ratio(coords: NDArray, labels: NDArray) -> float:
    """Compute intra-category / inter-category distance ratio.

    Lower values indicate better category separation.

    Args:
        coords: 2D array of coordinates (n_samples, n_dims)
        labels: 1D array of category labels

    Returns:
        Ratio of mean intra-category distance to mean inter-category distance
    """
    dist_matrix = squareform(pdist(coords))
    intra_distances = []
    inter_distances = []

    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                intra_distances.append(dist_matrix[i, j])
            else:
                inter_distances.append(dist_matrix[i, j])

    if not intra_distances or not inter_distances:
        return float("nan")

    return float(np.mean(intra_distances) / np.mean(inter_distances))


def evaluate_embeddings(
    embeddings: NDArray[np.float32],
    labels: NDArray,
    model_name: str,
    embedding_dim: int,
    categories: list[str],
    category_counts: dict[str, int],
    processing_time: float = 0.0,
    random_state: int = 42,
    perplexity: int = 30,
    n_neighbors: int = 10,
) -> EvaluationMetrics:
    """Run full evaluation on embeddings.

    Args:
        embeddings: 2D array of embeddings (n_samples, embedding_dim)
        labels: 1D array of category labels
        model_name: Name of the embedding model
        embedding_dim: Dimension of embeddings
        categories: List of unique category names
        category_counts: Dict mapping category name to count
        processing_time: Time taken to generate embeddings (seconds)
        random_state: Random seed for reproducibility
        perplexity: t-SNE perplexity parameter
        n_neighbors: Number of neighbors for trustworthiness calculation

    Returns:
        EvaluationMetrics dataclass with all computed metrics
    """
    timestamp = datetime.now().isoformat()

    # Ensure labels is numpy array
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)

    # PCA dimensionality reduction
    pca_2d = PCA(n_components=2, random_state=random_state)
    pca_3d = PCA(n_components=3, random_state=random_state)

    coords_pca_2d = pca_2d.fit_transform(embeddings)
    coords_pca_3d = pca_3d.fit_transform(embeddings)

    pca_variance_2d = float(np.sum(pca_2d.explained_variance_ratio_))
    pca_variance_3d = float(np.sum(pca_3d.explained_variance_ratio_))

    # t-SNE dimensionality reduction
    tsne_2d = TSNE(
        n_components=2,
        random_state=random_state,
        perplexity=perplexity,
        max_iter=1000,
    )
    tsne_3d = TSNE(
        n_components=3,
        random_state=random_state,
        perplexity=perplexity,
        max_iter=1000,
    )

    coords_tsne_2d = tsne_2d.fit_transform(embeddings)
    coords_tsne_3d = tsne_3d.fit_transform(embeddings)

    # Compute metrics for t-SNE
    silhouette_2d = float(silhouette_score(coords_tsne_2d, labels))
    silhouette_3d = float(silhouette_score(coords_tsne_3d, labels))

    trust_2d = float(trustworthiness(embeddings, coords_tsne_2d, n_neighbors=n_neighbors))
    trust_3d = float(trustworthiness(embeddings, coords_tsne_3d, n_neighbors=n_neighbors))

    dist_ratio_2d = compute_distance_ratio(coords_tsne_2d, labels)
    dist_ratio_3d = compute_distance_ratio(coords_tsne_3d, labels)

    # Compute metrics for PCA
    pca_silhouette_2d = float(silhouette_score(coords_pca_2d, labels))
    pca_silhouette_3d = float(silhouette_score(coords_pca_3d, labels))

    return EvaluationMetrics(
        model_name=model_name,
        embedding_dim=embedding_dim,
        timestamp=timestamp,
        total_images=len(embeddings),
        categories=categories,
        category_counts=category_counts,
        silhouette_2d=silhouette_2d,
        trustworthiness_2d=trust_2d,
        distance_ratio_2d=dist_ratio_2d,
        silhouette_3d=silhouette_3d,
        trustworthiness_3d=trust_3d,
        distance_ratio_3d=dist_ratio_3d,
        pca_silhouette_2d=pca_silhouette_2d,
        pca_silhouette_3d=pca_silhouette_3d,
        pca_variance_ratio_2d=pca_variance_2d,
        pca_variance_ratio_3d=pca_variance_3d,
        processing_time_seconds=processing_time,
    )
