"""Similarity search evaluation metrics for image retrieval."""

from dataclasses import asdict, dataclass, field
from datetime import datetime

import numpy as np
from numpy.typing import NDArray


@dataclass
class SimilaritySearchMetrics:
    """Container for similarity search evaluation metrics."""

    model_name: str
    timestamp: str
    total_queries: int

    # Dataset info
    categories: list[str]
    category_counts: dict[str, int]

    # Precision@K metrics
    precision_at_1: float
    precision_at_3: float
    precision_at_5: float
    precision_at_10: float
    precision_at_20: float

    # Recall@K metrics
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    recall_at_20: float

    # Ranking metrics
    mrr: float  # Mean Reciprocal Rank
    map_score: float  # Mean Average Precision
    ndcg_at_5: float
    ndcg_at_10: float

    # Hit rate metrics
    hit_rate_at_1: float
    hit_rate_at_5: float
    hit_rate_at_10: float

    # Per-category breakdown
    category_metrics: dict[str, dict] = field(default_factory=dict)

    # Confusion matrix data
    confusion_matrix: list[list[float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


def precision_at_k(
    retrieved_categories: list[str],
    query_category: str,
    k: int,
) -> float:
    """Compute Precision@K.

    Args:
        retrieved_categories: List of categories for retrieved items (ordered by rank)
        query_category: The category of the query item
        k: Number of top results to consider

    Returns:
        Proportion of relevant items in top-K results
    """
    if k <= 0:
        return 0.0
    top_k = retrieved_categories[:k]
    relevant = sum(1 for cat in top_k if cat == query_category)
    return relevant / k


def recall_at_k(
    retrieved_categories: list[str],
    query_category: str,
    k: int,
    category_total: int,
) -> float:
    """Compute Recall@K.

    Args:
        retrieved_categories: List of categories for retrieved items (ordered by rank)
        query_category: The category of the query item
        k: Number of top results to consider
        category_total: Total number of items in the query category (excluding query itself)

    Returns:
        Proportion of all relevant items that appear in top-K
    """
    if k <= 0 or category_total <= 0:
        return 0.0
    top_k = retrieved_categories[:k]
    relevant = sum(1 for cat in top_k if cat == query_category)
    # Cannot retrieve more than available
    max_retrievable = min(k, category_total)
    return relevant / max_retrievable


def reciprocal_rank(
    retrieved_categories: list[str],
    query_category: str,
) -> float:
    """Compute Reciprocal Rank.

    Args:
        retrieved_categories: List of categories for retrieved items (ordered by rank)
        query_category: The category of the query item

    Returns:
        1/rank of first relevant item, or 0 if no relevant item found
    """
    for i, cat in enumerate(retrieved_categories):
        if cat == query_category:
            return 1.0 / (i + 1)
    return 0.0


def average_precision(
    retrieved_categories: list[str],
    query_category: str,
) -> float:
    """Compute Average Precision for a single query.

    Args:
        retrieved_categories: List of categories for retrieved items (ordered by rank)
        query_category: The category of the query item

    Returns:
        Average precision score
    """
    relevant_count = 0
    precision_sum = 0.0

    for i, cat in enumerate(retrieved_categories):
        if cat == query_category:
            relevant_count += 1
            precision_sum += relevant_count / (i + 1)

    total_relevant = sum(1 for c in retrieved_categories if c == query_category)
    return precision_sum / total_relevant if total_relevant > 0 else 0.0


def ndcg_at_k(
    retrieved_categories: list[str],
    query_category: str,
    k: int,
    category_total: int,
) -> float:
    """Compute Normalized Discounted Cumulative Gain at K.

    Args:
        retrieved_categories: List of categories for retrieved items (ordered by rank)
        query_category: The category of the query item
        k: Number of top results to consider
        category_total: Total number of relevant items available

    Returns:
        nDCG@K score
    """
    if k <= 0:
        return 0.0

    # DCG: sum of relevance / log2(rank + 1)
    dcg = 0.0
    for i, cat in enumerate(retrieved_categories[:k]):
        relevance = 1.0 if cat == query_category else 0.0
        dcg += relevance / np.log2(i + 2)  # +2 because rank starts at 1

    # Ideal DCG: all relevant items at top positions
    ideal_relevant = min(k, category_total)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_relevant))

    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(
    retrieved_categories: list[str],
    query_category: str,
    k: int,
) -> float:
    """Compute Hit Rate (Success) at K.

    Args:
        retrieved_categories: List of categories for retrieved items (ordered by rank)
        query_category: The category of the query item
        k: Number of top results to consider

    Returns:
        1.0 if at least one relevant item in top-K, else 0.0
    """
    top_k = retrieved_categories[:k]
    return 1.0 if query_category in top_k else 0.0


def build_confusion_matrix(
    query_categories: list[str],
    retrieved_categories_list: list[list[str]],
    category_names: list[str],
    k: int = 10,
) -> NDArray[np.float64]:
    """Build confusion matrix showing cross-category retrieval patterns.

    Args:
        query_categories: List of query categories
        retrieved_categories_list: List of retrieved category lists for each query
        category_names: Ordered list of category names
        k: Number of top results to consider

    Returns:
        Confusion matrix (rows: query category, cols: retrieved category)
        Values are normalized by row (percentage of retrieved per query category)
    """
    n_categories = len(category_names)
    cat_to_idx = {cat: i for i, cat in enumerate(category_names)}

    # Count matrix
    counts = np.zeros((n_categories, n_categories), dtype=np.float64)

    for query_cat, retrieved_cats in zip(query_categories, retrieved_categories_list):
        query_idx = cat_to_idx.get(query_cat)
        if query_idx is None:
            continue

        for cat in retrieved_cats[:k]:
            retrieved_idx = cat_to_idx.get(cat)
            if retrieved_idx is not None:
                counts[query_idx, retrieved_idx] += 1

    # Normalize by row (percentage)
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    confusion = counts / row_sums

    return confusion


def evaluate_similarity_search(
    embeddings: NDArray[np.float32],
    categories: list[str],
    category_names: list[str],
    category_counts: dict[str, int],
    model_name: str = "unknown",
    k_values: list[int] | None = None,
) -> SimilaritySearchMetrics:
    """Run full similarity search evaluation using leave-one-out approach.

    Args:
        embeddings: 2D array of embeddings (n_samples, embedding_dim)
        categories: List of category labels for each sample
        category_names: List of unique category names
        category_counts: Dict mapping category name to count
        model_name: Name of the embedding model
        k_values: List of K values for evaluation (default: [1, 3, 5, 10, 20])

    Returns:
        SimilaritySearchMetrics dataclass with all computed metrics
    """
    if k_values is None:
        k_values = [1, 3, 5, 10, 20]

    timestamp = datetime.now().isoformat()
    n_samples = len(embeddings)

    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    normalized_embeddings = embeddings / norms

    # Compute all pairwise similarities
    similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)

    # Containers for per-query metrics
    all_precisions = {k: [] for k in k_values}
    all_recalls = {k: [] for k in k_values}
    all_hit_rates = {k: [] for k in k_values}
    all_ndcg = {k: [] for k in k_values}
    all_rr = []  # Reciprocal ranks
    all_ap = []  # Average precisions

    # For confusion matrix
    query_cats = []
    retrieved_cats_list = []

    # Per-category metrics
    category_precisions = {cat: {k: [] for k in k_values} for cat in category_names}

    # Leave-one-out evaluation
    max_k = max(k_values)
    for i in range(n_samples):
        query_category = categories[i]
        query_cats.append(query_category)

        # Get similarities (excluding self)
        similarities = similarity_matrix[i].copy()
        similarities[i] = -np.inf  # Exclude self

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][: max_k + 1]

        # Get categories of retrieved items
        retrieved_categories = [categories[idx] for idx in top_indices if idx != i][
            :max_k
        ]
        retrieved_cats_list.append(retrieved_categories)

        # Category total (excluding query itself)
        cat_total = category_counts[query_category] - 1

        # Compute metrics for each k
        for k in k_values:
            p_at_k = precision_at_k(retrieved_categories, query_category, k)
            r_at_k = recall_at_k(retrieved_categories, query_category, k, cat_total)
            h_at_k = hit_rate_at_k(retrieved_categories, query_category, k)
            n_at_k = ndcg_at_k(retrieved_categories, query_category, k, cat_total)

            all_precisions[k].append(p_at_k)
            all_recalls[k].append(r_at_k)
            all_hit_rates[k].append(h_at_k)
            all_ndcg[k].append(n_at_k)

            # Per-category tracking
            category_precisions[query_category][k].append(p_at_k)

        # RR and AP (computed over all retrieved items)
        rr = reciprocal_rank(retrieved_categories, query_category)
        ap = average_precision(retrieved_categories, query_category)
        all_rr.append(rr)
        all_ap.append(ap)

    # Aggregate metrics
    def mean_metric(values: list) -> float:
        return float(np.mean(values)) if values else 0.0

    # Build per-category metrics dict
    per_category = {}
    for cat in category_names:
        per_category[cat] = {
            f"precision_at_{k}": mean_metric(category_precisions[cat][k])
            for k in k_values
        }
        per_category[cat]["count"] = category_counts[cat]

    # Build confusion matrix
    confusion = build_confusion_matrix(
        query_cats, retrieved_cats_list, category_names, k=10
    )

    return SimilaritySearchMetrics(
        model_name=model_name,
        timestamp=timestamp,
        total_queries=n_samples,
        categories=category_names,
        category_counts=category_counts,
        # Precision@K
        precision_at_1=mean_metric(all_precisions[1]),
        precision_at_3=mean_metric(all_precisions[3]),
        precision_at_5=mean_metric(all_precisions[5]),
        precision_at_10=mean_metric(all_precisions[10]),
        precision_at_20=mean_metric(all_precisions[20]),
        # Recall@K
        recall_at_1=mean_metric(all_recalls[1]),
        recall_at_5=mean_metric(all_recalls[5]),
        recall_at_10=mean_metric(all_recalls[10]),
        recall_at_20=mean_metric(all_recalls[20]),
        # Ranking metrics
        mrr=mean_metric(all_rr),
        map_score=mean_metric(all_ap),
        ndcg_at_5=mean_metric(all_ndcg[5]),
        ndcg_at_10=mean_metric(all_ndcg[10]),
        # Hit rate
        hit_rate_at_1=mean_metric(all_hit_rates[1]),
        hit_rate_at_5=mean_metric(all_hit_rates[5]),
        hit_rate_at_10=mean_metric(all_hit_rates[10]),
        # Per-category and confusion matrix
        category_metrics=per_category,
        confusion_matrix=confusion.tolist(),
    )
