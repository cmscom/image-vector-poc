"""Image Vector PoC - Image vector search and analysis toolkit."""

from .core.protocols import (
    ImageEmbedder,
    MultimodalEmbedder,
    TextEmbedder,
    VectorStore,
)
from .core.types import BoundingBox, Detection, ImageRecord, SearchResult
from .embeddings.clip import CLIPEmbedder
from .embeddings.dinov2 import DINOv2Embedder
from .embeddings.japanese_clip import JapaneseCLIPEmbedder
from .embeddings.jina_clip import JinaCLIPEmbedder
from .embeddings.openclip import OpenCLIPEmbedder
from .embeddings.siglip import SigLIPEmbedder
from .embeddings.siglip2 import SigLIP2Embedder
from .evaluation.metrics import EvaluationMetrics, evaluate_embeddings
from .evaluation.reporter import EvaluationReporter
from .search.semantic import SemanticSearch
from .storage.duckdb_store import DuckDBVectorStore
from .tagging.zero_shot import ZeroShotTagger

__all__ = [
    # Protocols
    "ImageEmbedder",
    "TextEmbedder",
    "MultimodalEmbedder",
    "VectorStore",
    # Types
    "SearchResult",
    "Detection",
    "ImageRecord",
    "BoundingBox",
    # Embedders
    "SigLIPEmbedder",
    "SigLIP2Embedder",
    "CLIPEmbedder",
    "DINOv2Embedder",
    "OpenCLIPEmbedder",
    "JapaneseCLIPEmbedder",
    "JinaCLIPEmbedder",
    # Other Implementations
    "DuckDBVectorStore",
    "SemanticSearch",
    "ZeroShotTagger",
    # Evaluation
    "EvaluationMetrics",
    "EvaluationReporter",
    "evaluate_embeddings",
]
