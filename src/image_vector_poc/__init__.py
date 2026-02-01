"""Image Vector PoC - Image vector search and analysis toolkit."""

from .core.protocols import (
    ImageEmbedder,
    MultimodalEmbedder,
    TextEmbedder,
    VectorStore,
)
from .core.types import BoundingBox, Detection, ImageRecord, SearchResult
from .embeddings.siglip import SigLIPEmbedder
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
    # Implementations
    "SigLIPEmbedder",
    "DuckDBVectorStore",
    "SemanticSearch",
    "ZeroShotTagger",
]
