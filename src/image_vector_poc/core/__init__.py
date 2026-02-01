"""Core module with types and protocols."""

from .types import SearchResult, Detection, ImageRecord, BoundingBox
from .protocols import ImageEmbedder, TextEmbedder, MultimodalEmbedder, VectorStore

__all__ = [
    "SearchResult",
    "Detection",
    "ImageRecord",
    "BoundingBox",
    "ImageEmbedder",
    "TextEmbedder",
    "MultimodalEmbedder",
    "VectorStore",
]
