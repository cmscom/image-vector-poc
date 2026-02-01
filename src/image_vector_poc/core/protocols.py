"""Protocol definitions for image vector search components."""

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .types import SearchResult


@runtime_checkable
class ImageEmbedder(Protocol):
    """Protocol for image embedding models."""

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        ...

    @property
    def model_name(self) -> str:
        """Return model identifier."""
        ...

    def embed_image(self, image: Image.Image) -> NDArray[np.float32]:
        """Generate embedding for a single image."""
        ...

    def embed_images(self, images: list[Image.Image]) -> NDArray[np.float32]:
        """Generate embeddings for multiple images (batch)."""
        ...


@runtime_checkable
class TextEmbedder(Protocol):
    """Protocol for text embedding models."""

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        ...

    def embed_text(self, text: str) -> NDArray[np.float32]:
        """Generate embedding for a single text."""
        ...

    def embed_texts(self, texts: list[str]) -> NDArray[np.float32]:
        """Generate embeddings for multiple texts (batch)."""
        ...


@runtime_checkable
class MultimodalEmbedder(ImageEmbedder, TextEmbedder, Protocol):
    """Combined protocol for models that embed both images and text."""

    pass


@runtime_checkable
class VectorStore(Protocol):
    """Protocol for vector storage backends."""

    def add(
        self,
        ids: list[str],
        embeddings: NDArray[np.float32],
        file_paths: list[str] | None = None,
        metadata: list[dict] | None = None,
    ) -> None:
        """Add vectors to storage."""
        ...

    def search(
        self,
        query_embedding: NDArray[np.float32],
        k: int = 10,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Search for similar vectors."""
        ...

    def delete(self, ids: list[str]) -> None:
        """Delete vectors by ID."""
        ...

    def count(self) -> int:
        """Return number of vectors in storage."""
        ...
