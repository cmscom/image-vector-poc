"""Semantic search for images using text and image queries."""

from pathlib import Path
from typing import Any

from PIL import Image

from ..core.protocols import MultimodalEmbedder, VectorStore
from ..core.types import SearchResult


class SemanticSearch:
    """Semantic search combining text-to-image and image-to-image search.

    This class provides a unified interface for searching images
    using either natural language queries or reference images.

    Example:
        >>> from image_vector_poc.embeddings.siglip import SigLIPEmbedder
        >>> from image_vector_poc.storage.duckdb_store import DuckDBVectorStore
        >>>
        >>> embedder = SigLIPEmbedder()
        >>> store = DuckDBVectorStore("vectors.db", embedder.embedding_dim)
        >>> search = SemanticSearch(embedder, store)
        >>>
        >>> # Text-to-image search
        >>> results = search.search_by_text("a cat sleeping", k=5)
        >>>
        >>> # Image-to-image search
        >>> results = search.search_by_image("query.jpg", k=5)
    """

    def __init__(
        self,
        embedder: MultimodalEmbedder,
        store: VectorStore,
    ):
        """Initialize semantic search.

        Args:
            embedder: Multimodal embedder for images and text.
            store: Vector store for similarity search.
        """
        self.embedder = embedder
        self.store = store

    def search_by_text(
        self,
        query: str,
        k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search images by natural language query.

        Args:
            query: Text query describing the desired images.
            k: Number of results to return.
            filters: Optional metadata filters.

        Returns:
            List of SearchResult objects sorted by relevance.
        """
        query_embedding = self.embedder.embed_text(query)
        return self.store.search(query_embedding, k=k, filters=filters)

    def search_by_image(
        self,
        image: Image.Image | Path | str,
        k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar images.

        Args:
            image: Query image (PIL Image, path, or string path).
            k: Number of results to return.
            filters: Optional metadata filters.

        Returns:
            List of SearchResult objects sorted by similarity.
        """
        if isinstance(image, (Path, str)):
            image = Image.open(image).convert("RGB")

        query_embedding = self.embedder.embed_image(image)
        return self.store.search(query_embedding, k=k, filters=filters)

    def index_image(
        self,
        id: str,
        image: Image.Image | Path | str,
        file_path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Index a single image.

        Args:
            id: Unique identifier for the image.
            image: Image to index (PIL Image, path, or string path).
            file_path: Optional file path to store.
            metadata: Optional metadata to store.
        """
        if isinstance(image, (Path, str)):
            if file_path is None:
                file_path = str(image)
            image = Image.open(image).convert("RGB")

        embedding = self.embedder.embed_image(image)
        self.store.add(
            ids=[id],
            embeddings=[embedding],
            file_paths=[file_path] if file_path else None,
            metadata=[metadata] if metadata else None,
        )

    def index_images(
        self,
        ids: list[str],
        images: list[Image.Image | Path | str],
        file_paths: list[str] | None = None,
        metadata: list[dict[str, Any]] | None = None,
        batch_size: int = 32,
    ) -> None:
        """Index multiple images.

        Args:
            ids: Unique identifiers for each image.
            images: Images to index.
            file_paths: Optional file paths for each image.
            metadata: Optional metadata for each image.
            batch_size: Number of images to process at once.
        """
        n = len(ids)
        file_paths = file_paths or [None] * n
        metadata = metadata or [None] * n

        for i in range(0, n, batch_size):
            batch_ids = ids[i : i + batch_size]
            batch_images = images[i : i + batch_size]
            batch_paths = file_paths[i : i + batch_size]
            batch_meta = metadata[i : i + batch_size]

            # Load images if needed
            pil_images = []
            resolved_paths = []
            for img, path in zip(batch_images, batch_paths):
                if isinstance(img, (Path, str)):
                    resolved_paths.append(path or str(img))
                    pil_images.append(Image.open(img).convert("RGB"))
                else:
                    resolved_paths.append(path or "")
                    pil_images.append(img)

            # Batch embed
            embeddings = self.embedder.embed_images(pil_images)

            # Store
            self.store.add(
                ids=batch_ids,
                embeddings=embeddings,
                file_paths=resolved_paths,
                metadata=[m or {} for m in batch_meta],
            )
