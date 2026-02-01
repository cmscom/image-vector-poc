"""Zero-shot image classification using text-image similarity."""

import numpy as np
from PIL import Image

from ..core.protocols import MultimodalEmbedder


class ZeroShotTagger:
    """Zero-shot image tagger using multimodal embeddings.

    This tagger classifies images into categories without
    requiring training data, using text-image similarity.

    Example:
        >>> from image_vector_poc.embeddings.siglip import SigLIPEmbedder
        >>>
        >>> embedder = SigLIPEmbedder()
        >>> categories = ["cat", "dog", "bird", "car", "person"]
        >>> tagger = ZeroShotTagger(embedder, categories)
        >>>
        >>> image = Image.open("photo.jpg")
        >>> tags = tagger.tag(image, top_k=3)
        >>> for tag, score in tags:
        ...     print(f"{tag}: {score:.3f}")
    """

    DEFAULT_TEMPLATE = "This is a photo of {label}."

    def __init__(
        self,
        embedder: MultimodalEmbedder,
        categories: list[str],
        template: str | None = None,
    ):
        """Initialize the zero-shot tagger.

        Args:
            embedder: Multimodal embedder for images and text.
            categories: List of category labels.
            template: Prompt template with {label} placeholder.
        """
        self.embedder = embedder
        self.categories = list(categories)
        self.template = template or self.DEFAULT_TEMPLATE

        # Pre-compute category embeddings
        self._category_embeddings: np.ndarray | None = None
        self._compute_category_embeddings()

    def _compute_category_embeddings(self) -> None:
        """Pre-compute and cache category embeddings."""
        if not self.categories:
            self._category_embeddings = None
            return

        prompts = [self.template.format(label=cat) for cat in self.categories]
        self._category_embeddings = self.embedder.embed_texts(prompts)

    def tag(
        self,
        image: Image.Image,
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> list[tuple[str, float]]:
        """Tag an image with zero-shot classification.

        Args:
            image: PIL Image to tag.
            top_k: Maximum number of tags to return.
            threshold: Minimum similarity score for a tag.

        Returns:
            List of (category, score) tuples sorted by score descending.
        """
        if self._category_embeddings is None or len(self.categories) == 0:
            return []

        image_embedding = self.embedder.embed_image(image)

        # Compute cosine similarity with all categories
        similarities = np.dot(self._category_embeddings, image_embedding)

        # Get top-k results above threshold
        indices = np.argsort(similarities)[::-1][:top_k]
        results = [
            (self.categories[i], float(similarities[i]))
            for i in indices
            if similarities[i] >= threshold
        ]

        return results

    def tag_batch(
        self,
        images: list[Image.Image],
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> list[list[tuple[str, float]]]:
        """Tag multiple images.

        Args:
            images: List of PIL Images to tag.
            top_k: Maximum number of tags per image.
            threshold: Minimum similarity score for a tag.

        Returns:
            List of tag lists, one per image.
        """
        if self._category_embeddings is None or len(self.categories) == 0:
            return [[] for _ in images]

        image_embeddings = self.embedder.embed_images(images)

        # Compute similarities: (n_images, n_categories)
        similarities = np.dot(image_embeddings, self._category_embeddings.T)

        results = []
        for i in range(len(images)):
            sims = similarities[i]
            indices = np.argsort(sims)[::-1][:top_k]
            tags = [
                (self.categories[j], float(sims[j]))
                for j in indices
                if sims[j] >= threshold
            ]
            results.append(tags)

        return results

    def add_categories(self, new_categories: list[str]) -> None:
        """Add new categories and update embeddings.

        Args:
            new_categories: Categories to add.
        """
        self.categories.extend(new_categories)
        self._compute_category_embeddings()

    def set_categories(self, categories: list[str]) -> None:
        """Replace all categories.

        Args:
            categories: New list of categories.
        """
        self.categories = list(categories)
        self._compute_category_embeddings()

    def set_template(self, template: str) -> None:
        """Change the prompt template and recompute embeddings.

        Args:
            template: New template with {label} placeholder.
        """
        self.template = template
        self._compute_category_embeddings()
