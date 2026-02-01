"""Jina CLIP v2 multilingual embedder."""

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from transformers import AutoModel, AutoProcessor


class JinaCLIPEmbedder:
    """Jina CLIP v2 multilingual embedder for images and text.

    This embedder uses Jina AI's CLIP v2 model which supports 89 languages
    including Japanese, and provides 1024-dimensional embeddings.

    Example:
        >>> embedder = JinaCLIPEmbedder(device="cuda")
        >>> image = Image.open("photo.jpg")
        >>> image_emb = embedder.embed_image(image)
        >>> text_emb = embedder.embed_text("a photo of a cat")  # or Japanese text
        >>> similarity = np.dot(image_emb, text_emb)
    """

    DEFAULT_MODEL = "jinaai/jina-clip-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        normalize: bool = True,
    ):
        """Initialize the Jina CLIP embedder.

        Args:
            model_name: HuggingFace model identifier for Jina CLIP.
            device: Device to run the model on ("cuda", "cpu", or None for auto).
            normalize: Whether to L2-normalize embeddings (recommended for cosine similarity).
        """
        self._model_name = model_name
        self.normalize = normalize

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()

    @property
    def model_name(self) -> str:
        """Return model identifier."""
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        # Jina CLIP v2 has 1024-dimensional embeddings
        return 1024

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """L2-normalize embeddings."""
        if self.normalize:
            norm = np.linalg.norm(embeddings, axis=-1, keepdims=True)
            embeddings = embeddings / np.clip(norm, a_min=1e-8, a_max=None)
        return embeddings

    def embed_image(self, image: Image.Image) -> NDArray[np.float32]:
        """Generate embedding for a single image.

        Args:
            image: PIL Image to embed.

        Returns:
            1D numpy array of shape (embedding_dim,).
        """
        return self.embed_images([image])[0]

    def embed_images(self, images: list[Image.Image]) -> NDArray[np.float32]:
        """Generate embeddings for multiple images.

        Args:
            images: List of PIL Images to embed.

        Returns:
            2D numpy array of shape (n_images, embedding_dim).
        """
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)

        # Handle both tensor and BaseModelOutputWithPooling
        if hasattr(outputs, "pooler_output"):
            embeddings = outputs.pooler_output.cpu().numpy()
        else:
            embeddings = outputs.cpu().numpy()
        embeddings = self._normalize(embeddings)
        return embeddings.astype(np.float32)

    def embed_text(self, text: str) -> NDArray[np.float32]:
        """Generate embedding for a single text.

        Args:
            text: Text string to embed (supports 89 languages).

        Returns:
            1D numpy array of shape (embedding_dim,).
        """
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> NDArray[np.float32]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of text strings to embed (supports 89 languages).

        Returns:
            2D numpy array of shape (n_texts, embedding_dim).
        """
        inputs = self.processor(
            text=texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)

        # Handle both tensor and BaseModelOutputWithPooling
        if hasattr(outputs, "pooler_output"):
            embeddings = outputs.pooler_output.cpu().numpy()
        else:
            embeddings = outputs.cpu().numpy()
        embeddings = self._normalize(embeddings)
        return embeddings.astype(np.float32)
