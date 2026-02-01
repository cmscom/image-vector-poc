"""Meta DINOv2 image embedder (image-only, no text support)."""

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


class DINOv2Embedder:
    """Meta DINOv2 image embedder.

    This embedder uses Meta's DINOv2 model to generate embeddings for images.
    Note: DINOv2 is image-only and does not support text embeddings.

    Example:
        >>> embedder = DINOv2Embedder(device="cuda")
        >>> image = Image.open("photo.jpg")
        >>> image_emb = embedder.embed_image(image)
    """

    DEFAULT_MODEL = "facebook/dinov2-large"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        normalize: bool = True,
    ):
        """Initialize the DINOv2 embedder.

        Args:
            model_name: HuggingFace model identifier for DINOv2.
            device: Device to run the model on ("cuda", "cpu", or None for auto).
            normalize: Whether to L2-normalize embeddings (recommended for cosine similarity).
        """
        self._model_name = model_name
        self.normalize = normalize

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model = AutoModel.from_pretrained(model_name)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @property
    def model_name(self) -> str:
        """Return model identifier."""
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        return self.model.config.hidden_size

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
            outputs = self.model(**inputs)

        # Use CLS token embedding (first token)
        embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()
        embeddings = self._normalize(embeddings)
        return embeddings.astype(np.float32)
