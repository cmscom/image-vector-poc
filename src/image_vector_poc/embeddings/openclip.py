"""OpenCLIP embedder (LAION trained models)."""

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

import open_clip


class OpenCLIPEmbedder:
    """OpenCLIP multimodal embedder for images and text.

    This embedder uses OpenCLIP models trained on LAION datasets.
    Supports various model architectures (ViT-B, ViT-L, ViT-H, ViT-g, etc.).

    Example:
        >>> embedder = OpenCLIPEmbedder(device="cuda")
        >>> image = Image.open("photo.jpg")
        >>> image_emb = embedder.embed_image(image)
        >>> text_emb = embedder.embed_text("a photo of a cat")
        >>> similarity = np.dot(image_emb, text_emb)
    """

    DEFAULT_MODEL = "ViT-H-14"
    DEFAULT_PRETRAINED = "laion2b_s32b_b79k"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        pretrained: str = DEFAULT_PRETRAINED,
        device: str | None = None,
        normalize: bool = True,
    ):
        """Initialize the OpenCLIP embedder.

        Args:
            model_name: OpenCLIP model architecture (e.g., "ViT-H-14", "ViT-L-14").
            pretrained: Pretrained weights identifier (e.g., "laion2b_s32b_b79k").
            device: Device to run the model on ("cuda", "cpu", or None for auto).
            normalize: Whether to L2-normalize embeddings (recommended for cosine similarity).
        """
        self._model_name = f"openclip/{model_name}/{pretrained}"
        self._arch = model_name
        self._pretrained = pretrained
        self.normalize = normalize

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Get embedding dimension from model
        self._embedding_dim = self.model.visual.output_dim

    @property
    def model_name(self) -> str:
        """Return model identifier."""
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        return self._embedding_dim

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
        processed = torch.stack([self.preprocess(img) for img in images])
        processed = processed.to(self.device)

        with torch.no_grad():
            embeddings = self.model.encode_image(processed)

        embeddings = embeddings.cpu().numpy()
        embeddings = self._normalize(embeddings)
        return embeddings.astype(np.float32)

    def embed_text(self, text: str) -> NDArray[np.float32]:
        """Generate embedding for a single text.

        Args:
            text: Text string to embed.

        Returns:
            1D numpy array of shape (embedding_dim,).
        """
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> NDArray[np.float32]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            2D numpy array of shape (n_texts, embedding_dim).
        """
        tokens = self.tokenizer(texts).to(self.device)

        with torch.no_grad():
            embeddings = self.model.encode_text(tokens)

        embeddings = embeddings.cpu().numpy()
        embeddings = self._normalize(embeddings)
        return embeddings.astype(np.float32)
