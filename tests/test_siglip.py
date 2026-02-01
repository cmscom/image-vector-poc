"""Tests for SigLIP embedder."""

import numpy as np
import pytest
from PIL import Image

from image_vector_poc.embeddings.siglip import SigLIPEmbedder


@pytest.fixture(scope="module")
def embedder():
    """Create embedder once for all tests (model loading is slow)."""
    return SigLIPEmbedder(device="cuda")


def test_embedder_properties(embedder):
    assert embedder.model_name == "google/siglip-base-patch16-224"
    assert embedder.embedding_dim == 768


def test_embed_text(embedder):
    text = "a photo of a cat"
    embedding = embedder.embed_text(text)

    assert embedding.shape == (768,)
    assert embedding.dtype == np.float32
    # Should be normalized
    assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-5)


def test_embed_texts_batch(embedder):
    texts = ["a cat", "a dog", "a bird"]
    embeddings = embedder.embed_texts(texts)

    assert embeddings.shape == (3, 768)
    assert embeddings.dtype == np.float32


def test_embed_image(embedder):
    image = Image.new("RGB", (224, 224), color="blue")
    embedding = embedder.embed_image(image)

    assert embedding.shape == (768,)
    assert embedding.dtype == np.float32
    assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-5)


def test_embed_images_batch(embedder):
    images = [
        Image.new("RGB", (224, 224), color="red"),
        Image.new("RGB", (224, 224), color="green"),
    ]
    embeddings = embedder.embed_images(images)

    assert embeddings.shape == (2, 768)
    assert embeddings.dtype == np.float32


def test_text_image_similarity(embedder):
    # Red image should be more similar to "red" than "blue"
    red_image = Image.new("RGB", (224, 224), color="red")
    image_emb = embedder.embed_image(red_image)

    red_text_emb = embedder.embed_text("red color")
    blue_text_emb = embedder.embed_text("blue color")

    red_similarity = np.dot(image_emb, red_text_emb)
    blue_similarity = np.dot(image_emb, blue_text_emb)

    assert red_similarity > blue_similarity
