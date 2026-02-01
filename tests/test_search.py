"""Tests for semantic search."""

import numpy as np
import pytest
from PIL import Image

from image_vector_poc.embeddings.siglip import SigLIPEmbedder
from image_vector_poc.storage.duckdb_store import DuckDBVectorStore
from image_vector_poc.search.semantic import SemanticSearch


@pytest.fixture(scope="module")
def embedder():
    return SigLIPEmbedder(device="cuda")


@pytest.fixture
def store(embedder):
    s = DuckDBVectorStore(":memory:", embedding_dim=embedder.embedding_dim)
    yield s
    s.close()


@pytest.fixture
def search(embedder, store):
    return SemanticSearch(embedder, store)


def test_index_and_search_by_text(search, store):
    # Create and index a red image
    red_image = Image.new("RGB", (224, 224), color="red")
    search.index_image("red1", red_image, "red.jpg", {"color": "red"})
    store.create_index()

    # Search for red
    results = search.search_by_text("red color", k=1)

    assert len(results) == 1
    assert results[0].id == "red1"


def test_index_and_search_by_image(search, store):
    # Index some images
    red_image = Image.new("RGB", (224, 224), color="red")
    blue_image = Image.new("RGB", (224, 224), color="blue")

    search.index_image("red1", red_image, "red.jpg")
    search.index_image("blue1", blue_image, "blue.jpg")
    store.create_index()

    # Search with a red query image
    query = Image.new("RGB", (224, 224), color="red")
    results = search.search_by_image(query, k=2)

    assert len(results) == 2
    # Red should be more similar to red
    assert results[0].id == "red1"


def test_index_images_batch(search, store, embedder):
    images = [
        Image.new("RGB", (224, 224), color="red"),
        Image.new("RGB", (224, 224), color="green"),
        Image.new("RGB", (224, 224), color="blue"),
    ]

    search.index_images(
        ids=["r", "g", "b"],
        images=images,
        file_paths=["r.jpg", "g.jpg", "b.jpg"],
    )

    assert store.count() == 3
