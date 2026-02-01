"""Tests for DuckDB vector store."""

import numpy as np
import pytest

from image_vector_poc.storage.duckdb_store import DuckDBVectorStore


@pytest.fixture
def store():
    """Create an in-memory store for testing."""
    s = DuckDBVectorStore(":memory:", embedding_dim=128)
    yield s
    s.close()


def test_store_initialization(store):
    assert store.count() == 0
    assert store.embedding_dim == 128


def test_add_and_count(store):
    embeddings = np.random.randn(3, 128).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    store.add(
        ids=["id1", "id2", "id3"],
        embeddings=embeddings,
        file_paths=["a.jpg", "b.jpg", "c.jpg"],
    )

    assert store.count() == 3


def test_search(store):
    # Add a known vector
    query = np.random.randn(128).astype(np.float32)
    query = query / np.linalg.norm(query)

    store.add(ids=["target"], embeddings=[query], file_paths=["target.jpg"])

    # Search should find it with high score
    results = store.search(query, k=1)

    assert len(results) == 1
    assert results[0].id == "target"
    assert results[0].score > 0.99  # Should be ~1.0 for identical vector


def test_search_with_index(store):
    # Add vectors
    embeddings = np.random.randn(10, 128).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    ids = [f"id{i}" for i in range(10)]

    store.add(ids=ids, embeddings=embeddings)
    store.create_index(metric="cosine")

    # Search should still work
    results = store.search(embeddings[0], k=3)

    assert len(results) == 3
    assert results[0].id == "id0"  # Should find itself first


def test_delete(store):
    embeddings = np.random.randn(3, 128).astype(np.float32)
    store.add(ids=["a", "b", "c"], embeddings=embeddings)

    assert store.count() == 3

    store.delete(["b"])

    assert store.count() == 2


def test_get(store):
    embedding = np.random.randn(128).astype(np.float32)
    store.add(
        ids=["test"],
        embeddings=[embedding],
        file_paths=["test.jpg"],
        metadata=[{"tag": "sample"}],
    )

    result = store.get("test")

    assert result is not None
    assert result.id == "test"
    assert result.metadata["file_path"] == "test.jpg"
    assert result.metadata["tag"] == "sample"


def test_get_nonexistent(store):
    result = store.get("nonexistent")
    assert result is None


def test_clear(store):
    embeddings = np.random.randn(5, 128).astype(np.float32)
    store.add(ids=[f"id{i}" for i in range(5)], embeddings=embeddings)

    assert store.count() == 5

    store.clear()

    assert store.count() == 0


def test_context_manager():
    with DuckDBVectorStore(":memory:", embedding_dim=64) as store:
        assert store.count() == 0
