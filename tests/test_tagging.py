"""Tests for zero-shot tagging."""

import pytest
from PIL import Image

from image_vector_poc.embeddings.siglip import SigLIPEmbedder
from image_vector_poc.tagging.zero_shot import ZeroShotTagger


@pytest.fixture(scope="module")
def embedder():
    return SigLIPEmbedder(device="cuda")


@pytest.fixture
def tagger(embedder):
    categories = ["red", "blue", "green", "yellow"]
    return ZeroShotTagger(embedder, categories)


def test_tagger_initialization(tagger):
    assert len(tagger.categories) == 4
    assert tagger._category_embeddings is not None
    assert tagger._category_embeddings.shape[0] == 4


def test_tag_image(tagger):
    # Red image should get "red" as top tag
    red_image = Image.new("RGB", (224, 224), color="red")
    tags = tagger.tag(red_image, top_k=3)

    assert len(tags) <= 3
    # First tag should be "red"
    assert tags[0][0] == "red"
    # Score should be a float
    assert isinstance(tags[0][1], float)


def test_tag_with_threshold(tagger):
    image = Image.new("RGB", (224, 224), color="red")
    # High threshold should filter out low-confidence tags
    tags = tagger.tag(image, top_k=4, threshold=0.5)

    # All returned tags should have score >= 0.5
    for tag, score in tags:
        assert score >= 0.5


def test_tag_batch(tagger):
    images = [
        Image.new("RGB", (224, 224), color="red"),
        Image.new("RGB", (224, 224), color="blue"),
    ]

    results = tagger.tag_batch(images, top_k=2)

    assert len(results) == 2
    assert results[0][0][0] == "red"  # First image top tag
    assert results[1][0][0] == "blue"  # Second image top tag


def test_add_categories(tagger):
    initial_count = len(tagger.categories)
    tagger.add_categories(["purple", "orange"])

    assert len(tagger.categories) == initial_count + 2
    assert "purple" in tagger.categories


def test_set_categories(tagger):
    tagger.set_categories(["black", "white"])

    assert len(tagger.categories) == 2
    assert "red" not in tagger.categories
    assert "black" in tagger.categories


def test_set_template(embedder):
    categories = ["cat", "dog"]
    tagger = ZeroShotTagger(embedder, categories)

    # Change template
    tagger.set_template("A photograph showing a {label}.")

    # Embeddings should be recomputed
    assert tagger._category_embeddings is not None
