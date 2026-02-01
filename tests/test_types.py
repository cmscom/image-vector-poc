"""Tests for core types."""

from image_vector_poc.core.types import BoundingBox, SearchResult, ImageRecord
from PIL import Image


def test_bounding_box_properties():
    bbox = BoundingBox(x1=10, y1=20, x2=110, y2=120)

    assert bbox.width == 100
    assert bbox.height == 100
    assert bbox.to_tuple() == (10, 20, 110, 120)


def test_bounding_box_crop():
    bbox = BoundingBox(x1=0, y1=0, x2=50, y2=50)
    image = Image.new("RGB", (100, 100), color="red")

    cropped = bbox.crop(image)

    assert cropped.size == (50, 50)


def test_search_result():
    result = SearchResult(
        id="test_id",
        score=0.95,
        distance=0.05,
        metadata={"file_path": "test.jpg"},
    )

    assert result.id == "test_id"
    assert result.score == 0.95
    assert result.distance == 0.05
    assert result.metadata["file_path"] == "test.jpg"


def test_image_record():
    from pathlib import Path

    record = ImageRecord(
        id="img1",
        path=Path("images/test.jpg"),
        tags=["cat", "animal"],
    )

    assert record.id == "img1"
    assert record.path == Path("images/test.jpg")
    assert "cat" in record.tags
    assert record.embedding is None
