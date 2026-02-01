"""Data types and structures for image vector search."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image


@dataclass
class BoundingBox:
    """Bounding box coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int

    def crop(self, image: Image.Image) -> Image.Image:
        """Crop the region from an image."""
        return image.crop((self.x1, self.y1, self.x2, self.y2))

    @property
    def width(self) -> int:
        """Width of the bounding box."""
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        """Height of the bounding box."""
        return self.y2 - self.y1

    def to_tuple(self) -> tuple[int, int, int, int]:
        """Convert to tuple (x1, y1, x2, y2)."""
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class SearchResult:
    """Result from a vector similarity search."""

    id: str
    score: float  # Similarity score (higher = more similar)
    distance: float  # Distance (lower = more similar)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Detection:
    """Detected object or face in an image."""

    label: str
    confidence: float
    bbox: BoundingBox
    embedding: NDArray[np.float32] | None = None


@dataclass
class ImageRecord:
    """Represents an indexed image."""

    id: str
    path: Path
    embedding: NDArray[np.float32] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
