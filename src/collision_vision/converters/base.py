"""Base classes and data structures shared by all annotation converters.

The YOLOv8-Seg label format is one ``.txt`` file per image, with one line per
object instance::

    <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>

where every coordinate is normalized to ``[0, 1]`` by the image width/height.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Minimum number of vertices for a valid polygon (a triangle).
_MIN_POLYGON_POINTS = 3


@dataclass
class Polygon:
    """A single segmentation polygon in absolute pixel coordinates.

    Attributes:
        class_id: Zero-based class index.
        points: Ordered ``(x, y)`` vertices in pixels.
    """

    class_id: int
    points: list[tuple[float, float]]


@dataclass
class ImageAnnotation:
    """All polygon annotations for a single image.

    Attributes:
        filename: Image file name (as referenced by the annotation source).
        width: Image width in pixels.
        height: Image height in pixels.
        polygons: Segmentation polygons for this image.
    """

    filename: str
    width: int
    height: int
    polygons: list[Polygon] = field(default_factory=list)


class BaseConverter(ABC):
    """Abstract base class for annotation-to-YOLOv8-Seg converters.

    Subclasses implement :meth:`parse` to yield :class:`ImageAnnotation`
    objects; the base class handles normalization and label writing.
    """

    def __init__(self, annotations_path: str | Path, class_map: dict[str, int] | None = None):
        """Initialize the converter.

        Args:
            annotations_path: Path to the annotation export (JSON).
            class_map: Optional mapping of lowercase source-label -> class index.
        """
        self.annotations_path = Path(annotations_path)
        if not self.annotations_path.is_file():
            raise FileNotFoundError(f"Annotation file not found: {self.annotations_path}")
        self.class_map = {k.lower(): v for k, v in (class_map or {}).items()}

    @abstractmethod
    def parse(self) -> list[ImageAnnotation]:
        """Parse the source annotations into :class:`ImageAnnotation` objects."""
        raise NotImplementedError

    @staticmethod
    def normalize(
        points: list[tuple[float, float]], width: int, height: int
    ) -> list[float]:
        """Normalize and clip polygon vertices to ``[0, 1]``.

        Args:
            points: Absolute ``(x, y)`` pixel vertices.
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            A flat list ``[x1, y1, x2, y2, ...]`` of normalized coordinates.
        """
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image size: {width}x{height}")
        flat: list[float] = []
        for x, y in points:
            nx = min(max(x / width, 0.0), 1.0)
            ny = min(max(y / height, 0.0), 1.0)
            flat.extend((nx, ny))
        return flat

    def to_yolo_lines(self, annotation: ImageAnnotation) -> list[str]:
        """Convert one image's polygons into YOLOv8-Seg label lines."""
        lines: list[str] = []
        for polygon in annotation.polygons:
            if len(polygon.points) < _MIN_POLYGON_POINTS:
                logger.warning(
                    "Skipping polygon with %d point(s) in %s (need >= %d)",
                    len(polygon.points),
                    annotation.filename,
                    _MIN_POLYGON_POINTS,
                )
                continue
            coords = self.normalize(polygon.points, annotation.width, annotation.height)
            coord_str = " ".join(f"{c:.6f}" for c in coords)
            lines.append(f"{polygon.class_id} {coord_str}")
        return lines

    def write_labels(self, output_dir: str | Path) -> int:
        """Parse annotations and write one ``.txt`` label file per image.

        Args:
            output_dir: Directory to write label files into (created if needed).

        Returns:
            The number of label files written.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        for annotation in self.parse():
            lines = self.to_yolo_lines(annotation)
            if not lines:
                continue
            label_path = output_dir / f"{Path(annotation.filename).stem}.txt"
            label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            written += 1
        logger.info("Wrote %d label file(s) to %s", written, output_dir)
        return written
