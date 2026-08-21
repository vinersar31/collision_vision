"""Convert VGG Image Annotator (VIA) JSON exports to YOLOv8-Seg labels.

Supports both common VIA layouts:

* **VIA 1.x** (``via_region_data.json``) — a flat mapping of
  ``"<filename><size>"`` keys to entries whose ``regions`` is a *dict*.
* **VIA 2.x** project export — a wrapper with a ``_via_img_metadata`` mapping
  whose entries have ``regions`` as a *list*.

VIA does not store image dimensions, so the original images are read from
``images_dir`` to obtain their width/height.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
from pathlib import Path
from typing import Any

from PIL import Image

from .base import BaseConverter, ImageAnnotation, Polygon

logger = logging.getLogger(__name__)


class VIAConverter(BaseConverter):
    """Converter for VGG Image Annotator polygon annotations."""

    def __init__(
        self,
        annotations_path: str | Path,
        images_dir: str | Path,
        class_map: dict[str, int] | None = None,
        attribute_key: str | None = None,
        default_class_id: int = 0,
    ):
        """Initialize the VIA converter.

        Args:
            annotations_path: Path to the VIA JSON export.
            images_dir: Directory containing the referenced images (used to
                read each image's width/height).
            class_map: Mapping of lowercase damage label -> class index.
            attribute_key: Which ``region_attributes`` key holds the class
                label. If ``None``, the first non-empty attribute value is used.
            default_class_id: Class index used when no label can be resolved.
        """
        super().__init__(annotations_path, class_map)
        self.images_dir = Path(images_dir)
        if not self.images_dir.is_dir():
            raise NotADirectoryError(f"Images directory not found: {self.images_dir}")
        self.attribute_key = attribute_key
        self.default_class_id = default_class_id
        self._size_cache: dict[str, tuple[int, int]] = {}

    def _load(self) -> dict[str, Any]:
        """Load the VIA JSON and normalize to a ``{key: entry}`` mapping."""
        data = json.loads(self.annotations_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "_via_img_metadata" in data:
            data = data["_via_img_metadata"]
        if not isinstance(data, dict):
            raise ValueError("Unrecognized VIA JSON structure (expected a mapping).")
        return data

    def _image_size(self, filename: str) -> tuple[int, int]:
        """Return ``(width, height)`` for an image, reading it once and caching."""
        if filename in self._size_cache:
            return self._size_cache[filename]
        image_path = (self.images_dir / filename).resolve()
        if not image_path.is_relative_to(self.images_dir.resolve()):
            raise ValueError(f"Path traversal detected: {filename}")
        try:
            with Image.open(str(image_path)) as img:
                width, height = img.size
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        except OSError:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        self._size_cache[filename] = (width, height)
        return width, height

    def _resolve_class_id(self, region_attributes: dict[str, Any]) -> int:
        """Map a region's attributes to a class index."""
        if not region_attributes:
            return self.default_class_id
        if self.attribute_key is not None:
            label = region_attributes.get(self.attribute_key)
        else:
            label = next(
                (v for v in region_attributes.values() if v not in (None, "")), None
            )
        if label is None:
            return self.default_class_id
        return self.class_map.get(str(label).lower(), self.default_class_id)

    @staticmethod
    def _shape_to_points(shape: dict[str, Any]) -> list[tuple[float, float]]:
        """Extract polygon vertices from a VIA ``shape_attributes`` object.

        Supports ``polygon`` / ``polyline`` (point lists) and ``rect`` (converted
        to its four corners). Other shapes yield no points.
        """
        name = shape.get("name")
        if name in ("polygon", "polyline"):
            xs = shape.get("all_points_x") or []
            ys = shape.get("all_points_y") or []
            return list(zip(xs, ys))
        if name == "rect":
            x, y = shape.get("x", 0), shape.get("y", 0)
            w, h = shape.get("width", 0), shape.get("height", 0)
            return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        return []

    def parse(self) -> list[ImageAnnotation]:
        """Parse the VIA export into :class:`ImageAnnotation` objects."""
        raw = self._load()

        valid_entries = []
        for entry in raw.values():
            filename = entry.get("filename")
            if not filename:
                continue
            regions = entry.get("regions", [])
            if isinstance(regions, dict):  # VIA 1.x stores regions as a dict
                regions = list(regions.values())
            if not regions:
                continue
            valid_entries.append((filename, regions))

        # Pre-fetch image sizes concurrently to minimize I/O wait
        filenames_to_load = {
            filename
            for filename, _ in valid_entries
            if filename not in self._size_cache
        }

        if filenames_to_load:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Execution raises exceptions if any file is not found
                list(executor.map(self._image_size, filenames_to_load))

        annotations: list[ImageAnnotation] = []
        for filename, regions in valid_entries:
            width, height = self._image_size(filename)
            polygons: list[Polygon] = []
            for region in regions:
                points = self._shape_to_points(region.get("shape_attributes", {}))
                if not points:
                    continue
                class_id = self._resolve_class_id(region.get("region_attributes", {}))
                polygons.append(Polygon(class_id=class_id, points=points))
            annotations.append(
                ImageAnnotation(
                    filename=filename, width=width, height=height, polygons=polygons
                )
            )
        return annotations
