"""Convert COCO instance-segmentation JSON to YOLOv8-Seg labels.

Handles the standard COCO layout with ``images``, ``annotations`` and
``categories`` arrays. Only **polygon** segmentations are supported; run-length
encoded (RLE) masks are detected and skipped with a warning so that no heavy
``pycocotools`` dependency is required.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from .base import BaseConverter, ImageAnnotation, Polygon

logger = logging.getLogger(__name__)


class COCOConverter(BaseConverter):
    """Converter for COCO polygon segmentation annotations."""

    def __init__(
        self,
        annotations_path: str | Path,
        class_map: dict[str, int] | None = None,
    ):
        """Initialize the COCO converter.

        Args:
            annotations_path: Path to the COCO ``instances`` JSON file.
            class_map: Optional mapping of lowercase category name -> class
                index. Categories absent from the map are indexed by their order
                of appearance in the ``categories`` array.
        """
        super().__init__(annotations_path, class_map)

    def _build_category_map(self, categories: list[dict[str, Any]]) -> dict[int, int]:
        """Map COCO ``category_id`` values to zero-based class indices."""
        mapping: dict[int, int] = {}
        for idx, category in enumerate(categories):
            name = str(category.get("name", "")).lower()
            mapping[category["id"]] = self.class_map.get(name, idx)
        return mapping

    def parse(self) -> list[ImageAnnotation]:
        """Parse the COCO export into :class:`ImageAnnotation` objects."""
        data = json.loads(self.annotations_path.read_text(encoding="utf-8"))
        images = data.get("images", [])
        categories = data.get("categories", [])
        category_to_class = self._build_category_map(categories)

        annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in data.get("annotations", []):
            annotations_by_image[annotation["image_id"]].append(annotation)

        results: list[ImageAnnotation] = []
        for image in images:
            width, height = image.get("width"), image.get("height")
            filename = image.get("file_name")
            if not filename or not width or not height:
                logger.warning(
                    "Skipping image with missing metadata: %s", image.get("id")
                )
                continue

            polygons: list[Polygon] = []
            for annotation in annotations_by_image.get(image["id"], []):
                segmentation = annotation.get("segmentation")
                if isinstance(segmentation, dict):  # RLE mask
                    logger.warning(
                        "RLE segmentation is not supported; skipping annotation %s in %s",
                        annotation.get("id"),
                        filename,
                    )
                    continue
                if not segmentation:
                    continue
                class_id = category_to_class.get(annotation["category_id"], 0)
                # Each element is one polygon part: a flat [x1, y1, x2, y2, ...] list.
                for part in segmentation:
                    it = iter(part)
                    points = list(zip(it, it))
                    if points:
                        polygons.append(Polygon(class_id=class_id, points=points))

            results.append(
                ImageAnnotation(
                    filename=filename, width=width, height=height, polygons=polygons
                )
            )
        return results
