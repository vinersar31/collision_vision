"""Run YOLOv8-Seg inference and extract clean, per-instance damage masks."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# An input image may be a path or an already-decoded BGR array.
ImageInput = Union[str, Path, np.ndarray]


@dataclass
class DamageInstance:
    """A single detected damage region with its extracted mask pixels.

    Attributes:
        class_id: Zero-based class index.
        class_name: Human-readable class name.
        confidence: Detection confidence in ``[0, 1]``.
        mask: Binary mask (``uint8`` ``{0, 1}``) at the original image resolution.
        polygon: ``(N, 2)`` int array of the mask contour in pixel coordinates.
        bbox: Bounding box as ``(x1, y1, x2, y2)`` in pixels.
        area_px: Number of pixels covered by the mask.
    """

    class_id: int
    class_name: str
    confidence: float
    mask: np.ndarray
    polygon: np.ndarray
    bbox: tuple[int, int, int, int]
    area_px: int

    def area_fraction(self, image_shape: tuple[int, int]) -> float:
        """Return the mask area as a fraction of the whole image."""
        height, width = image_shape[:2]
        total = height * width
        return self.area_px / total if total else 0.0


class DamageSegmenter:
    """Load a trained YOLOv8-Seg model and produce :class:`DamageInstance` masks."""

    def __init__(
        self,
        weights: str | Path,
        conf: float = 0.25,
        iou: float = 0.7,
        device: str | int | None = None,
    ):
        """Initialize the segmenter.

        Args:
            weights: Path to trained ``.pt`` weights (or a model name to fetch).
            conf: Default confidence threshold.
            iou: IoU threshold for non-maximum suppression.
            device: Inference device (``None`` = auto, e.g. ``0``, ``"cpu"``).
        """
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.conf = conf
        self.iou = iou
        self.device = device
        # {class_id: name} provided by the trained model.
        self.class_names: dict[int, str] = dict(self.model.names)

    def predict(self, image: ImageInput, conf: float | None = None) -> list[DamageInstance]:
        """Run segmentation on a single image.

        Args:
            image: Image path or a BGR ``numpy`` array.
            conf: Optional per-call confidence override.

        Returns:
            A list of detected :class:`DamageInstance` objects (possibly empty).
        """
        results = self.model.predict(
            source=image,
            conf=self.conf if conf is None else conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )
        result = results[0]
        if result.masks is None:
            return []

        height, width = result.orig_shape
        # `masks.xy` are contours already scaled to the original image size,
        # which avoids letterbox/resize alignment issues.
        polygons = result.masks.xy
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)

        instances: list[DamageInstance] = []
        for i, polygon in enumerate(polygons):
            if polygon is None or len(polygon) < 3:
                continue
            polygon_int = polygon.astype(np.int32)
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [polygon_int], color=1)

            class_id = int(classes[i])
            instances.append(
                DamageInstance(
                    class_id=class_id,
                    class_name=self.class_names.get(class_id, str(class_id)),
                    confidence=float(confidences[i]),
                    mask=mask,
                    polygon=polygon_int,
                    bbox=tuple(int(v) for v in boxes[i]),
                    area_px=int(mask.sum()),
                )
            )
        return instances
