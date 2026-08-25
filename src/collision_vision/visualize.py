"""Overlay segmentation masks, contours and labels onto an image."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from .inference import DamageInstance

# Distinct, high-contrast BGR colors cycled per class index.
_DEFAULT_PALETTE: tuple[tuple[int, int, int], ...] = (
    (56, 56, 255),  # red      -> class 0 (e.g. dent)
    (56, 255, 56),  # green    -> class 1 (e.g. scratch)
    (255, 128, 0),  # blue     -> class 2 (e.g. structural)
    (0, 215, 255),  # amber
    (255, 0, 255),  # magenta
    (255, 255, 0),  # cyan
)


class MaskVisualizer:
    """Render :class:`DamageInstance` masks as a translucent overlay."""

    def __init__(
        self,
        class_colors: Sequence[tuple[int, int, int]] | None = None,
        alpha: float = 0.4,
    ):
        """Initialize the visualizer.

        Args:
            class_colors: BGR colors indexed by class id. Cycles if there are
                more classes than colors. Defaults to a built-in palette.
            alpha: Opacity of the mask fill in ``[0, 1]``.
        """
        self.palette = tuple(class_colors) if class_colors else _DEFAULT_PALETTE
        self.alpha = float(np.clip(alpha, 0.0, 1.0))

    def color_for(self, class_id: int) -> tuple[int, int, int]:
        """Return the BGR color assigned to a class id."""
        return self.palette[class_id % len(self.palette)]

    def overlay(
        self,
        image: np.ndarray,
        instances: Sequence[DamageInstance],
        draw_contours: bool = True,
        draw_labels: bool = True,
    ) -> np.ndarray:
        """Return a copy of ``image`` with masks/contours/labels drawn on it.

        Args:
            image: Source BGR image.
            instances: Damage instances to render.
            draw_contours: Whether to outline each mask.
            draw_labels: Whether to draw ``"<class> <conf>"`` tags.

        Returns:
            A new BGR image with the overlay applied.
        """
        base = image.copy()
        if not instances:
            return base

        # Blend all mask fills in one pass so overlapping regions stay readable.
        fill = base.copy()
        polygons_by_class: dict[int, list[np.ndarray]] = {}
        for instance in instances:
            polygons_by_class.setdefault(instance.class_id, []).append(instance.polygon)

        for class_id, polygons in polygons_by_class.items():
            color = self.color_for(class_id)
            cv2.fillPoly(fill, polygons, color=color)
        cv2.addWeighted(fill, self.alpha, base, 1.0 - self.alpha, 0.0, dst=base)

        for instance in instances:
            color = self.color_for(instance.class_id)
            if draw_contours and len(instance.polygon) >= 3:
                cv2.polylines(
                    base, (instance.polygon,), isClosed=True, color=color, thickness=2
                )
            if draw_labels:
                self._draw_label(base, instance, color)
        return base

    @staticmethod
    def _draw_label(
        image: np.ndarray, instance: DamageInstance, color: tuple[int, int, int]
    ) -> None:
        """Draw a filled label tag at the top-left of an instance's bbox."""
        x1, y1, _, _ = instance.bbox
        text = f"{instance.class_name} {instance.confidence:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(y1, th + baseline)
        cv2.rectangle(
            image, (x1, top - th - baseline), (x1 + tw, top), color, thickness=-1
        )
        cv2.putText(
            image,
            text,
            (x1, top - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            lineType=cv2.LINE_AA,
        )
