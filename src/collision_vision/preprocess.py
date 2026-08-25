"""End-to-end preprocessing: annotations -> YOLO labels -> train/val -> data.yaml."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import yaml

from .converters import COCOConverter, VIAConverter
from .converters.base import BaseConverter
from .split import DatasetSplitter

logger = logging.getLogger(__name__)

# Damage classes for CollisionVision (order defines the class indices).
DEFAULT_CLASSES = ["dent", "scratch", "structural"]


class DatasetPreprocessor:
    """Convert raw annotations into a training-ready YOLOv8-Seg dataset.

    The pipeline is: convert annotations to YOLO labels, split images/labels
    into ``train``/``val``, then emit an Ultralytics ``data.yaml`` descriptor.
    """

    def __init__(self, class_names: list[str] | None = None):
        """Initialize the preprocessor.

        Args:
            class_names: Ordered class names. The index of each name becomes its
                class id. Defaults to :data:`DEFAULT_CLASSES`.
        """
        self.class_names = class_names or list(DEFAULT_CLASSES)
        self.class_map = {
            name.lower(): idx for idx, name in enumerate(self.class_names)
        }

    def _build_converter(
        self, fmt: str, annotations: str | Path, images_dir: str | Path, **kwargs
    ) -> BaseConverter:
        """Instantiate the converter for the given annotation ``fmt``."""
        fmt = fmt.lower()
        if fmt == "via":
            return VIAConverter(
                annotations_path=annotations,
                images_dir=images_dir,
                class_map=self.class_map,
                attribute_key=kwargs.get("attribute_key"),
                default_class_id=kwargs.get("default_class_id", 0),
            )
        if fmt == "coco":
            return COCOConverter(annotations_path=annotations, class_map=self.class_map)
        raise ValueError(
            f"Unsupported annotation format: {fmt!r} (expected 'via' or 'coco')"
        )

    def generate_data_yaml(self, output_dir: str | Path) -> Path:
        """Write the Ultralytics ``data.yaml`` descriptor.

        Args:
            output_dir: Dataset root containing ``images/{train,val}``.

        Returns:
            Path to the written ``data.yaml``.
        """
        output_dir = Path(output_dir).resolve()
        descriptor = {
            "path": str(output_dir),
            "train": "images/train",
            "val": "images/val",
            "nc": len(self.class_names),
            "names": {idx: name for idx, name in enumerate(self.class_names)},
        }
        data_yaml = output_dir / "data.yaml"
        data_yaml.write_text(
            yaml.safe_dump(descriptor, sort_keys=False), encoding="utf-8"
        )
        logger.info("Wrote dataset descriptor to %s", data_yaml)
        return data_yaml

    def run(
        self,
        fmt: str,
        annotations: str | Path,
        images_dir: str | Path,
        output_dir: str | Path,
        val_ratio: float = 0.2,
        seed: int = 0,
        **converter_kwargs,
    ) -> Path:
        """Run the full preprocessing pipeline.

        Args:
            fmt: Annotation format, ``"via"`` or ``"coco"``.
            annotations: Path to the annotation export.
            images_dir: Directory with the source images.
            output_dir: Destination root for the processed dataset.
            val_ratio: Fraction of samples for validation.
            seed: Random seed for the split.
            **converter_kwargs: Extra options forwarded to the converter
                (e.g. ``attribute_key`` for VIA).

        Returns:
            Path to the generated ``data.yaml``.
        """
        converter = self._build_converter(
            fmt, annotations, images_dir, **converter_kwargs
        )

        with tempfile.TemporaryDirectory() as tmp:
            labels_tmp = Path(tmp) / "labels"
            written = converter.write_labels(labels_tmp)
            if written == 0:
                raise RuntimeError("No labels were produced from the annotations.")

            DatasetSplitter(
                images_dir=images_dir,
                labels_dir=labels_tmp,
                output_dir=output_dir,
                val_ratio=val_ratio,
                seed=seed,
            ).split()

        return self.generate_data_yaml(output_dir)
