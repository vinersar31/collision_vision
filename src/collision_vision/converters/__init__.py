"""Annotation converters that turn labeling-tool exports into YOLOv8-Seg labels."""

from .base import BaseConverter, ImageAnnotation, Polygon
from .coco_converter import COCOConverter
from .via_converter import VIAConverter

__all__ = [
    "BaseConverter",
    "ImageAnnotation",
    "Polygon",
    "COCOConverter",
    "VIAConverter",
]
