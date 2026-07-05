"""CollisionVision — car-damage instance segmentation with YOLOv8-Seg.

Only lightweight modules are eagerly imported here; the trainer and inference
engine pull in ``ultralytics``/``torch`` lazily to keep imports cheap.
"""

from __future__ import annotations

from .config import Config
from .converters import BaseConverter, COCOConverter, VIAConverter
from .preprocess import DEFAULT_CLASSES, DatasetPreprocessor
from .split import DatasetSplitter

__version__ = "0.1.0"

__all__ = [
    "Config",
    "BaseConverter",
    "COCOConverter",
    "VIAConverter",
    "DatasetPreprocessor",
    "DatasetSplitter",
    "DEFAULT_CLASSES",
    "__version__",
]
