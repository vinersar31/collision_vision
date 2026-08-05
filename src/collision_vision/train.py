"""Fine-tune a YOLOv8-Seg model using the project ``config.yaml``."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .config import Config

logger = logging.getLogger(__name__)


class SegmentationTrainer:
    """Thin, config-driven wrapper around :class:`ultralytics.YOLO` training.

    ``ultralytics`` (and its ``torch`` dependency) is imported lazily so that
    importing this module stays cheap for tooling and tests.
    """

    def __init__(self, config: Config):
        """Initialize the trainer.

        Args:
            config: Parsed training configuration.
        """
        self.config = config
        self._model = None  # lazily constructed YOLO instance

    @property
    def model(self):
        """The underlying ``ultralytics.YOLO`` model (loaded on first access)."""
        if self._model is None:
            from ultralytics import YOLO

            logger.info("Loading model weights: %s", self.config.model)
            self._model = YOLO(self.config.model)
        return self._model

    def train(self, **overrides: Any):
        """Fine-tune the model.

        Args:
            **overrides: Hyperparameters that take precedence over ``config.yaml``
                (e.g. ``epochs=1`` for a smoke test).

        Returns:
            The Ultralytics training results object.
        """
        kwargs = self.config.train_kwargs(**overrides)
        logger.info("Starting training with: %s", kwargs)
        return self.model.train(**kwargs)
