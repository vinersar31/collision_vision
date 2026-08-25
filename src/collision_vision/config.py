"""Typed loader for the training ``config.yaml`` file.

The config file mixes two concerns:

* ``model`` — the checkpoint passed to :class:`ultralytics.YOLO`.
* every other key — hyperparameters forwarded verbatim to ``YOLO.train(...)``.

:class:`Config` keeps these separate so the trainer can wire them up cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_MODEL = "yolov8s-seg.pt"
DEFAULT_DATA = "data/processed/data.yaml"


@dataclass
class Config:
    """Training configuration loaded from ``config.yaml``.

    Attributes:
        model: Pre-trained checkpoint to fine-tune (e.g. ``yolov8s-seg.pt``).
        data: Path to the Ultralytics dataset descriptor (``data.yaml``).
        params: Remaining hyperparameters forwarded to ``YOLO.train(...)``.
    """

    model: str = DEFAULT_MODEL
    data: str = DEFAULT_DATA
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load a :class:`Config` from a YAML file."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Config root must be a mapping, got {raw.__class__.__name__}")
        raw = dict(raw)  # shallow copy — we mutate via pop
        model = raw.pop("model", DEFAULT_MODEL)
        data = raw.pop("data", DEFAULT_DATA)
        # Drop empty values so Ultralytics falls back to its own defaults.
        params = {k: v for k, v in raw.items() if v is not None}
        return cls(model=model, data=data, params=params)

    def train_kwargs(self, **overrides: Any) -> dict[str, Any]:
        """Build the keyword arguments for ``YOLO.train(...)``.

        Args:
            **overrides: Values that take precedence over the file (e.g. a CLI
                ``--epochs`` flag). ``None`` values are ignored.

        Returns:
            A dict including ``data`` and all configured hyperparameters.
        """
        kwargs: dict[str, Any] = {"data": self.data, **self.params}
        kwargs.update({k: v for k, v in overrides.items() if v is not None})
        return kwargs
