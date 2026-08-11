"""Split a flat image/label collection into YOLO ``train`` / ``val`` folders."""

from __future__ import annotations

import logging
import random
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# Image extensions considered when pairing images with label files.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class DatasetSplitter:
    """Pair images with YOLO label files and split them into train/val sets.

    The resulting layout is the standard Ultralytics structure::

        <output_dir>/images/train  <output_dir>/labels/train
        <output_dir>/images/val    <output_dir>/labels/val
    """

    def __init__(
        self,
        images_dir: str | Path,
        labels_dir: str | Path,
        output_dir: str | Path,
        val_ratio: float = 0.2,
        seed: int = 0,
        move: bool = False,
    ):
        """Initialize the splitter.

        Args:
            images_dir: Directory containing the source images.
            labels_dir: Directory containing the YOLO ``.txt`` label files.
            output_dir: Destination root for the split dataset.
            val_ratio: Fraction of samples assigned to the validation set.
            seed: Random seed for the shuffle (reproducible splits).
            move: If ``True`` move files instead of copying them.
        """
        if not 0.0 <= val_ratio < 1.0:
            raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.output_dir = Path(output_dir)
        self.val_ratio = val_ratio
        self.seed = seed
        self._transfer = shutil.move if move else shutil.copy2

    def _paired_samples(self) -> list[tuple[Path, Path]]:
        """Return ``(image_path, label_path)`` pairs that both exist."""
        pairs: list[tuple[Path, Path]] = []
        for image_path in sorted(self.images_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label_path = self.labels_dir / f"{image_path.stem}.txt"
            if label_path.is_file():
                pairs.append((image_path, label_path))
            else:
                logger.warning("No label for image %s; skipping", image_path.name)
        return pairs

    def split(self) -> dict[str, int]:
        """Perform the split.

        Returns:
            A mapping ``{"train": n_train, "val": n_val}``.
        """
        pairs = self._paired_samples()
        if not pairs:
            raise RuntimeError(
                f"No image/label pairs found between {self.images_dir} and {self.labels_dir}"
            )

        random.Random(self.seed).shuffle(pairs)
        n_val = int(len(pairs) * self.val_ratio)
        splits = {"val": pairs[:n_val], "train": pairs[n_val:]}

        for split_name, split_pairs in splits.items():
            image_out = self.output_dir / "images" / split_name
            label_out = self.output_dir / "labels" / split_name
            image_out.mkdir(parents=True, exist_ok=True)
            label_out.mkdir(parents=True, exist_ok=True)
            for image_path, label_path in split_pairs:
                self._transfer(str(image_path), str(image_out / image_path.name))
                self._transfer(str(label_path), str(label_out / label_path.name))

        counts = {name: len(pairs_) for name, pairs_ in splits.items()}
        logger.info("Split dataset: %d train / %d val", counts["train"], counts["val"])
        return counts
