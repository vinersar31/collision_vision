from __future__ import annotations

import logging
from pathlib import Path

import pytest

from collision_vision.split import DatasetSplitter


@pytest.fixture
def dummy_dataset(tmp_path: Path) -> dict[str, Path]:
    """Create a temporary dataset with 10 image/label pairs."""
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    output_dir = tmp_path / "output"

    images_dir.mkdir()
    labels_dir.mkdir()
    output_dir.mkdir()

    # Create 10 dummy image/label pairs
    for i in range(10):
        img = images_dir / f"img_{i}.jpg"
        lbl = labels_dir / f"img_{i}.txt"
        img.write_text("dummy image content")
        lbl.write_text("dummy label content")

    return {
        "images_dir": images_dir,
        "labels_dir": labels_dir,
        "output_dir": output_dir,
    }


def test_invalid_val_ratio(dummy_dataset):
    """Test that ValueError is raised for invalid val_ratio."""
    with pytest.raises(ValueError, match="val_ratio must be in"):
        DatasetSplitter(**dummy_dataset, val_ratio=-0.1)

    with pytest.raises(ValueError, match="val_ratio must be in"):
        DatasetSplitter(**dummy_dataset, val_ratio=1.0)

    with pytest.raises(ValueError, match="val_ratio must be in"):
        DatasetSplitter(**dummy_dataset, val_ratio=1.5)


def test_empty_dataset(tmp_path: Path):
    """Test that RuntimeError is raised when no pairs are found."""
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    output_dir = tmp_path / "output"

    images_dir.mkdir()
    labels_dir.mkdir()
    output_dir.mkdir()

    splitter = DatasetSplitter(
        images_dir=images_dir,
        labels_dir=labels_dir,
        output_dir=output_dir,
    )

    with pytest.raises(RuntimeError, match="No image/label pairs found"):
        splitter.split()


def test_split_copy_default(dummy_dataset):
    """Test successful copy split (default)."""
    splitter = DatasetSplitter(**dummy_dataset, val_ratio=0.2)
    counts = splitter.split()

    assert counts == {"train": 8, "val": 2}

    # Check that original files are still there
    assert len(list(dummy_dataset["images_dir"].iterdir())) == 10
    assert len(list(dummy_dataset["labels_dir"].iterdir())) == 10

    # Check that split files were created
    train_images = list((dummy_dataset["output_dir"] / "images" / "train").iterdir())
    train_labels = list((dummy_dataset["output_dir"] / "labels" / "train").iterdir())
    val_images = list((dummy_dataset["output_dir"] / "images" / "val").iterdir())
    val_labels = list((dummy_dataset["output_dir"] / "labels" / "val").iterdir())

    assert len(train_images) == 8
    assert len(train_labels) == 8
    assert len(val_images) == 2
    assert len(val_labels) == 2


def test_split_move(dummy_dataset):
    """Test successful move split."""
    splitter = DatasetSplitter(**dummy_dataset, val_ratio=0.3, move=True)
    counts = splitter.split()

    assert counts == {"train": 7, "val": 3}

    # Check that original files were removed
    assert len(list(dummy_dataset["images_dir"].iterdir())) == 0
    assert len(list(dummy_dataset["labels_dir"].iterdir())) == 0

    # Check that split files were moved to output
    train_images = list((dummy_dataset["output_dir"] / "images" / "train").iterdir())
    train_labels = list((dummy_dataset["output_dir"] / "labels" / "train").iterdir())
    val_images = list((dummy_dataset["output_dir"] / "images" / "val").iterdir())
    val_labels = list((dummy_dataset["output_dir"] / "labels" / "val").iterdir())

    assert len(train_images) == 7
    assert len(train_labels) == 7
    assert len(val_images) == 3
    assert len(val_labels) == 3


def test_missing_label_logged_and_skipped(dummy_dataset, caplog):
    """Test that missing label results in a warning and the image is skipped."""
    # Remove one label
    (dummy_dataset["labels_dir"] / "img_0.txt").unlink()

    splitter = DatasetSplitter(**dummy_dataset, val_ratio=0.2)
    with caplog.at_level(logging.WARNING):
        counts = splitter.split()

    assert "No label for image img_0.jpg; skipping" in caplog.text
    # 9 pairs left (val_ratio 0.2 means 1 val, 8 train)
    assert counts == {"train": 8, "val": 1}


def test_seed_reproducibility(dummy_dataset):
    """Test that two splitters with same seed produce identical splits."""
    import shutil

    # Backup original dummy dataset to apply to the second splitter
    dummy_backup = {k: dummy_dataset[k].parent / f"{k}_backup" for k in dummy_dataset}
    for k in dummy_dataset:
        shutil.copytree(dummy_dataset[k], dummy_backup[k])

    splitter_1 = DatasetSplitter(**dummy_dataset, val_ratio=0.2, seed=42)
    counts_1 = splitter_1.split()
    train_1 = {
        p.name for p in (dummy_dataset["output_dir"] / "images" / "train").iterdir()
    }

    splitter_2 = DatasetSplitter(
        images_dir=dummy_backup["images_dir"],
        labels_dir=dummy_backup["labels_dir"],
        output_dir=dummy_backup["output_dir"],
        val_ratio=0.2,
        seed=42,
    )
    counts_2 = splitter_2.split()
    train_2 = {
        p.name for p in (dummy_backup["output_dir"] / "images" / "train").iterdir()
    }

    assert counts_1 == counts_2
    assert train_1 == train_2
