import pytest

from collision_vision.split import DatasetSplitter


@pytest.fixture
def mock_dataset(tmp_path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    output_dir = tmp_path / "output"
    images_dir.mkdir()
    labels_dir.mkdir()

    # Create 4 valid pairs
    for i in range(1, 5):
        (images_dir / f"img{i}.jpg").touch()
        (labels_dir / f"img{i}.txt").touch()

    # Unmatched image
    (images_dir / "no_label.jpg").touch()

    # Unmatched label
    (labels_dir / "no_image.txt").touch()

    # Invalid extension
    (images_dir / "img5.gif").touch()
    (labels_dir / "img5.txt").touch()

    return images_dir, labels_dir, output_dir


def test_dataset_splitter_copy(mock_dataset):
    images_dir, labels_dir, output_dir = mock_dataset
    splitter = DatasetSplitter(
        images_dir, labels_dir, output_dir, val_ratio=0.5, move=False
    )
    counts = splitter.split()

    assert counts == {"train": 2, "val": 2}

    # Assert source files remain
    assert len(list(images_dir.iterdir())) == 6  # 4 valid, 1 no_label, 1 .gif
    assert (
        len(list(labels_dir.iterdir())) == 6
    )  # 4 valid, 1 no_image, 1 for .gif (img5.txt)

    # Assert output files exist
    for split in ["train", "val"]:
        assert len(list((output_dir / "images" / split).iterdir())) == 2
        assert len(list((output_dir / "labels" / split).iterdir())) == 2


def test_dataset_splitter_move(mock_dataset):
    images_dir, labels_dir, output_dir = mock_dataset
    splitter = DatasetSplitter(
        images_dir, labels_dir, output_dir, val_ratio=0.5, move=True
    )
    counts = splitter.split()

    assert counts == {"train": 2, "val": 2}

    # Assert valid source files were moved
    assert len(list(images_dir.iterdir())) == 2  # 1 no_label, 1 .gif
    assert len(list(labels_dir.iterdir())) == 2  # 1 no_image, 1 for .gif

    # Assert output files exist
    for split in ["train", "val"]:
        assert len(list((output_dir / "images" / split).iterdir())) == 2
        assert len(list((output_dir / "labels" / split).iterdir())) == 2


def test_dataset_splitter_invalid_val_ratio(tmp_path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    output_dir = tmp_path / "output"

    with pytest.raises(ValueError, match="val_ratio must be in \\[0, 1\\)"):
        DatasetSplitter(images_dir, labels_dir, output_dir, val_ratio=-0.1)

    with pytest.raises(ValueError, match="val_ratio must be in \\[0, 1\\)"):
        DatasetSplitter(images_dir, labels_dir, output_dir, val_ratio=1.0)

    with pytest.raises(ValueError, match="val_ratio must be in \\[0, 1\\)"):
        DatasetSplitter(images_dir, labels_dir, output_dir, val_ratio=1.5)


def test_dataset_splitter_no_pairs(tmp_path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    output_dir = tmp_path / "output"
    images_dir.mkdir()
    labels_dir.mkdir()

    splitter = DatasetSplitter(images_dir, labels_dir, output_dir)

    with pytest.raises(RuntimeError, match="No image/label pairs found"):
        splitter.split()
