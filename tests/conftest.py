"""Shared pytest fixtures: synthetic VIA / COCO annotations and a dummy image."""

import json

import cv2
import numpy as np
import pytest

# Known image geometry and class layout used across the converter tests.
IMG_W, IMG_H = 200, 100
IMAGE_NAME = "car.jpg"
CLASS_MAP = {"dent": 0, "scratch": 1, "structural": 2}

# A rectangle-shaped polygon (4 vertices) in absolute pixel coordinates.
POLY_X = [20, 60, 60, 20]
POLY_Y = [10, 10, 40, 40]


@pytest.fixture
def expected() -> dict:
    """Metadata the tests assert against (keeps values in one place)."""
    return {
        "img_w": IMG_W,
        "img_h": IMG_H,
        "image_name": IMAGE_NAME,
        "class_map": dict(CLASS_MAP),
        # First vertex (20, 10) normalized by (200, 100).
        "first_xy": (POLY_X[0] / IMG_W, POLY_Y[0] / IMG_H),
    }


@pytest.fixture
def images_dir(tmp_path):
    """A directory containing one dummy image of known dimensions."""
    directory = tmp_path / "images"
    directory.mkdir()
    image = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    cv2.imwrite(str(directory / IMAGE_NAME), image)
    return directory


@pytest.fixture
def via_1x_json(tmp_path):
    """VIA 1.x export (``via_region_data.json``) with regions as a dict."""
    data = {
        "car.jpg123": {
            "filename": IMAGE_NAME,
            "size": 123,
            "regions": {
                "0": {
                    "shape_attributes": {
                        "name": "polygon",
                        "all_points_x": POLY_X,
                        "all_points_y": POLY_Y,
                    },
                    "region_attributes": {"damage": "scratch"},
                }
            },
            "file_attributes": {},
        }
    }
    path = tmp_path / "via_region_data.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def via_2x_json(tmp_path):
    """VIA 2.x project export (``_via_img_metadata``) with regions as a list."""
    data = {
        "_via_img_metadata": {
            "car.jpg123": {
                "filename": IMAGE_NAME,
                "size": 123,
                "regions": [
                    {
                        "shape_attributes": {
                            "name": "polygon",
                            "all_points_x": POLY_X,
                            "all_points_y": POLY_Y,
                        },
                        "region_attributes": {"damage": "dent"},
                    }
                ],
            }
        }
    }
    path = tmp_path / "via_project.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def coco_json(tmp_path):
    """Standard COCO polygon-segmentation export."""
    data = {
        "images": [{"id": 1, "file_name": IMAGE_NAME, "width": IMG_W, "height": IMG_H}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 2,  # -> "scratch"
                "segmentation": [
                    [
                        POLY_X[0],
                        POLY_Y[0],
                        POLY_X[1],
                        POLY_Y[1],
                        POLY_X[2],
                        POLY_Y[2],
                        POLY_X[3],
                        POLY_Y[3],
                    ]
                ],
                "iscrowd": 0,
            }
        ],
        "categories": [
            {"id": 1, "name": "dent"},
            {"id": 2, "name": "scratch"},
            {"id": 3, "name": "structural"},
        ],
    }
    path = tmp_path / "coco.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def coco_rle_json(tmp_path):
    """COCO export whose annotation uses an (unsupported) RLE mask."""
    data = {
        "images": [{"id": 1, "file_name": IMAGE_NAME, "width": IMG_W, "height": IMG_H}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "segmentation": {"counts": "abc123", "size": [IMG_H, IMG_W]},
                "iscrowd": 1,
            }
        ],
        "categories": [{"id": 1, "name": "dent"}],
    }
    path = tmp_path / "coco_rle.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path
