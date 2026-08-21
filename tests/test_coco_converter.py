"""Tests for :class:`collision_vision.converters.COCOConverter`."""

import pytest

from collision_vision.converters import COCOConverter


def test_coco_parse(coco_json, expected):
    converter = COCOConverter(coco_json, class_map=expected["class_map"])
    annotations = converter.parse()

    assert len(annotations) == 1
    annotation = annotations[0]
    assert (annotation.width, annotation.height) == (
        expected["img_w"],
        expected["img_h"],
    )
    assert len(annotation.polygons) == 1
    assert annotation.polygons[0].class_id == 1  # category 2 -> "scratch" -> class 1
    assert len(annotation.polygons[0].points) == 4


def test_coco_yolo_lines_are_normalized(coco_json, expected):
    converter = COCOConverter(coco_json, class_map=expected["class_map"])
    lines = converter.to_yolo_lines(converter.parse()[0])

    assert len(lines) == 1
    tokens = lines[0].split()
    assert tokens[0] == "1"
    coords = [float(t) for t in tokens[1:]]
    assert len(coords) == 8
    assert all(0.0 <= c <= 1.0 for c in coords)
    assert coords[0] == pytest.approx(expected["first_xy"][0])
    assert coords[1] == pytest.approx(expected["first_xy"][1])


def test_coco_category_indexed_by_order_without_map(coco_json):
    converter = COCOConverter(coco_json)  # no class_map -> index by appearance
    annotations = converter.parse()

    # category_id 2 is the second category -> index 1.
    assert annotations[0].polygons[0].class_id == 1


def test_coco_rle_segmentation_is_skipped(coco_rle_json):
    converter = COCOConverter(coco_rle_json, class_map={"dent": 0})
    annotations = converter.parse()

    assert len(annotations) == 1
    assert annotations[0].polygons == []


def test_coco_write_labels(coco_json, tmp_path):
    converter = COCOConverter(coco_json)
    out_dir = tmp_path / "labels"

    written = converter.write_labels(out_dir)

    assert written == 1
    assert (out_dir / "car.txt").is_file()
