"""Tests for :class:`collision_vision.converters.VIAConverter`."""

from __future__ import annotations

import pytest

from collision_vision.converters import VIAConverter


def test_via_1x_parse(via_1x_json, images_dir, expected):
    converter = VIAConverter(
        via_1x_json, images_dir, class_map=expected["class_map"], attribute_key="damage"
    )
    annotations = converter.parse()

    assert len(annotations) == 1
    annotation = annotations[0]
    assert (annotation.width, annotation.height) == (expected["img_w"], expected["img_h"])
    assert len(annotation.polygons) == 1
    assert annotation.polygons[0].class_id == 1  # "scratch"
    assert len(annotation.polygons[0].points) == 4


def test_via_2x_parse_resolves_class(via_2x_json, images_dir, expected):
    converter = VIAConverter(
        via_2x_json, images_dir, class_map=expected["class_map"], attribute_key="damage"
    )
    annotations = converter.parse()

    assert len(annotations) == 1
    assert annotations[0].polygons[0].class_id == 0  # "dent"


def test_via_yolo_lines_are_normalized(via_1x_json, images_dir, expected):
    converter = VIAConverter(
        via_1x_json, images_dir, class_map=expected["class_map"], attribute_key="damage"
    )
    lines = converter.to_yolo_lines(converter.parse()[0])

    assert len(lines) == 1
    tokens = lines[0].split()
    assert tokens[0] == "1"  # class id
    coords = [float(t) for t in tokens[1:]]
    assert len(coords) == 8  # 4 vertices * 2
    assert all(0.0 <= c <= 1.0 for c in coords)
    assert coords[0] == pytest.approx(expected["first_xy"][0])
    assert coords[1] == pytest.approx(expected["first_xy"][1])


def test_via_write_labels(via_1x_json, images_dir, tmp_path):
    converter = VIAConverter(via_1x_json, images_dir, attribute_key="damage")
    out_dir = tmp_path / "labels"

    written = converter.write_labels(out_dir)

    assert written == 1
    label_file = out_dir / "car.txt"
    assert label_file.is_file()
    assert label_file.read_text(encoding="utf-8").strip()


def test_via_unresolved_attribute_uses_default_class(via_1x_json, images_dir, expected):
    converter = VIAConverter(
        via_1x_json,
        images_dir,
        class_map=expected["class_map"],
        attribute_key="does_not_exist",
        default_class_id=2,
    )
    annotations = converter.parse()

    assert annotations[0].polygons[0].class_id == 2


def test_via_missing_image_raises(via_1x_json, tmp_path):
    empty_dir = tmp_path / "empty_images"
    empty_dir.mkdir()
    converter = VIAConverter(via_1x_json, empty_dir, attribute_key="damage")

    with pytest.raises(FileNotFoundError):
        converter.parse()
