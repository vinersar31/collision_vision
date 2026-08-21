import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from collision_vision.inference import DamageInstance, DamageSegmenter


def test_damage_instance_area_fraction():
    instance = DamageInstance(
        class_id=0,
        class_name="dent",
        confidence=0.9,
        mask=np.zeros((10, 10)),
        polygon=np.array([[0, 0], [0, 5], [5, 5], [5, 0]]),
        bbox=(0, 0, 5, 5),
        area_px=25,
    )
    # image shape (10, 10) -> area 100. fraction = 25/100 = 0.25
    assert instance.area_fraction((10, 10)) == 0.25

    # test area fraction for zero total area
    assert instance.area_fraction((0, 0)) == 0.0


def test_damage_instance_zero_area():
    instance = DamageInstance(
        class_id=0,
        class_name="dent",
        confidence=0.9,
        mask=np.zeros((10, 10)),
        polygon=np.array([[0, 0], [0, 5], [5, 5], [5, 0]]),
        bbox=(0, 0, 5, 5),
        area_px=0,
    )
    assert instance.area_fraction((10, 10)) == 0.0


@patch("ultralytics.YOLO")
def test_segmenter_predict_empty_masks(mock_yolo_class):
    mock_model = MagicMock()
    mock_model.names = {0: "dent"}
    mock_yolo_class.return_value = mock_model

    mock_result = MagicMock()
    mock_result.masks = None
    mock_model.predict.return_value = [mock_result]

    segmenter = DamageSegmenter("dummy.pt")
    assert segmenter.class_names == {0: "dent"}

    image = np.zeros((10, 10, 3), dtype=np.uint8)
    instances = segmenter.predict(image)

    assert len(instances) == 0
    mock_model.predict.assert_called_once_with(
        source=image,
        conf=0.25,
        iou=0.7,
        device=None,
        verbose=False,
    )


@patch("ultralytics.YOLO")
def test_segmenter_predict_valid_polygons(mock_yolo_class):
    mock_model = MagicMock()
    mock_model.names = {0: "dent", 1: "scratch"}
    mock_yolo_class.return_value = mock_model

    mock_result = MagicMock()
    mock_result.orig_shape = (100, 200)

    mock_masks = MagicMock()
    mock_masks.xy = [np.array([[10.0, 10.0], [10.0, 20.0], [20.0, 20.0], [20.0, 10.0]])]
    mock_result.masks = mock_masks

    mock_boxes = MagicMock()

    mock_cls_tensor = MagicMock()
    mock_cls_tensor.cpu.return_value.numpy.return_value = np.array([1.0])
    mock_boxes.cls = mock_cls_tensor

    mock_conf_tensor = MagicMock()
    mock_conf_tensor.cpu.return_value.numpy.return_value = np.array([0.95])
    mock_boxes.conf = mock_conf_tensor

    mock_xyxy_tensor = MagicMock()
    mock_xyxy_tensor.cpu.return_value.numpy.return_value = np.array(
        [[10.1, 10.1, 20.2, 20.2]]
    )
    mock_boxes.xyxy = mock_xyxy_tensor

    mock_result.boxes = mock_boxes

    mock_model.predict.return_value = [mock_result]

    segmenter = DamageSegmenter("dummy.pt")
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    instances = segmenter.predict(image, conf=0.5)

    assert len(instances) == 1
    inst = instances[0]
    assert inst.class_id == 1
    assert inst.class_name == "scratch"
    assert inst.confidence == pytest.approx(0.95)
    assert inst.bbox == (10, 10, 20, 20)
    assert inst.area_px == 100
    assert inst.mask.shape == (100, 200)
    assert inst.mask.dtype == np.uint8
    np.testing.assert_array_equal(
        inst.polygon, np.array([[10, 10], [10, 20], [20, 20], [20, 10]], dtype=np.int32)
    )

    mock_model.predict.assert_called_once_with(
        source=image,
        conf=0.5,
        iou=0.7,
        device=None,
        verbose=False,
    )


@patch("ultralytics.YOLO")
def test_segmenter_predict_invalid_polygons(mock_yolo_class):
    mock_model = MagicMock()
    mock_model.names = {0: "dent"}
    mock_yolo_class.return_value = mock_model

    mock_result = MagicMock()
    mock_result.orig_shape = (100, 200)

    mock_masks = MagicMock()
    mock_masks.xy = [
        np.array([[10.0, 10.0], [20.0, 20.0]]),  # < 3 points
        None,  # Invalid
    ]
    mock_result.masks = mock_masks

    mock_boxes = MagicMock()

    mock_cls_tensor = MagicMock()
    mock_cls_tensor.cpu.return_value.numpy.return_value = np.array([0, 0])
    mock_boxes.cls = mock_cls_tensor

    mock_conf_tensor = MagicMock()
    mock_conf_tensor.cpu.return_value.numpy.return_value = np.array([0.9, 0.8])
    mock_boxes.conf = mock_conf_tensor

    mock_xyxy_tensor = MagicMock()
    mock_xyxy_tensor.cpu.return_value.numpy.return_value = np.array(
        [[10, 10, 20, 20], [0, 0, 5, 5]]
    )
    mock_boxes.xyxy = mock_xyxy_tensor

    mock_result.boxes = mock_boxes

    mock_model.predict.return_value = [mock_result]

    segmenter = DamageSegmenter("dummy.pt")
    instances = segmenter.predict("dummy.jpg")

    assert len(instances) == 0


@patch("ultralytics.YOLO")
def test_segmenter_predict_fallback_class_name(mock_yolo_class):
    mock_model = MagicMock()
    mock_model.names = {0: "dent"}
    mock_yolo_class.return_value = mock_model

    mock_result = MagicMock()
    mock_result.orig_shape = (100, 200)

    mock_masks = MagicMock()
    mock_masks.xy = [np.array([[10.0, 10.0], [10.0, 20.0], [20.0, 20.0]])]
    mock_result.masks = mock_masks

    mock_boxes = MagicMock()

    mock_cls_tensor = MagicMock()
    mock_cls_tensor.cpu.return_value.numpy.return_value = np.array([99.0])
    mock_boxes.cls = mock_cls_tensor

    mock_conf_tensor = MagicMock()
    mock_conf_tensor.cpu.return_value.numpy.return_value = np.array([0.99])
    mock_boxes.conf = mock_conf_tensor

    mock_xyxy_tensor = MagicMock()
    mock_xyxy_tensor.cpu.return_value.numpy.return_value = np.array(
        [[10.1, 10.1, 20.2, 20.2]]
    )
    mock_boxes.xyxy = mock_xyxy_tensor

    mock_result.boxes = mock_boxes

    mock_model.predict.return_value = [mock_result]

    segmenter = DamageSegmenter("dummy.pt")
    instances = segmenter.predict("dummy.jpg")

    assert len(instances) == 1
    assert instances[0].class_id == 99
    assert instances[0].class_name == "99"
