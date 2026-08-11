from __future__ import annotations

import numpy as np
import pytest

from collision_vision.inference import DamageInstance
from collision_vision.visualize import MaskVisualizer, _DEFAULT_PALETTE

@pytest.fixture
def dummy_image() -> np.ndarray:
    """Return a black 100x100 BGR image."""
    return np.zeros((100, 100, 3), dtype=np.uint8)

@pytest.fixture
def valid_instance() -> DamageInstance:
    """Return a simple rectangular DamageInstance."""
    return DamageInstance(
        class_id=0,
        class_name="dent",
        confidence=0.9,
        mask=np.ones((100, 100), dtype=np.uint8),
        polygon=np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=np.int32),
        bbox=(10, 10, 90, 90),
        area_px=6400,
    )

@pytest.fixture
def valid_instance_2() -> DamageInstance:
    """Return another simple rectangular DamageInstance with class_id 1."""
    return DamageInstance(
        class_id=1,
        class_name="scratch",
        confidence=0.8,
        mask=np.ones((100, 100), dtype=np.uint8),
        polygon=np.array([[20, 20], [80, 20], [80, 80], [20, 80]], dtype=np.int32),
        bbox=(20, 20, 80, 80),
        area_px=3600,
    )


@pytest.fixture
def degenerate_instance() -> DamageInstance:
    """Return a DamageInstance with fewer than 3 vertices (invalid polygon)."""
    return DamageInstance(
        class_id=2,
        class_name="structural",
        confidence=0.7,
        mask=np.ones((100, 100), dtype=np.uint8),
        polygon=np.array([[50, 50], [60, 60]], dtype=np.int32),
        bbox=(50, 50, 60, 60),
        area_px=0,
    )


def test_visualizer_init():
    # Test default initialization
    vis = MaskVisualizer()
    assert vis.palette == _DEFAULT_PALETTE
    assert vis.alpha == 0.4

    # Test custom palette and alpha clipping
    custom_palette = ((255, 255, 255), (0, 0, 0))
    vis = MaskVisualizer(class_colors=custom_palette, alpha=1.5)
    assert vis.palette == custom_palette
    assert vis.alpha == 1.0  # Clipped to 1.0

    vis = MaskVisualizer(alpha=-0.5)
    assert vis.alpha == 0.0  # Clipped to 0.0


def test_color_for():
    custom_palette = ((255, 255, 255), (0, 0, 0))
    vis = MaskVisualizer(class_colors=custom_palette)

    assert vis.color_for(0) == (255, 255, 255)
    assert vis.color_for(1) == (0, 0, 0)
    # Test cycling
    assert vis.color_for(2) == (255, 255, 255)
    assert vis.color_for(3) == (0, 0, 0)


def test_overlay_empty_instances(dummy_image):
    vis = MaskVisualizer()
    result = vis.overlay(dummy_image, [])

    # Assert result is identical but a copy
    assert np.array_equal(result, dummy_image)
    assert result is not dummy_image


def test_overlay_valid_instances(dummy_image, valid_instance, valid_instance_2):
    vis = MaskVisualizer()
    result = vis.overlay(dummy_image, [valid_instance, valid_instance_2])

    assert result.shape == dummy_image.shape
    assert result.dtype == dummy_image.dtype
    # The image should be modified
    assert not np.array_equal(result, dummy_image)


@pytest.mark.parametrize("draw_contours", [True, False])
@pytest.mark.parametrize("draw_labels", [True, False])
def test_overlay_draw_flags(dummy_image, valid_instance, draw_contours, draw_labels):
    vis = MaskVisualizer()
    result = vis.overlay(
        dummy_image,
        [valid_instance],
        draw_contours=draw_contours,
        draw_labels=draw_labels
    )

    assert result.shape == dummy_image.shape
    assert result.dtype == dummy_image.dtype


def test_overlay_degenerate_polygon(dummy_image, degenerate_instance):
    vis = MaskVisualizer()
    # It shouldn't crash
    result = vis.overlay(dummy_image, [degenerate_instance])

    assert result.shape == dummy_image.shape
    assert result.dtype == dummy_image.dtype
