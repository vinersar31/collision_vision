"""Tests for :class:`collision_vision.converters.base.BaseConverter`."""

import pytest

from collision_vision.converters.base import BaseConverter


def test_normalize_basic():
    """Test basic normalization of coordinates."""
    points = [(100.0, 50.0), (200.0, 150.0)]
    width = 400
    height = 300

    expected = [0.25, 1.0 / 6.0, 0.5, 0.5]
    result = BaseConverter.normalize(points, width, height)

    assert len(result) == 4
    for r, e in zip(result, expected):
        assert r == pytest.approx(e)


def test_normalize_clipping():
    """Test that coordinates outside the image bounds are clipped to [0, 1]."""
    points = [(-10.0, -20.0), (500.0, 400.0)]
    width = 400
    height = 300

    expected = [0.0, 0.0, 1.0, 1.0]
    result = BaseConverter.normalize(points, width, height)

    assert len(result) == 4
    for r, e in zip(result, expected):
        assert r == pytest.approx(e)


def test_normalize_empty():
    """Test handling of an empty list of points."""
    result = BaseConverter.normalize([], 400, 300)
    assert result == []


def test_normalize_invalid_size():
    """Test that a ValueError is raised for invalid image sizes."""
    with pytest.raises(ValueError, match="Invalid image size"):
        BaseConverter.normalize([(10.0, 10.0)], 0, 300)

    with pytest.raises(ValueError, match="Invalid image size"):
        BaseConverter.normalize([(10.0, 10.0)], 400, -10)
