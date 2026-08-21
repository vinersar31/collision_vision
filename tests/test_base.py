"""Tests for :class:`collision_vision.converters.BaseConverter`."""

from __future__ import annotations

import pytest

from collision_vision.converters.base import BaseConverter


def test_normalize_valid_points():
    points = [(10.0, 20.0), (30.0, 40.0)]
    width = 100
    height = 200
    expected = [10.0 / 100, 20.0 / 200, 30.0 / 100, 40.0 / 200]
    result = BaseConverter.normalize(points, width, height)
    assert result == expected


def test_normalize_clips_out_of_bounds():
    points = [(-10.0, 250.0), (150.0, -20.0)]
    width = 100
    height = 200
    expected = [0.0, 1.0, 1.0, 0.0]
    result = BaseConverter.normalize(points, width, height)
    assert result == expected


def test_normalize_invalid_size():
    points = [(10.0, 20.0)]
    with pytest.raises(ValueError, match="Invalid image size: 0x200"):
        BaseConverter.normalize(points, 0, 200)

    with pytest.raises(ValueError, match="Invalid image size: 100x-5"):
        BaseConverter.normalize(points, 100, -5)
