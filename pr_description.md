🎯 **What:** The testing gap in `src/collision_vision/converters/base.py` for the `BaseConverter.normalize` method has been addressed by adding a new test file `tests/test_base_converter.py`.

📊 **Coverage:** The new tests cover:
- Basic normalization of coordinates to the `[0, 1]` range.
- Clipping of coordinates that fall outside the image boundaries.
- Handling of an empty input list of points.
- Raising a `ValueError` for invalid image sizes (width or height <= 0).

✨ **Result:** Test coverage for `BaseConverter` has improved, and the `normalize` method is now fully tested for various scenarios, making the codebase more reliable.
