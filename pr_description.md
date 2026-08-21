🧹 Clean up unused imports across codebase

🎯 **What:** Removed unused imports (specifically `from pathlib import Path` in `tests/test_split.py` and the `from __future__ import annotations` across the codebase which was unused for the target python runtime and marked as unused across multiple files).
💡 **Why:** Reduces clutter, fixes unused imports, and improves code hygiene.
✅ **Verification:** Verified with `ruff`, `flake8` and ran `make test` to ensure functionality remains unchanged. Formatted code with `ruff format` and `black`.
✨ **Result:** A cleaner codebase with no unused imports.
