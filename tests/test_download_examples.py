import pytest
from pathlib import Path
import sys
import os

# Add scripts directory to path to import download_examples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.download_examples import download

def test_download_rejects_invalid_scheme(tmp_path: Path, capsys: pytest.CaptureFixture):
    """Test that download() rejects URLs with schemes other than http/https."""
    dest = tmp_path / "test.jpg"

    # file:// scheme should be rejected
    result = download("file:///etc/passwd", dest)

    assert result is False
    assert not dest.exists()

    # Check that appropriate message was printed
    captured = capsys.readouterr()
    assert "! rejected invalid scheme 'file' for URL file:///etc/passwd" in captured.out

    # ftp:// scheme should be rejected
    result = download("ftp://example.com/file.jpg", dest)
    assert result is False

    # Test valid scheme format (but doesn't actually connect to avoid network dependency)
    # We'll just mock urlopen to return immediately
    from unittest.mock import patch, MagicMock

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"fake image data"
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        result = download("https://example.com/image.jpg", dest)

        assert result is True
        assert dest.exists()
        assert dest.read_bytes() == b"fake image data"
