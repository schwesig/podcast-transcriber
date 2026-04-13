from pathlib import Path
from unittest.mock import patch, MagicMock
from src.downloader import download_audio


def test_download_creates_file(tmp_path):
    dest = tmp_path / "audio.mp3"
    mock_response = MagicMock()
    mock_response.headers = {"content-length": "6"}
    mock_response.iter_content = lambda chunk_size: [b"hello\n"]
    mock_response.__enter__ = lambda s: s
    mock_response.__exit__ = MagicMock(return_value=False)

    with patch("src.downloader.requests.get", return_value=mock_response):
        download_audio("https://example.com/ep.mp3", dest)

    assert dest.exists()
    assert dest.read_bytes() == b"hello\n"


def test_download_skips_if_exists(tmp_path):
    dest = tmp_path / "audio.mp3"
    dest.write_bytes(b"existing")

    with patch("src.downloader.requests.get") as mock_get:
        download_audio("https://example.com/ep.mp3", dest)
        mock_get.assert_not_called()
