"""Tests for the HTTP Whisper backend."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import TranscribeConfig
from src.backend import get_transcriber
from src.backend.http_whisper import HttpWhisperTranscriber


def _cfg(**kwargs) -> TranscribeConfig:
    base = dict(model="small", device="cpu", compute_type="int8", backend="http",
                whisper_server_url="http://test-server:8080")
    base.update(kwargs)
    return TranscribeConfig(**base)


def test_registry_returns_http_transcriber():
    t = get_transcriber(_cfg())
    assert isinstance(t, HttpWhisperTranscriber)


def test_engine_name():
    t = HttpWhisperTranscriber(_cfg())
    assert t.engine_name == "http-whisper"


def test_server_url_from_config():
    t = HttpWhisperTranscriber(_cfg(whisper_server_url="http://myserver:9000"))
    assert t._url == "http://myserver:9000"


def test_server_url_from_env(monkeypatch):
    monkeypatch.setenv("WHISPER_SERVER_URL", "http://envserver:1234")
    t = HttpWhisperTranscriber(_cfg(whisper_server_url=""))
    assert t._url == "http://envserver:1234"


def test_no_url_raises():
    with pytest.raises(ValueError, match="WHISPER_SERVER_URL"):
        HttpWhisperTranscriber(_cfg(whisper_server_url=""))


def test_transcribe_verbose_json(tmp_path):
    audio = tmp_path / "ep.mp3"
    audio.write_bytes(b"fake")

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "segments": [
            {"start": 0.0, "end": 5.0, "text": "Hello world"},
            {"start": 5.0, "end": 10.0, "text": "Second segment"},
        ]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.post", return_value=mock_response), \
         patch("src.backend.http_whisper.get_audio_duration", return_value=10.0):
        t = HttpWhisperTranscriber(_cfg())
        segments = t.transcribe(audio)

    assert len(segments) == 2
    assert segments[0].start == 0.0
    assert segments[0].end == 5.0
    assert segments[0].text == "Hello world"
    assert segments[1].text == "Second segment"


def test_transcribe_fallback_plain_text(tmp_path):
    audio = tmp_path / "ep.mp3"
    audio.write_bytes(b"fake")

    mock_response = MagicMock()
    mock_response.json.return_value = {"text": "Plain text fallback"}
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.post", return_value=mock_response), \
         patch("src.backend.http_whisper.get_audio_duration", return_value=5.0):
        t = HttpWhisperTranscriber(_cfg())
        segments = t.transcribe(audio)

    assert len(segments) == 1
    assert segments[0].text == "Plain text fallback"
    assert segments[0].start == 0.0
    assert segments[0].end == 5.0


def test_transcribe_raises_on_http_error(tmp_path):
    import httpx
    audio = tmp_path / "ep.mp3"
    audio.write_bytes(b"fake")

    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "500", request=MagicMock(), response=MagicMock()
    )

    with patch("httpx.post", return_value=mock_response), \
         patch("src.backend.http_whisper.get_audio_duration", return_value=5.0):
        t = HttpWhisperTranscriber(_cfg())
        with pytest.raises(httpx.HTTPStatusError):
            t.transcribe(audio)


def test_feeds_txt_backend_http_parsed(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parent.parent))
    feedparser_mock = MagicMock()
    import sys
    sys.modules.setdefault("feedparser", feedparser_mock)

    from src.feeds import parse_feeds_file
    feeds_file = tmp_path / "feeds.txt"
    feeds_file.write_text(
        "https://example.com/feed.rss backend=http "
        "whisper_server_url=http://100.111.245.24:8080 language=de\n"
    )
    configs = parse_feeds_file(feeds_file)
    assert len(configs) == 1
    assert configs[0].backend == "http"
    assert configs[0].whisper_server_url == "http://100.111.245.24:8080"
    assert configs[0].language == "de"
