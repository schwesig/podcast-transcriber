"""Tests for backend registry and mlx fallback logic."""
import platform
from unittest.mock import patch

from src.config import TranscribeConfig
from src.backend import get_transcriber
from src.backend.local import LocalWhisperTranscriber


def _cfg(**kwargs) -> TranscribeConfig:
    base = dict(model="small", device="cpu", compute_type="int8", backend="auto")
    base.update(kwargs)
    return TranscribeConfig(**base)


def test_auto_backend_non_apple_returns_local():
    """On non-Apple-Silicon, auto always resolves to faster-whisper."""
    with patch("platform.system", return_value="Linux"), \
         patch("platform.machine", return_value="x86_64"):
        transcriber = get_transcriber(_cfg(backend="auto"))
    assert isinstance(transcriber, LocalWhisperTranscriber)


def test_explicit_faster_whisper_returns_local():
    transcriber = get_transcriber(_cfg(backend="faster-whisper"))
    assert isinstance(transcriber, LocalWhisperTranscriber)


def test_unknown_backend_raises():
    with __import__("pytest").raises(ValueError, match="Unknown backend"):
        get_transcriber(_cfg(backend="typo"))


def test_mlx_not_installed_falls_back_to_local():
    """backend=mlx but mlx_whisper not installed → LocalWhisperTranscriber."""
    with patch("src.backend._mlx_available", return_value=False):
        transcriber = get_transcriber(_cfg(backend="mlx"))
    assert isinstance(transcriber, LocalWhisperTranscriber)


def test_mlx_turbo_no_variant_falls_back():
    """turbo has no mlx variant → falls back even if mlx is 'available'."""
    with patch("src.backend._mlx_available", return_value=True):
        transcriber = get_transcriber(_cfg(model="turbo", backend="mlx"))
    assert isinstance(transcriber, LocalWhisperTranscriber)


def test_mlx_distil_large_no_variant_falls_back():
    with patch("src.backend._mlx_available", return_value=True):
        transcriber = get_transcriber(_cfg(model="distil-large-v3", backend="mlx"))
    assert isinstance(transcriber, LocalWhisperTranscriber)


def test_auto_apple_silicon_mlx_available_uses_mlx():
    """On Apple Silicon with mlx installed → MlxWhisperTranscriber."""
    from src.backend.mlx_whisper import MlxWhisperTranscriber
    with patch("platform.system", return_value="Darwin"), \
         patch("platform.machine", return_value="arm64"), \
         patch("src.backend._mlx_available", return_value=True):
        transcriber = get_transcriber(_cfg(backend="auto", model="small"))
    assert isinstance(transcriber, MlxWhisperTranscriber)


def test_auto_apple_silicon_mlx_not_installed_falls_back():
    """On Apple Silicon but mlx not installed → LocalWhisperTranscriber."""
    with patch("platform.system", return_value="Darwin"), \
         patch("platform.machine", return_value="arm64"), \
         patch("src.backend._mlx_available", return_value=False):
        transcriber = get_transcriber(_cfg(backend="auto", model="small"))
    assert isinstance(transcriber, LocalWhisperTranscriber)
