# tests/test_audio.py
import os
from pathlib import Path
from src.audio import validate_audio_file, prepare_audio

FIXTURE = Path("tests/fixtures/hello.wav")

def test_validate_existing_file():
    assert validate_audio_file(FIXTURE) is True

def test_validate_missing_file():
    assert validate_audio_file(Path("no_such_file.wav")) is False

def test_prepare_audio_returns_path(tmp_path):
    out = prepare_audio(FIXTURE, tmp_path)
    assert out.exists()
    assert out.suffix == ".wav"
