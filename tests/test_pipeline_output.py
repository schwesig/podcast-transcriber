import json
from pathlib import Path
from src.output import RichSegment
from src.pipeline.output import write_txt, write_json, write_srt, write_vtt


def _segs() -> list[RichSegment]:
    return [
        RichSegment(
            start=0.0, end=1.5, text="Hello world",
            model_used="distil-large-v3", difficulty="green", reason_flags=[],
            original_text=None, avg_logprob=-0.3, no_speech_prob=0.1, compression_ratio=1.2,
        ),
        RichSegment(
            start=1.5, end=3.0, text="Goodbye world",
            model_used="turbo", difficulty="yellow", reason_flags=["low_logprob"],
            original_text="Goodbye wörld", avg_logprob=-0.7, no_speech_prob=0.1, compression_ratio=1.1,
        ),
    ]


def test_write_txt(tmp_path):
    out = tmp_path / "transcript.txt"
    write_txt(_segs(), out)
    content = out.read_text()
    assert "Hello world" in content
    assert "Goodbye world" in content


def test_write_json_structure(tmp_path):
    out = tmp_path / "transcript.json"
    write_json(_segs(), out)
    data = json.loads(out.read_text())
    assert len(data) == 2
    assert data[0]["text"] == "Hello world"
    assert data[0]["model_used"] == "distil-large-v3"
    assert data[0]["difficulty"] == "green"
    assert data[0]["reason_flags"] == []
    assert data[0]["original_text"] is None
    assert data[1]["model_used"] == "turbo"
    assert data[1]["original_text"] == "Goodbye wörld"


def test_write_srt(tmp_path):
    out = tmp_path / "transcript.srt"
    write_srt(_segs(), out)
    content = out.read_text()
    assert "00:00:00,000 --> 00:00:01,500" in content
    assert "Hello world" in content
    assert "Goodbye world" in content


def test_write_vtt(tmp_path):
    out = tmp_path / "transcript.vtt"
    write_vtt(_segs(), out)
    content = out.read_text()
    assert content.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:01.500" in content
    assert "Hello world" in content
