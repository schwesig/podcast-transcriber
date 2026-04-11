# tests/test_output.py
import json
from pathlib import Path
from src.output import write_txt, write_json, write_srt, Segment

def _segs():
    return [
        Segment(start=0.0, end=1.5, text="Hello world"),
        Segment(start=1.5, end=3.0, text="Goodbye world"),
    ]

def test_write_txt(tmp_path):
    out = tmp_path / "out.txt"
    write_txt(_segs(), out)
    content = out.read_text()
    assert "Hello world" in content
    assert "Goodbye world" in content

def test_write_json(tmp_path):
    out = tmp_path / "out.json"
    write_json(_segs(), out)
    data = json.loads(out.read_text())
    assert len(data) == 2
    assert data[0]["text"] == "Hello world"
    assert data[0]["start"] == 0.0

def test_write_srt(tmp_path):
    out = tmp_path / "out.srt"
    write_srt(_segs(), out)
    content = out.read_text()
    assert "00:00:00,000 --> 00:00:01,500" in content
    assert "Hello world" in content
