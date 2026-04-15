import json
from dataclasses import asdict
from pathlib import Path

from src.output import RichSegment


def _fmt_srt_time(seconds: float) -> str:
    ms = int((seconds % 1) * 1000)
    s = int(seconds) % 60
    m = int(seconds) // 60 % 60
    h = int(seconds) // 3600
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def _fmt_vtt_time(seconds: float) -> str:
    ms = int((seconds % 1) * 1000)
    s = int(seconds) % 60
    m = int(seconds) // 60 % 60
    h = int(seconds) // 3600
    return f"{h:02}:{m:02}:{s:02}.{ms:03}"


def write_txt(segments: list[RichSegment], path: Path) -> None:
    path.write_text("\n".join(seg.text.strip() for seg in segments) + "\n")


def write_json(segments: list[RichSegment], path: Path) -> None:
    path.write_text(json.dumps([asdict(s) for s in segments], indent=2, ensure_ascii=False))


def write_srt(segments: list[RichSegment], path: Path) -> None:
    lines = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{_fmt_srt_time(seg.start)} --> {_fmt_srt_time(seg.end)}")
        lines.append(seg.text.strip())
        lines.append("")
    path.write_text("\n".join(lines))


def write_vtt(segments: list[RichSegment], path: Path) -> None:
    lines = ["WEBVTT", ""]
    for seg in segments:
        lines.append(f"{_fmt_vtt_time(seg.start)} --> {_fmt_vtt_time(seg.end)}")
        lines.append(seg.text.strip())
        lines.append("")
    path.write_text("\n".join(lines))
