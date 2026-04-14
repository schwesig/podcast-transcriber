import json
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class Segment:
    start: float
    end: float
    text: str

def _fmt_srt_time(seconds: float) -> str:
    ms = int((seconds % 1) * 1000)
    s = int(seconds) % 60
    m = int(seconds) // 60 % 60
    h = int(seconds) // 3600
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def write_txt(segments: list[Segment], path: Path) -> None:
    path.write_text("\n".join(seg.text.strip() for seg in segments) + "\n")

def write_json(segments: list[Segment], path: Path) -> None:
    path.write_text(json.dumps([asdict(s) for s in segments], indent=2))

def write_metadata(podcast: str, ep, path: Path) -> None:
    """Write episode metadata as JSON. ep is a feeds.Episode instance."""
    data = {
        "podcast": podcast,
        "episode_number": ep.episode_number,
        "title": ep.title,
        "date": ep.pub_date,
        "duration": ep.duration,
        "summary": ep.summary,
        "shownotes": ep.shownotes,
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def write_nfo(
    audio_path: Path,
    segments: list[Segment],
    transcription_seconds: float,
    model: str,
    path: Path,
) -> None:
    """Write transcription stats as JSON .nfo file."""
    word_count = sum(len(seg.text.split()) for seg in segments)
    audio_duration = segments[-1].end if segments else 0.0
    ratio = round(audio_duration / transcription_seconds, 2) if transcription_seconds > 0 else None
    data = {
        "audio_file_size_mb": round(audio_path.stat().st_size / 1_048_576, 2),
        "audio_duration_seconds": round(audio_duration, 1),
        "word_count": word_count,
        "transcription_seconds": round(transcription_seconds, 1),
        "realtime_ratio": ratio,
        "model": model,
    }
    path.write_text(json.dumps(data, indent=2))


def write_srt(segments: list[Segment], path: Path) -> None:
    lines = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{_fmt_srt_time(seg.start)} --> {_fmt_srt_time(seg.end)}")
        lines.append(seg.text.strip())
        lines.append("")
    path.write_text("\n".join(lines))
