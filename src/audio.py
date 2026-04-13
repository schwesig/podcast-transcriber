import json
import subprocess
from pathlib import Path

def validate_audio_file(path: Path) -> bool:
    return path.is_file()

def get_audio_duration(path: Path) -> float:
    """Return audio duration in seconds using ffprobe. Returns 0.0 on failure."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except Exception:
        return 0.0


def prepare_audio(input_path: Path, work_dir: Path) -> Path:
    """Convert audio to 16kHz mono WAV suitable for Whisper."""
    out = work_dir / (input_path.stem + "_16k.wav")
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(input_path),
            "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    return out
