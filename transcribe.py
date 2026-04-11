#!/usr/bin/env python3
"""
Usage:
  transcribe.py <audio_file> [options]

Options:
  --model       Model name: tiny, base, small, medium, large-v3  (default: base)
  --device      cpu | cuda | auto  (default: auto)
  --compute     int8 | float16 | float32  (default: int8)
  --language    ISO 639-1 code, e.g. en, de  (default: auto-detect)
  --formats     Comma-separated: txt,json,srt  (default: txt)
  --output-dir  Directory for output files  (default: same dir as input)
"""
import argparse
import sys
import tempfile
from pathlib import Path

from src.audio import validate_audio_file, prepare_audio
from src.backend import get_transcriber
from src.config import TranscribeConfig
from src.output import write_txt, write_json, write_srt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transcribe audio with local Whisper")
    p.add_argument("audio_file", help="Path to audio file")
    p.add_argument("--model", default="base", help="Whisper model (default: base)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                   help="Compute device (default: auto)")
    p.add_argument("--compute", default="int8",
                   choices=["int8", "float16", "float32"],
                   help="Compute type (default: int8)")
    p.add_argument("--language", default=None, help="Language code (default: auto)")
    p.add_argument("--formats", default="txt",
                   help="Output formats: txt,json,srt (default: txt)")
    p.add_argument("--output-dir", default=None, help="Output directory (default: input dir)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    audio_path = Path(args.audio_file)

    if not validate_audio_file(audio_path):
        print(f"ERROR: File not found: {audio_path}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else audio_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    formats = [f.strip() for f in args.formats.split(",")]
    cfg = TranscribeConfig(
        model=args.model,
        device=args.device,
        compute_type=args.compute,
        language=args.language,
        output_formats=formats,
    )

    print(f"Model: {cfg.model} | Device: {cfg.device} | Compute: {cfg.compute_type}")
    print(f"Transcribing: {audio_path}")

    with tempfile.TemporaryDirectory() as tmp:
        wav = prepare_audio(audio_path, Path(tmp))
        transcriber = get_transcriber(cfg)
        segments = transcriber.transcribe(wav)

    stem = audio_path.stem
    writers = {"txt": write_txt, "json": write_json, "srt": write_srt}
    for fmt in formats:
        if fmt not in writers:
            print(f"WARNING: unknown format '{fmt}', skipping", file=sys.stderr)
            continue
        out = output_dir / f"{stem}.{fmt}"
        writers[fmt](segments, out)
        print(f"  -> {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
