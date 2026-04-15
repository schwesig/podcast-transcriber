#!/usr/bin/env python3
"""
Usage:
  transcribe_podcast <input_file_or_folder> [options]
"""
import argparse
import sys
from pathlib import Path

from src.pipeline.config import PipelineConfig
from src.pipeline.stages import run_pipeline, run_pipeline_folder


def build_config(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        language=args.language,
        enable_large_pass=args.enable_large_pass,
        beam_size=args.beam_size,
        vad=not args.no_vad,
        word_timestamps=args.word_timestamps,
        export_srt=args.export_srt,
        export_vtt=args.export_vtt,
        device=args.device,
        compute_type=args.compute_type,
        dry_run=args.dry_run,
        verbose=args.verbose,
        model_cache_dir=args.model_cache_dir,
        output_dir=args.output_dir,
        yellow_logprob=args.yellow_logprob,
        red_logprob=args.red_logprob,
        yellow_no_speech=args.yellow_no_speech,
        red_no_speech=args.red_no_speech,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe podcast audio using a staged multi-pass pipeline."
    )
    parser.add_argument("input", help="Audio file or folder of audio files")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: alongside input)")
    parser.add_argument("--language", default=None, help="Language code e.g. de, en (default: auto-detect)")
    parser.add_argument("--enable-large-pass", action="store_true", help="Enable red -> large-v3 re-transcription")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size (default: 5)")
    parser.add_argument("--no-vad", action="store_true", help="Disable VAD filter (default: VAD on)")
    parser.add_argument("--word-timestamps", action="store_true", help="Enable word-level timestamps in JSON")
    parser.add_argument("--export-srt", action="store_true", help="Write .srt subtitle file")
    parser.add_argument("--export-vtt", action="store_true", help="Write .vtt subtitle file")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device (default: auto)")
    parser.add_argument("--compute-type", default="int8", help="Compute type (default: int8)")
    parser.add_argument("--dry-run", action="store_true", help="Score only, print report, no re-transcription")
    parser.add_argument("--verbose", action="store_true", help="Print per-segment detail")
    parser.add_argument("--model-cache-dir", default=".models", help="Model cache directory (default: .models)")
    parser.add_argument("--yellow-logprob", type=float, default=-0.6, help="Yellow logprob threshold (default: -0.6)")
    parser.add_argument("--red-logprob", type=float, default=-1.0, help="Red logprob threshold (default: -1.0)")
    parser.add_argument("--yellow-no-speech", type=float, default=0.3, help="Yellow no-speech threshold (default: 0.3)")
    parser.add_argument("--red-no-speech", type=float, default=0.6, help="Red no-speech threshold (default: 0.6)")

    args = parser.parse_args()
    cfg = build_config(args)
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"ERROR: {input_path} does not exist", file=sys.stderr)
        return 1

    if input_path.is_dir():
        run_pipeline_folder(input_path, cfg)
    else:
        run_pipeline(input_path, cfg)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
