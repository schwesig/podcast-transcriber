import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

from src.audio import prepare_audio, get_audio_duration
from src.backend.local import LocalWhisperTranscriber
from src.config import TranscribeConfig
from src.output import RichSegment
from src.pipeline.config import PipelineConfig
from src.pipeline.scorer import score_segment
from src.pipeline import output as pipeline_output


def _make_transcriber(model: str, cfg: PipelineConfig) -> LocalWhisperTranscriber:
    tc = TranscribeConfig(
        model=model,
        device=cfg.device,
        compute_type=cfg.compute_type,
        language=cfg.language,
        model_cache_dir=cfg.model_cache_dir,
    )
    return LocalWhisperTranscriber(tc)


def _transcribe_file(
    audio_wav: Path,
    model: str,
    cfg: PipelineConfig,
) -> list[RichSegment]:
    transcriber = _make_transcriber(model, cfg)
    return transcriber.transcribe_rich(
        audio_wav,
        beam_size=cfg.beam_size,
        vad_filter=cfg.vad,
        word_timestamps=cfg.word_timestamps,
        language=cfg.language,
    )


def _slice_audio(source: Path, start: float, end: float, dest: Path) -> None:
    """Extract audio slice [start, end] seconds into dest using ffmpeg."""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-to", str(end),
            "-i", str(source),
            str(dest),
        ],
        check=True,
        capture_output=True,
    )


def _retranscribe_segment(
    seg: RichSegment,
    wav_source: Path,
    model: str,
    cfg: PipelineConfig,
) -> RichSegment:
    """Re-transcribe a single segment audio slice with a different model."""
    with tempfile.TemporaryDirectory() as tmp:
        slice_path = Path(tmp) / "slice.wav"
        _slice_audio(wav_source, seg.start, seg.end, slice_path)
        transcriber = _make_transcriber(model, cfg)
        new_segs = transcriber.transcribe_rich(
            slice_path,
            beam_size=cfg.beam_size,
            vad_filter=False,  # VAD not useful for short slices
            word_timestamps=cfg.word_timestamps,
            language=cfg.language,
        )
    if not new_segs:
        return seg  # no speech detected in slice — keep original
    new_text = " ".join(s.text.strip() for s in new_segs)
    return replace(
        seg,
        text=new_text,
        original_text=seg.text,
        model_used=model,
    )


def _print_dry_run_report(segments: list[RichSegment]) -> None:
    green = sum(1 for s in segments if s.difficulty == "green")
    yellow = sum(1 for s in segments if s.difficulty == "yellow")
    red = sum(1 for s in segments if s.difficulty == "red")
    print(f"\n  Dry-run scoring report: {len(segments)} segments")
    print(f"    green={green}  yellow={yellow}  red={red}")
    for seg in segments:
        if seg.difficulty != "green":
            flags = ", ".join(seg.reason_flags) or "—"
            print(f"    [{seg.difficulty:6}] {seg.start:6.1f}s  {seg.text[:60]!r}  flags={flags}")


def run_pipeline(input_path: Path, cfg: PipelineConfig) -> None:
    """Run the full multi-pass transcription pipeline on a single audio file."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    out_dir = Path(cfg.output_dir) if cfg.output_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    print(f"\n>>> {input_path.name}")
    print(f"  Stage 1: Preprocessing")
    with tempfile.TemporaryDirectory() as tmp:
        wav = prepare_audio(input_path, Path(tmp))

        print(f"  Stage 2: First pass ({cfg.first_pass_model})")
        segments = _transcribe_file(wav, cfg.first_pass_model, cfg)

        print(f"  Stage 3: Scoring {len(segments)} segments")
        segments = [score_segment(s, cfg) for s in segments]

        if cfg.dry_run:
            _print_dry_run_report(segments)
            return

        yellow = [s for s in segments if s.difficulty == "yellow"]
        red = [s for s in segments if s.difficulty == "red"]

        if yellow:
            print(f"  Stage 4a: Re-transcribing {len(yellow)} yellow segments ({cfg.yellow_pass_model})")
            improved = {(s.start, s.end): _retranscribe_segment(s, wav, cfg.yellow_pass_model, cfg) for s in yellow}
            segments = [improved.get((s.start, s.end), s) for s in segments]

        if red and cfg.enable_large_pass:
            print(f"  Stage 4b: Re-transcribing {len(red)} red segments ({cfg.red_pass_model})")
            improved = {(s.start, s.end): _retranscribe_segment(s, wav, cfg.red_pass_model, cfg) for s in red}
            segments = [improved.get((s.start, s.end), s) for s in segments]
        elif red and not cfg.enable_large_pass:
            print(f"  Stage 4b: {len(red)} red segments — re-running with {cfg.yellow_pass_model} (large pass disabled)")
            improved = {(s.start, s.end): _retranscribe_segment(s, wav, cfg.yellow_pass_model, cfg) for s in red}
            segments = [improved.get((s.start, s.end), s) for s in segments]

        segments.sort(key=lambda s: s.start)

    print(f"  Stage 5: Writing output")
    pipeline_output.write_txt(segments, out_dir / f"{stem}.txt")
    pipeline_output.write_json(segments, out_dir / f"{stem}.json")
    if cfg.export_srt:
        pipeline_output.write_srt(segments, out_dir / f"{stem}.srt")
    if cfg.export_vtt:
        pipeline_output.write_vtt(segments, out_dir / f"{stem}.vtt")

    print(f"  -> {out_dir / f'{stem}.txt'}")
    print(f"  -> {out_dir / f'{stem}.json'}")
    if cfg.export_srt:
        print(f"  -> {out_dir / f'{stem}.srt'}")
    if cfg.export_vtt:
        print(f"  -> {out_dir / f'{stem}.vtt'}")


def run_pipeline_folder(folder: Path, cfg: PipelineConfig) -> None:
    """Run pipeline on all audio files in a folder."""
    audio_extensions = {".mp3", ".m4a", ".wav", ".flac", ".ogg"}
    files = [f for f in sorted(folder.iterdir()) if f.suffix.lower() in audio_extensions]
    if not files:
        print(f"No audio files found in {folder}")
        return
    print(f"Found {len(files)} audio file(s) in {folder}")
    for f in files:
        file_out_dir = Path(cfg.output_dir) / f.stem if cfg.output_dir else f.parent / f.stem
        file_cfg = replace(cfg, output_dir=str(file_out_dir))
        run_pipeline(f, file_cfg)
