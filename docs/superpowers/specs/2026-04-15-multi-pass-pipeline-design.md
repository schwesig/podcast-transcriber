# Multi-Pass Transcription Pipeline — Design Spec

**Date:** 2026-04-15
**Status:** Approved

## Overview

A staged transcription pipeline that runs a cheap first pass (distil-large-v3) and only escalates difficult segments to more expensive models (turbo, large-v3). Implemented as an extension to the existing `src/backend/` and exposed via a new `transcribe_podcast` CLI, while also being usable from `podcast_sync.py`.

**Core principle:** Not every segment deserves the most expensive model. Only the difficult parts should.

---

## 1. Module Structure

```
src/
  pipeline/
    __init__.py          # exports run_pipeline()
    config.py            # PipelineConfig dataclass
    stages.py            # orchestrates all 5 stages
    scorer.py            # difficulty classification per segment
    output.py            # write txt/json/srt/vtt from RichSegment list
  backend/
    local.py             # unchanged — single-model transcriber
    __init__.py          # unchanged — get_transcriber factory

transcribe_podcast.py    # CLI entrypoint → installed as console script
podcast_sync.py          # unchanged — can construct PipelineConfig directly later
pyproject.toml           # adds console_scripts entry point
```

### RichSegment dataclass

Extends the existing `Segment` concept with provenance metadata:

```python
@dataclass
class RichSegment:
    start: float
    end: float
    text: str
    model_used: str           # e.g. "distil-large-v3", "turbo", "large-v3"
    difficulty: str           # "green" | "yellow" | "red"
    reason_flags: list[str]   # e.g. ["low_logprob", "repetition"]
    original_text: str | None # text from first pass if replaced
    avg_logprob: float | None
    no_speech_prob: float | None
    compression_ratio: float | None
```

---

## 2. Difficulty Scoring

`scorer.py` classifies each segment into green / yellow / red.

### Signal sources (from faster-whisper per segment)

| Signal | What it means |
|---|---|
| `avg_logprob` | Log-probability average — lower = less confident |
| `no_speech_prob` | Probability this segment is silence/noise |
| `compression_ratio` | Outliers indicate repetition or hallucination |

### Text heuristics

- Repeated n-grams (≥3 consecutive identical words → yellow, ≥5 → red)
- Very short segments (< 2 words)
- Very long segments (> 60 words)
- Unusual punctuation density (`???`, `...` spam, `!!!!!`)
- Known hallucination phrases: `"Thanks for watching"`, `"Subscribe"`, `"Vielen Dank fürs Zuschauen"`, etc.

### Default thresholds (balanced, all configurable via CLI)

| Signal | Yellow | Red |
|---|---|---|
| `avg_logprob` | < -0.6 | < -1.0 |
| `no_speech_prob` | > 0.3 | > 0.6 |
| `compression_ratio` | > 2.0 or < 0.8 | > 2.5 or < 0.5 |
| Repeated words | ≥ 3 consecutive | ≥ 5 consecutive |

**Rules:**
- Red if **any** red threshold is hit
- Yellow if any yellow threshold is hit (but no red)
- Multiple flags accumulate in `reason_flags`

---

## 3. Pipeline Stages

### Stage 1: Preprocessing
- Validate file exists and is readable
- Convert to 16kHz mono WAV via existing `prepare_audio()` in `src/audio.py`
- No changes to `src/audio.py`

### Stage 2: First pass — distil-large-v3
- `LocalWhisperTranscriber` with `word_timestamps=True`, `vad_filter=True`
- Collect `RichSegment` list with `avg_logprob`, `no_speech_prob`, `compression_ratio` from faster-whisper segment metadata
- faster-whisper exposes these natively per segment

### Stage 3: Difficulty scoring
- `scorer.py` classifies each segment → green / yellow / red
- If `--dry-run`: print classification report, stop here

### Stage 4: Re-transcription (targeted)
- Yellow → re-run audio slice with **turbo** (always)
- Red → re-run audio slice with **large-v3** (only if `--enable-large-pass`)
- Audio slice extracted via ffmpeg trim (not full file)
- Replace `text`, store `original_text`, update `model_used`
- If `--enable-large-pass` is off, red segments fall back to turbo

### Stage 5: Assembly + output
- Merge all `RichSegment`s sorted by `start`
- Write `transcript.txt` — plain text
- Write `transcript.json` — full RichSegment list
- Optionally write `transcript.srt` (`--export-srt`)
- Optionally write `transcript.vtt` (`--export-vtt`)
- Output dir: `--output-dir`, default = same directory as input file

**Folder input:** iterate all `.mp3`, `.m4a`, `.wav`, `.flac` files; process each independently; output into `--output-dir/<stem>/` or alongside input.

---

## 4. CLI

Installed as `transcribe_podcast` via `pyproject.toml` console script.

```
transcribe_podcast <input>              # file or folder

--output-dir PATH                       # default: alongside input
--language CODE                         # e.g. de, en (default: auto-detect)
--enable-large-pass                     # enable red → large-v3 (default: off)
--beam-size INT                         # default: 5
--vad / --no-vad                        # VAD filter (default: on)
--word-timestamps                       # word-level timestamps in JSON
--export-srt                            # write .srt
--export-vtt                            # write .vtt
--device cpu|cuda                       # default: auto
--compute-type int8|float16|...         # default: int8
--dry-run                               # score only, print report, no re-transcription
--verbose                               # print per-segment detail
--model-cache-dir PATH                  # default: .models

# Threshold overrides
--yellow-logprob FLOAT                  # default: -0.6
--red-logprob FLOAT                     # default: -1.0
--yellow-no-speech FLOAT                # default: 0.3
--red-no-speech FLOAT                   # default: 0.6
```

### PipelineConfig

`src/pipeline/config.py` holds all settings as a dataclass. The CLI parses args and builds the config. `podcast_sync.py` can construct `PipelineConfig` directly without going through the CLI.

---

## 5. pyproject.toml

New file (replaces or supplements `requirements.txt`) adding:

```toml
[project.scripts]
transcribe_podcast = "transcribe_podcast:main"
```

Existing `requirements.txt` stays for backwards compatibility.

---

## 6. Heuristic Tuning Guide

All thresholds are in `PipelineConfig` and overridable via CLI flags. To tune:

1. Run with `--dry-run --verbose` on a known-good episode to see segment classifications
2. If too many yellow/red: raise `--yellow-logprob` closer to 0, raise `--yellow-no-speech`
3. If quality still poor after re-transcription: lower thresholds, add phrases to the hallucination list in `scorer.py`
4. The `reason_flags` in `transcript.json` show exactly why each segment was escalated

The hallucination phrase list in `scorer.py` is a plain Python list — easy to extend per language or podcast type.

---

## 7. Files to Create

| File | Purpose |
|---|---|
| `src/pipeline/__init__.py` | Exports `run_pipeline()` |
| `src/pipeline/config.py` | `PipelineConfig` dataclass |
| `src/pipeline/stages.py` | Stage orchestration |
| `src/pipeline/scorer.py` | Difficulty classification |
| `src/pipeline/output.py` | Write txt/json/srt/vtt |
| `transcribe_podcast.py` | CLI entrypoint |
| `pyproject.toml` | Console script + metadata |

## 8. Files to Modify

| File | Change |
|---|---|
| `requirements.txt` | Add no new deps (distil-large-v3, turbo, large-v3 all via faster-whisper) |
| `src/backend/local.py` | Expose `avg_logprob`, `no_speech_prob`, `compression_ratio` per segment |
