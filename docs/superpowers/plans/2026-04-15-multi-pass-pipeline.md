# Multi-Pass Transcription Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a staged transcription pipeline that runs distil-large-v3 as a cheap first pass and re-transcribes only difficult segments with turbo or large-v3.

**Architecture:** New `src/pipeline/` package owns difficulty scoring, multi-pass orchestration, and output. `src/backend/local.py` is extended to expose per-segment metadata (logprob, no_speech_prob, compression_ratio). A new `transcribe_podcast.py` CLI entrypoint is installed as a console script via `pyproject.toml`.

**Tech Stack:** Python 3.11+, faster-whisper, ffmpeg (subprocess), argparse, dataclasses, pytest

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/pipeline/__init__.py` | Create | Export `run_pipeline()` |
| `src/pipeline/config.py` | Create | `PipelineConfig` dataclass with all thresholds/flags |
| `src/pipeline/scorer.py` | Create | Classify segments: green / yellow / red |
| `src/pipeline/stages.py` | Create | Orchestrate all 5 pipeline stages |
| `src/pipeline/output.py` | Create | Write txt/json/srt/vtt from `RichSegment` list |
| `src/backend/local.py` | Modify | Return `RichSegment` with metadata from faster-whisper |
| `transcribe_podcast.py` | Create | CLI entrypoint: parse args → build config → call `run_pipeline()` |
| `pyproject.toml` | Create | Console script entry point + project metadata |
| `tests/test_scorer.py` | Create | Unit tests for difficulty classifier |
| `tests/test_pipeline_output.py` | Create | Unit tests for pipeline output writers |
| `tests/test_stages.py` | Create | Integration tests for stage orchestration |

---

## Task 1: Extend `src/backend/local.py` to expose per-segment metadata

faster-whisper's segment objects expose `avg_logprob`, `no_speech_prob`, and `compression_ratio`. Currently these are discarded. We need them for scoring.

**Files:**
- Modify: `src/backend/local.py`
- Modify: `src/output.py` — add `RichSegment` dataclass
- Modify: `tests/test_transcriber.py` — update assertion

- [ ] **Step 1: Add `RichSegment` to `src/output.py`**

Open `src/output.py` and add after the existing `Segment` dataclass (after line 9):

```python
@dataclass
class RichSegment:
    start: float
    end: float
    text: str
    model_used: str
    difficulty: str                    # "green" | "yellow" | "red"
    reason_flags: list[str]
    original_text: str | None
    avg_logprob: float | None
    no_speech_prob: float | None
    compression_ratio: float | None
```

- [ ] **Step 2: Write failing test for updated transcriber**

Add to `tests/test_transcriber.py`:

```python
from src.output import RichSegment

def test_local_transcriber_returns_rich_segments():
    cfg = TranscribeConfig(model="tiny", device="cpu", compute_type="int8")
    transcriber = get_transcriber(cfg)
    segments = transcriber.transcribe_rich(FIXTURE)
    assert isinstance(segments, list)
    for seg in segments:
        assert isinstance(seg, RichSegment)
        assert seg.model_used == "tiny"
        assert seg.difficulty == "green"
        assert seg.reason_flags == []
        assert seg.original_text is None
```

- [ ] **Step 3: Run test to verify it fails**

```bash
cd /home/tschwesi/claude/podCastBib && source .venv/bin/activate && pytest tests/test_transcriber.py::test_local_transcriber_returns_rich_segments -v
```

Expected: `FAILED` — `AttributeError: 'LocalWhisperTranscriber' object has no attribute 'transcribe_rich'`

- [ ] **Step 4: Implement `transcribe_rich` in `src/backend/local.py`**

Add the method to `LocalWhisperTranscriber` (keep existing `transcribe` unchanged):

```python
from src.output import Segment, RichSegment

def transcribe_rich(
    self,
    audio_path: Path,
    *,
    beam_size: int = 5,
    vad_filter: bool = True,
    word_timestamps: bool = False,
    language: str | None = None,
) -> list[RichSegment]:
    lang = language or self._language
    segments_iter, _ = self._model.transcribe(
        str(audio_path),
        language=lang,
        beam_size=beam_size,
        vad_filter=vad_filter,
        word_timestamps=word_timestamps,
    )
    duration = get_audio_duration(audio_path)
    results = []
    for s in segments_iter:
        results.append(RichSegment(
            start=s.start,
            end=s.end,
            text=s.text,
            model_used=self._model_name,
            difficulty="green",
            reason_flags=[],
            original_text=None,
            avg_logprob=s.avg_logprob,
            no_speech_prob=s.no_speech_prob,
            compression_ratio=s.compression_ratio,
        ))
        if duration > 0:
            pct = min(int(s.end / duration * 100), 100)
            print(f"\r  Transcribing... {pct}%", end="", flush=True)
    if duration > 0:
        print(f"\r  Transcribing... 100%")
    return results
```

Also store the model name in `__init__` so it's accessible:

```python
def __init__(self, config: TranscribeConfig):
    device = config.device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    self._model = WhisperModel(
        config.model,
        device=device,
        compute_type=config.compute_type,
        download_root=config.model_cache_dir,
    )
    self._language = config.language
    self._model_name = config.model
```

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest tests/test_transcriber.py::test_local_transcriber_returns_rich_segments -v
```

Expected: `PASSED`

- [ ] **Step 6: Run full test suite to confirm no regressions**

```bash
pytest tests/ -v
```

Expected: all existing tests still pass.

- [ ] **Step 7: Commit**

```bash
git add src/output.py src/backend/local.py tests/test_transcriber.py
git commit -m "feat: add RichSegment and transcribe_rich to LocalWhisperTranscriber"
```

---

## Task 2: `src/pipeline/config.py` — PipelineConfig dataclass

**Files:**
- Create: `src/pipeline/__init__.py`
- Create: `src/pipeline/config.py`

- [ ] **Step 1: Create `src/pipeline/__init__.py`** (empty for now)

```python
# src/pipeline/__init__.py
from src.pipeline.stages import run_pipeline

__all__ = ["run_pipeline"]
```

Wait — `stages.py` doesn't exist yet. Write it as a stub so the import doesn't fail:

```python
# src/pipeline/__init__.py
```

Leave it empty. We'll add the export in Task 5.

- [ ] **Step 2: Write failing test for PipelineConfig**

Create `tests/test_pipeline_config.py`:

```python
from src.pipeline.config import PipelineConfig

def test_pipeline_config_defaults():
    cfg = PipelineConfig()
    assert cfg.first_pass_model == "distil-large-v3"
    assert cfg.yellow_pass_model == "turbo"
    assert cfg.red_pass_model == "large-v3"
    assert cfg.yellow_logprob == -0.6
    assert cfg.red_logprob == -1.0
    assert cfg.yellow_no_speech == 0.3
    assert cfg.red_no_speech == 0.6
    assert cfg.enable_large_pass is False
    assert cfg.vad is True
    assert cfg.beam_size == 5
    assert cfg.export_srt is False
    assert cfg.export_vtt is False
    assert cfg.dry_run is False
    assert cfg.verbose is False
    assert cfg.device == "auto"
    assert cfg.compute_type == "int8"
    assert cfg.model_cache_dir == ".models"
    assert cfg.language is None
    assert cfg.word_timestamps is False
```

- [ ] **Step 3: Run test to verify it fails**

```bash
pytest tests/test_pipeline_config.py -v
```

Expected: `FAILED` — `ModuleNotFoundError: No module named 'src.pipeline'`

- [ ] **Step 4: Create `src/pipeline/config.py`**

```python
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PipelineConfig:
    # Models
    first_pass_model: str = "distil-large-v3"
    yellow_pass_model: str = "turbo"
    red_pass_model: str = "large-v3"

    # Scoring thresholds
    yellow_logprob: float = -0.6
    red_logprob: float = -1.0
    yellow_no_speech: float = 0.3
    red_no_speech: float = 0.6
    yellow_compression_high: float = 2.0
    red_compression_high: float = 2.5
    yellow_compression_low: float = 0.8
    red_compression_low: float = 0.5
    yellow_repeat_words: int = 3
    red_repeat_words: int = 5

    # Pipeline flags
    enable_large_pass: bool = False
    vad: bool = True
    beam_size: int = 5
    word_timestamps: bool = False
    export_srt: bool = False
    export_vtt: bool = False
    dry_run: bool = False
    verbose: bool = False

    # Hardware
    device: str = "auto"
    compute_type: str = "int8"
    model_cache_dir: str = ".models"

    # Language
    language: Optional[str] = None

    # Output
    output_dir: Optional[str] = None
```

Also create `src/pipeline/__init__.py` (empty):

```python
```

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest tests/test_pipeline_config.py -v
```

Expected: `PASSED`

- [ ] **Step 6: Commit**

```bash
git add src/pipeline/__init__.py src/pipeline/config.py tests/test_pipeline_config.py
git commit -m "feat: add PipelineConfig dataclass"
```

---

## Task 3: `src/pipeline/scorer.py` — difficulty classification

**Files:**
- Create: `src/pipeline/scorer.py`
- Create: `tests/test_scorer.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_scorer.py`:

```python
import pytest
from src.output import RichSegment
from src.pipeline.config import PipelineConfig
from src.pipeline.scorer import score_segment, HALLUCINATION_PHRASES


def _seg(**kwargs) -> RichSegment:
    defaults = dict(
        start=0.0, end=2.0, text="Hello world this is normal speech.",
        model_used="distil-large-v3", difficulty="green", reason_flags=[],
        original_text=None, avg_logprob=-0.3, no_speech_prob=0.1,
        compression_ratio=1.2,
    )
    defaults.update(kwargs)
    return RichSegment(**defaults)


cfg = PipelineConfig()


def test_green_segment():
    seg = _seg()
    result = score_segment(seg, cfg)
    assert result.difficulty == "green"
    assert result.reason_flags == []


def test_yellow_low_logprob():
    seg = _seg(avg_logprob=-0.7)
    result = score_segment(seg, cfg)
    assert result.difficulty == "yellow"
    assert "low_logprob" in result.reason_flags


def test_red_very_low_logprob():
    seg = _seg(avg_logprob=-1.1)
    result = score_segment(seg, cfg)
    assert result.difficulty == "red"
    assert "low_logprob" in result.reason_flags


def test_yellow_no_speech():
    seg = _seg(no_speech_prob=0.4)
    result = score_segment(seg, cfg)
    assert result.difficulty == "yellow"
    assert "no_speech" in result.reason_flags


def test_red_no_speech():
    seg = _seg(no_speech_prob=0.7)
    result = score_segment(seg, cfg)
    assert result.difficulty == "red"
    assert "no_speech" in result.reason_flags


def test_yellow_high_compression():
    seg = _seg(compression_ratio=2.1)
    result = score_segment(seg, cfg)
    assert result.difficulty == "yellow"
    assert "compression_ratio" in result.reason_flags


def test_red_high_compression():
    seg = _seg(compression_ratio=2.6)
    result = score_segment(seg, cfg)
    assert result.difficulty == "red"
    assert "compression_ratio" in result.reason_flags


def test_yellow_repeated_words():
    seg = _seg(text="the the the is something")
    result = score_segment(seg, cfg)
    assert result.difficulty == "yellow"
    assert "repeated_words" in result.reason_flags


def test_red_repeated_words():
    seg = _seg(text="the the the the the is something")
    result = score_segment(seg, cfg)
    assert result.difficulty == "red"
    assert "repeated_words" in result.reason_flags


def test_yellow_short_segment():
    seg = _seg(text="OK")
    result = score_segment(seg, cfg)
    assert result.difficulty == "yellow"
    assert "short_segment" in result.reason_flags


def test_yellow_long_segment():
    seg = _seg(text=" ".join(["word"] * 65))
    result = score_segment(seg, cfg)
    assert result.difficulty == "yellow"
    assert "long_segment" in result.reason_flags


def test_red_hallucination_phrase():
    seg = _seg(text="Thanks for watching!")
    result = score_segment(seg, cfg)
    assert result.difficulty == "red"
    assert "hallucination_phrase" in result.reason_flags


def test_red_wins_over_yellow():
    # Both low logprob (yellow) and very low no_speech (red) — result must be red
    seg = _seg(avg_logprob=-0.7, no_speech_prob=0.7)
    result = score_segment(seg, cfg)
    assert result.difficulty == "red"


def test_multiple_flags_accumulated():
    seg = _seg(avg_logprob=-0.7, no_speech_prob=0.4)
    result = score_segment(seg, cfg)
    assert "low_logprob" in result.reason_flags
    assert "no_speech" in result.reason_flags
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_scorer.py -v
```

Expected: `FAILED` — `ModuleNotFoundError: No module named 'src.pipeline.scorer'`

- [ ] **Step 3: Implement `src/pipeline/scorer.py`**

```python
import re
from dataclasses import replace
from src.output import RichSegment
from src.pipeline.config import PipelineConfig


HALLUCINATION_PHRASES = [
    "thanks for watching",
    "subscribe",
    "like and subscribe",
    "vielen dank fürs zuschauen",
    "don't forget to subscribe",
    "see you in the next video",
    "bye bye",
    "please subscribe",
]


def _max_consecutive_repeated_words(text: str) -> int:
    words = text.lower().split()
    if not words:
        return 0
    max_run = 1
    run = 1
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 1
    return max_run


def score_segment(seg: RichSegment, cfg: PipelineConfig) -> RichSegment:
    """Classify segment difficulty and populate reason_flags. Returns new RichSegment."""
    flags: list[str] = []
    red_flags: list[str] = []

    text = seg.text.strip()
    word_count = len(text.split())

    # --- logprob ---
    if seg.avg_logprob is not None:
        if seg.avg_logprob < cfg.red_logprob:
            red_flags.append("low_logprob")
        elif seg.avg_logprob < cfg.yellow_logprob:
            flags.append("low_logprob")

    # --- no_speech_prob ---
    if seg.no_speech_prob is not None:
        if seg.no_speech_prob > cfg.red_no_speech:
            red_flags.append("no_speech")
        elif seg.no_speech_prob > cfg.yellow_no_speech:
            flags.append("no_speech")

    # --- compression_ratio ---
    if seg.compression_ratio is not None:
        cr = seg.compression_ratio
        if cr > cfg.red_compression_high or cr < cfg.red_compression_low:
            red_flags.append("compression_ratio")
        elif cr > cfg.yellow_compression_high or cr < cfg.yellow_compression_low:
            flags.append("compression_ratio")

    # --- repeated words ---
    max_run = _max_consecutive_repeated_words(text)
    if max_run >= cfg.red_repeat_words:
        red_flags.append("repeated_words")
    elif max_run >= cfg.yellow_repeat_words:
        flags.append("repeated_words")

    # --- segment length ---
    if word_count < 2:
        flags.append("short_segment")
    elif word_count > 60:
        flags.append("long_segment")

    # --- hallucination phrases ---
    text_lower = text.lower()
    for phrase in HALLUCINATION_PHRASES:
        if phrase in text_lower:
            red_flags.append("hallucination_phrase")
            break

    all_flags = flags + red_flags
    if red_flags:
        difficulty = "red"
    elif flags:
        difficulty = "yellow"
    else:
        difficulty = "green"

    return replace(seg, difficulty=difficulty, reason_flags=all_flags)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_scorer.py -v
```

Expected: all `PASSED`

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/ -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/pipeline/scorer.py tests/test_scorer.py
git commit -m "feat: add difficulty scorer (green/yellow/red) with configurable thresholds"
```

---

## Task 4: `src/pipeline/output.py` — write txt/json/srt/vtt from RichSegments

**Files:**
- Create: `src/pipeline/output.py`
- Create: `tests/test_pipeline_output.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_pipeline_output.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_pipeline_output.py -v
```

Expected: `FAILED` — `ModuleNotFoundError: No module named 'src.pipeline.output'`

- [ ] **Step 3: Implement `src/pipeline/output.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_pipeline_output.py -v
```

Expected: all `PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/output.py tests/test_pipeline_output.py
git commit -m "feat: add pipeline output writers (txt/json/srt/vtt)"
```

---

## Task 5: `src/pipeline/stages.py` — pipeline orchestration

This is the core stage runner. It wires preprocessing → first pass → scoring → re-transcription → assembly.

**Files:**
- Create: `src/pipeline/stages.py`
- Modify: `src/pipeline/__init__.py`
- Create: `tests/test_stages.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_stages.py`:

```python
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.output import RichSegment
from src.pipeline.config import PipelineConfig
from src.pipeline.stages import run_pipeline


FIXTURE = Path("tests/fixtures/hello.wav")


def _make_rich_seg(text="hello", difficulty="green", logprob=-0.3, no_speech=0.05):
    return RichSegment(
        start=0.0, end=1.0, text=text,
        model_used="distil-large-v3", difficulty=difficulty,
        reason_flags=[], original_text=None,
        avg_logprob=logprob, no_speech_prob=no_speech, compression_ratio=1.2,
    )


def test_run_pipeline_dry_run(tmp_path, capsys):
    cfg = PipelineConfig(dry_run=True, output_dir=str(tmp_path))
    with patch("src.pipeline.stages._transcribe_file") as mock_t:
        mock_t.return_value = [_make_rich_seg("hello world")]
        run_pipeline(FIXTURE, cfg)
    captured = capsys.readouterr()
    assert "green" in captured.out
    # no output files written in dry-run
    assert not list(tmp_path.glob("*.txt"))


def test_run_pipeline_writes_txt(tmp_path):
    cfg = PipelineConfig(
        first_pass_model="tiny",
        device="cpu",
        compute_type="int8",
        output_dir=str(tmp_path),
        vad=False,
    )
    run_pipeline(FIXTURE, cfg)
    txt_files = list(tmp_path.glob("*.txt"))
    assert len(txt_files) == 1


def test_run_pipeline_writes_json(tmp_path):
    cfg = PipelineConfig(
        first_pass_model="tiny",
        device="cpu",
        compute_type="int8",
        output_dir=str(tmp_path),
        vad=False,
    )
    run_pipeline(FIXTURE, cfg)
    json_files = list(tmp_path.glob("*.json"))
    assert len(json_files) == 1


def test_run_pipeline_writes_srt_when_requested(tmp_path):
    cfg = PipelineConfig(
        first_pass_model="tiny",
        device="cpu",
        compute_type="int8",
        output_dir=str(tmp_path),
        export_srt=True,
        vad=False,
    )
    run_pipeline(FIXTURE, cfg)
    assert list(tmp_path.glob("*.srt"))


def test_run_pipeline_writes_vtt_when_requested(tmp_path):
    cfg = PipelineConfig(
        first_pass_model="tiny",
        device="cpu",
        compute_type="int8",
        output_dir=str(tmp_path),
        export_vtt=True,
        vad=False,
    )
    run_pipeline(FIXTURE, cfg)
    assert list(tmp_path.glob("*.vtt"))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_stages.py -v
```

Expected: `FAILED` — `ImportError: cannot import name 'run_pipeline'`

- [ ] **Step 3: Implement `src/pipeline/stages.py`**

```python
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
            improved = {id(s): _retranscribe_segment(s, wav, cfg.yellow_pass_model, cfg) for s in yellow}
            segments = [improved.get(id(s), s) for s in segments]

        if red and cfg.enable_large_pass:
            print(f"  Stage 4b: Re-transcribing {len(red)} red segments ({cfg.red_pass_model})")
            improved = {id(s): _retranscribe_segment(s, wav, cfg.red_pass_model, cfg) for s in red}
            segments = [improved.get(id(s), s) for s in segments]
        elif red and not cfg.enable_large_pass:
            print(f"  Stage 4b: {len(red)} red segments — re-running with {cfg.yellow_pass_model} (large pass disabled)")
            improved = {id(s): _retranscribe_segment(s, wav, cfg.yellow_pass_model, cfg) for s in red}
            segments = [improved.get(id(s), s) for s in segments]

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
        file_cfg = PipelineConfig(**{**cfg.__dict__, "output_dir": str(file_out_dir)})
        run_pipeline(f, file_cfg)
```

- [ ] **Step 4: Update `src/pipeline/__init__.py`**

```python
from src.pipeline.stages import run_pipeline, run_pipeline_folder

__all__ = ["run_pipeline", "run_pipeline_folder"]
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_stages.py -v
```

Expected: all `PASSED` (the integration tests will actually run tiny model on hello.wav — this takes a few seconds)

- [ ] **Step 6: Run full test suite**

```bash
pytest tests/ -v
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/pipeline/stages.py src/pipeline/__init__.py tests/test_stages.py
git commit -m "feat: add pipeline stage orchestration with multi-pass re-transcription"
```

---

## Task 6: `transcribe_podcast.py` CLI + `pyproject.toml`

**Files:**
- Create: `transcribe_podcast.py`
- Create: `pyproject.toml`

- [ ] **Step 1: Create `transcribe_podcast.py`**

```python
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
    parser.add_argument("--enable-large-pass", action="store_true", help="Enable red → large-v3 re-transcription")
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
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "podcast-transcriber"
version = "0.1.0"
description = "Staged multi-pass podcast transcription using faster-whisper"
requires-python = ">=3.11"
dependencies = [
    "faster-whisper>=1.1.0,<2.0",
    "feedparser>=6.0,<7.0",
    "requests>=2.31,<3.0",
]

[project.scripts]
transcribe_podcast = "transcribe_podcast:main"

[tool.setuptools.packages.find]
where = ["."]
include = ["src*"]
```

- [ ] **Step 3: Install in editable mode so the console script works**

```bash
cd /home/tschwesi/claude/podCastBib && source .venv/bin/activate && pip install -e . --quiet
```

- [ ] **Step 4: Smoke test the CLI**

```bash
transcribe_podcast --help
```

Expected: prints usage with all flags listed, no errors.

- [ ] **Step 5: Dry-run smoke test on real fixture**

```bash
transcribe_podcast tests/fixtures/hello.wav --dry-run --output-dir /tmp/test_pipeline
```

Expected: prints scoring report, no files written.

- [ ] **Step 6: Run full test suite one final time**

```bash
pytest tests/ -v
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add transcribe_podcast.py pyproject.toml
git commit -m "feat: add transcribe_podcast CLI and pyproject.toml console script"
```

---

## Task 7: Push and release

- [ ] **Step 1: Push to remote**

```bash
git push
```

- [ ] **Step 2: Create release tag**

```bash
gh release create v0.2.0 --title "v0.2.0 — Multi-pass transcription pipeline" --notes "$(cat <<'EOF'
## New: Multi-pass transcription pipeline

Adds `transcribe_podcast` CLI — a staged pipeline that runs distil-large-v3 as a cheap first pass and re-transcribes only difficult segments with turbo or large-v3.

### Features

- `transcribe_podcast <file_or_folder>` — new standalone CLI
- Stage 1: Audio preprocessing (16kHz mono WAV)
- Stage 2: First pass with distil-large-v3
- Stage 3: Difficulty scoring (green/yellow/red) using logprob, no_speech_prob, compression_ratio + text heuristics
- Stage 4: Re-transcription of yellow (turbo) and red (large-v3) segments only
- Stage 5: Output — txt, json, optional srt/vtt
- `--dry-run` mode: score and report without re-transcribing
- All thresholds configurable via CLI flags
- Installed as console script via pyproject.toml

### Heuristics applied

- Low avg_logprob (< -0.6 yellow, < -1.0 red)
- High no_speech_prob (> 0.3 yellow, > 0.6 red)
- Compression ratio outliers
- Repeated words
- Short/long segments
- Known hallucination phrases
EOF
)"
```

---

## Heuristic tuning reference

All thresholds live in `PipelineConfig` and are overridable via CLI:

| CLI flag | Default | Effect |
|---|---|---|
| `--yellow-logprob` | -0.6 | Raise toward 0 to escalate fewer segments |
| `--red-logprob` | -1.0 | Raise toward 0 to make red harder to trigger |
| `--yellow-no-speech` | 0.3 | Lower to catch more noise segments |
| `--red-no-speech` | 0.6 | Lower to aggressively re-run noisy segments |

To inspect which segments are being escalated and why, run with `--dry-run --verbose`.
The `reason_flags` field in `transcript.json` shows exactly why each segment was escalated.
