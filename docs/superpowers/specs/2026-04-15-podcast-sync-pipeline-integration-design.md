# podcast_sync.py Pipeline Integration — Design Spec

**Date:** 2026-04-15
**Status:** Approved

## Overview

Connect `podcast_sync.py` to the new multi-pass transcription pipeline. Feeds can opt in via `pipeline=fast` or `pipeline=full` in `feeds.txt`. Feeds without `pipeline=` continue using the existing single-pass behavior unchanged.

**Core principle:** Opt-in per feed. No breaking changes to existing behavior.

---

## 1. feeds.txt format change

New optional key `pipeline=` alongside existing `model=` and `language=`:

```
# Old behavior unchanged:
https://example.com/feed model=small language=de

# New pipeline modes:
https://example.com/feed model=small language=en pipeline=fast
https://example.com/feed model=small language=de pipeline=full
```

- `pipeline=fast` — distil-large-v3 for all passes (scoring metadata written, no actual re-transcription upgrade)
- `pipeline=full` — distil-large-v3 → turbo (yellow) → large-v3 (red, requires `--enable-large-pass` or always-on in sync)
- no `pipeline=` key — existing single-pass path, `model=` honored as before

---

## 2. FeedConfig change (`src/feeds.py`)

Add `pipeline` field:

```python
@dataclass
class FeedConfig:
    url: str
    model: str = "small"
    language: Optional[str] = None
    pipeline: Optional[str] = None  # "full" | "fast" | None
```

`parse_feeds_file` already handles `key=value` pairs. Add:

```python
elif k == "pipeline":
    kwargs["pipeline"] = v
```

---

## 3. `process_episode` change (`podcast_sync.py`)

**If `feed_config.pipeline` is set:**

```python
from src.pipeline.config import PipelineConfig
from src.pipeline.stages import run_pipeline

pipeline_cfg = PipelineConfig(
    first_pass_model="distil-large-v3",
    yellow_pass_model="distil-large-v3" if feed_config.pipeline == "fast" else "turbo",
    red_pass_model="distil-large-v3" if feed_config.pipeline == "fast" else "large-v3",
    language=language,
    output_dir=str(ep_dir),
    vad=True,
    device="auto",
    compute_type="int8",
    model_cache_dir=".models",
)
run_pipeline(audio_path, pipeline_cfg)
```

`run_pipeline` writes `{stem}.txt` and `{stem}.json` to `ep_dir`. The existing re-transcribe prompt (`is_processed` check) still runs before this block — no change there.

**If `feed_config.pipeline` is None:**

Existing path unchanged — `get_transcriber(cfg).transcribe(wav)` → write txt/srt/json/nfo.

---

## 4. Output differences for pipeline episodes

| File | pipeline episode | non-pipeline episode |
|---|---|---|
| `.txt` | ✅ written by `run_pipeline` | ✅ written as before |
| `.json` | ✅ written by `run_pipeline` (RichSegment format with model_used, difficulty, reason_flags) | ✅ written as before (simple segment format) |
| `.srt` | ❌ not written | ✅ written as before |
| `.nfo` | ❌ not written | ✅ written as before |

The `.json` for pipeline episodes is richer (includes `model_used`, `difficulty`, `reason_flags`, `original_text` per segment) which is acceptable — it's machine-readable JSON consumed by `podcast-transcripts`.

---

## 5. Files to change

| File | Change |
|---|---|
| `src/feeds.py` | Add `pipeline: Optional[str] = None` to `FeedConfig`; parse `pipeline=` in `parse_feeds_file` |
| `podcast_sync.py` | Import `PipelineConfig`, `run_pipeline`; branch on `feed_config.pipeline` in `process_episode` |

**No changes to:** `src/pipeline/`, `src/backend/`, `src/output.py`, `feeds.txt` (user edits manually).

---

## 6. Verification

1. Add `pipeline=fast` to one feed in `feeds.txt`
2. Run `python podcast_sync.py`, select that feed, download one episode
3. Check output: `{ep_dir}/{stem}.txt` and `{ep_dir}/{stem}.json` exist
4. Check `.json`: segments have `model_used`, `difficulty`, `reason_flags` fields
5. Run `python podcast_sync.py` again on same episode — re-transcribe prompt should appear
6. Select another feed without `pipeline=` — verify old behavior unchanged, `.nfo` written
