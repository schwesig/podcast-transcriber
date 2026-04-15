# podcast_sync Pipeline Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow feeds in `feeds.txt` to opt into the multi-pass pipeline via `pipeline=fast` or `pipeline=full`, while leaving feeds without that key on the existing single-pass path.

**Architecture:** Two small changes: (1) `FeedConfig` in `src/feeds.py` gains a `pipeline` field parsed from `feeds.txt`; (2) `process_episode` in `podcast_sync.py` branches on that field — pipeline feeds call `run_pipeline()`, non-pipeline feeds use the existing transcriber path unchanged.

**Tech Stack:** Python, faster-whisper, existing `src/pipeline/` modules

---

## File Map

| File | Action | Change |
|---|---|---|
| `src/feeds.py` | Modify | Add `pipeline: Optional[str] = None` to `FeedConfig`; parse `pipeline=` key |
| `podcast_sync.py` | Modify | Import pipeline modules; branch in `process_episode` |
| `tests/test_feeds.py` | Modify | Add test for `pipeline=` parsing |

---

## Task 1: Add `pipeline` field to `FeedConfig` and parse it

**Files:**
- Modify: `src/feeds.py`
- Modify: `tests/test_feeds.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_feeds.py`:

```python
def test_parse_feeds_file_pipeline(tmp_path):
    feeds_file = tmp_path / "feeds.txt"
    feeds_file.write_text(
        "https://example.com/feed1.xml model=small language=de pipeline=full\n"
        "https://example.com/feed2.xml pipeline=fast\n"
        "https://example.com/feed3.xml model=small language=en\n"
    )
    configs = parse_feeds_file(feeds_file)
    assert configs[0].pipeline == "full"
    assert configs[1].pipeline == "fast"
    assert configs[2].pipeline is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/tschwesi/claude/podCastBib && source .venv/bin/activate && pytest tests/test_feeds.py::test_parse_feeds_file_pipeline -v
```

Expected: `FAILED` — `AttributeError: 'FeedConfig' object has no attribute 'pipeline'`

- [ ] **Step 3: Add `pipeline` field to `FeedConfig` and parse it**

In `src/feeds.py`, change `FeedConfig` from:

```python
@dataclass
class FeedConfig:
    url: str
    model: str = "small"
    language: Optional[str] = None
```

to:

```python
@dataclass
class FeedConfig:
    url: str
    model: str = "small"
    language: Optional[str] = None
    pipeline: Optional[str] = None  # "full" | "fast" | None
```

In `parse_feeds_file`, change the kwargs initialization and parsing block from:

```python
        kwargs: dict = {"model": "small", "language": None}
        for part in parts[1:]:
            if "=" in part:
                k, v = part.split("=", 1)
                if k == "model":
                    kwargs["model"] = v
                elif k == "language":
                    kwargs["language"] = v
```

to:

```python
        kwargs: dict = {"model": "small", "language": None, "pipeline": None}
        for part in parts[1:]:
            if "=" in part:
                k, v = part.split("=", 1)
                if k == "model":
                    kwargs["model"] = v
                elif k == "language":
                    kwargs["language"] = v
                elif k == "pipeline":
                    kwargs["pipeline"] = v
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/tschwesi/claude/podCastBib && source .venv/bin/activate && pytest tests/test_feeds.py::test_parse_feeds_file_pipeline -v
```

Expected: `PASSED`

- [ ] **Step 5: Run full test suite**

```bash
cd /home/tschwesi/claude/podCastBib && source .venv/bin/activate && pytest tests/ -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
cd /home/tschwesi/claude/podCastBib && git add src/feeds.py tests/test_feeds.py && git commit -m "feat: add pipeline field to FeedConfig and parse from feeds.txt"
```

---

## Task 2: Branch on `feed_config.pipeline` in `process_episode`

**Files:**
- Modify: `podcast_sync.py`

- [ ] **Step 1: Add imports to `podcast_sync.py`**

At the top of `podcast_sync.py`, after the existing imports, add:

```python
from src.pipeline.config import PipelineConfig
from src.pipeline.stages import run_pipeline
```

The full import block should look like:

```python
from src.feeds import parse_feeds_file, parse_rss, FeedConfig, ParsedFeed, Episode
from src.downloader import download_audio
from src.sync_state import is_processed
from src.config import TranscribeConfig
from src.backend import get_transcriber
from src.audio import prepare_audio
from src.output import write_txt, write_srt, write_metadata, write_nfo
from src.pipeline.config import PipelineConfig
from src.pipeline.stages import run_pipeline
```

- [ ] **Step 2: Replace the transcription block in `process_episode`**

Find the current transcription block (lines 124–147):

```python
    cfg = TranscribeConfig(
        model=feed_config.model,
        device="auto",
        compute_type="int8",
        language=language,
        output_formats=["txt", "srt"],
    )

    print(f"  Transcribing with model={cfg.model} language={language or 'auto'} ...")
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory() as tmp:
        wav = prepare_audio(audio_path, Path(tmp))
        transcriber = get_transcriber(cfg)
        segments = transcriber.transcribe(wav)
    transcription_seconds = time.monotonic() - t0

    write_txt(segments, ep_dir / f"{stem}.txt")
    write_srt(segments, ep_dir / f"{stem}.srt")
    write_metadata(feed_title, ep, ep_dir / f"{stem}.json")
    write_nfo(audio_path, segments, transcription_seconds, cfg.model, ep_dir / f"{stem}.nfo")
    print(f"  -> {ep_dir / f'{stem}.txt'}")
    print(f"  -> {ep_dir / f'{stem}.srt'}")
    print(f"  -> {ep_dir / f'{stem}.json'}")
    print(f"  -> {ep_dir / f'{stem}.nfo'}")
```

Replace it with:

```python
    if feed_config.pipeline in ("fast", "full"):
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
        print(f"  Transcribing with pipeline={feed_config.pipeline} language={language or 'auto'} ...")
        run_pipeline(audio_path, pipeline_cfg)
    else:
        cfg = TranscribeConfig(
            model=feed_config.model,
            device="auto",
            compute_type="int8",
            language=language,
            output_formats=["txt", "srt"],
        )

        print(f"  Transcribing with model={cfg.model} language={language or 'auto'} ...")
        t0 = time.monotonic()
        with tempfile.TemporaryDirectory() as tmp:
            wav = prepare_audio(audio_path, Path(tmp))
            transcriber = get_transcriber(cfg)
            segments = transcriber.transcribe(wav)
        transcription_seconds = time.monotonic() - t0

        write_txt(segments, ep_dir / f"{stem}.txt")
        write_srt(segments, ep_dir / f"{stem}.srt")
        write_metadata(feed_title, ep, ep_dir / f"{stem}.json")
        write_nfo(audio_path, segments, transcription_seconds, cfg.model, ep_dir / f"{stem}.nfo")
        print(f"  -> {ep_dir / f'{stem}.txt'}")
        print(f"  -> {ep_dir / f'{stem}.srt'}")
        print(f"  -> {ep_dir / f'{stem}.json'}")
        print(f"  -> {ep_dir / f'{stem}.nfo'}")
```

- [ ] **Step 3: Run full test suite**

```bash
cd /home/tschwesi/claude/podCastBib && source .venv/bin/activate && pytest tests/ -v
```

Expected: all pass (no existing tests cover `process_episode` directly — the import change is what matters here).

- [ ] **Step 4: Smoke test the pipeline branch manually**

```bash
cd /home/tschwesi/claude/podCastBib && source .venv/bin/activate && python -c "
from src.feeds import FeedConfig
from src.pipeline.config import PipelineConfig
from src.pipeline.stages import run_pipeline
from pathlib import Path

feed_cfg = FeedConfig(url='x', pipeline='fast', language='en')
assert feed_cfg.pipeline == 'fast'

pipeline_cfg = PipelineConfig(
    first_pass_model='tiny',
    yellow_pass_model='tiny',
    red_pass_model='tiny',
    language='en',
    output_dir='/tmp/sync_smoke',
    vad=False,
    device='cpu',
    compute_type='int8',
)
run_pipeline(Path('tests/fixtures/hello.wav'), pipeline_cfg)
print('smoke test passed')
"
```

Expected: prints `smoke test passed`, files written to `/tmp/sync_smoke/`.

- [ ] **Step 5: Commit**

```bash
cd /home/tschwesi/claude/podCastBib && git add podcast_sync.py && git commit -m "feat: route pipeline= feeds through multi-pass pipeline in podcast_sync"
```

---

## Task 3: Update `pick_feed` display to show pipeline mode

The current `pick_feed` display shows `model=` and `language=`. For pipeline feeds, showing `pipeline=` instead of `model=` is more informative.

**Files:**
- Modify: `podcast_sync.py`

- [ ] **Step 1: Update `pick_feed` display**

Find in `podcast_sync.py`:

```python
def pick_feed(configs: list[FeedConfig]) -> FeedConfig:
    print("\nAvailable feeds:")
    for i, cfg in enumerate(configs, 1):
        print(f"  [{i}] {cfg.url}  model={cfg.model}  language={cfg.language or 'auto'}")
```

Replace with:

```python
def pick_feed(configs: list[FeedConfig]) -> FeedConfig:
    print("\nAvailable feeds:")
    for i, cfg in enumerate(configs, 1):
        mode = f"pipeline={cfg.pipeline}" if cfg.pipeline else f"model={cfg.model}"
        print(f"  [{i}] {cfg.url}  {mode}  language={cfg.language or 'auto'}")
```

- [ ] **Step 2: Run full test suite**

```bash
cd /home/tschwesi/claude/podCastBib && source .venv/bin/activate && pytest tests/ -v
```

Expected: all pass.

- [ ] **Step 3: Commit**

```bash
cd /home/tschwesi/claude/podCastBib && git add podcast_sync.py && git commit -m "feat: show pipeline mode in feed picker display"
```

---

## Task 4: Push and release v0.3.0

- [ ] **Step 1: Push**

```bash
cd /home/tschwesi/claude/podCastBib && git push
```

- [ ] **Step 2: Create release**

```bash
gh release create v0.3.0 --title "v0.3.0 — podcast_sync pipeline integration" --notes "$(cat <<'EOF'
## New: pipeline= option in feeds.txt

`podcast_sync.py` now supports opt-in multi-pass transcription per feed.

### Usage

Add `pipeline=fast` or `pipeline=full` to any feed in `feeds.txt`:

```
https://example.com/feed model=small language=de pipeline=full
https://example.com/feed2 language=en pipeline=fast
```

### Modes

- `pipeline=fast` — distil-large-v3 for all passes, scoring metadata in JSON, no re-transcription
- `pipeline=full` — distil-large-v3 first pass, turbo for yellow segments, large-v3 for red segments
- no pipeline= — existing single-pass behavior unchanged

### Output for pipeline feeds

- `.txt` and `.json` written (JSON includes model_used, difficulty, reason_flags per segment)
- `.srt` and `.nfo` not written (use `transcribe_podcast` CLI for those)
EOF
)"
```

---

## Verification (end-to-end)

1. Add `pipeline=fast` to one feed in `feeds.txt`
2. Run `python podcast_sync.py`, select that feed, pick one episode
3. Check `podcasts/<feed>/<episode>/<stem>.txt` and `.json` exist
4. Open `.json` — verify segments have `model_used`, `difficulty`, `reason_flags`
5. Run again on same episode — re-transcribe prompt appears (existing `is_processed` check still works)
6. Select a feed without `pipeline=` — `.nfo` and `.srt` written as before
