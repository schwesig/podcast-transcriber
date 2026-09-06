# podcast-transcriber — Feature Overview

Local podcast transcription pipeline. No cloud required. Downloads episodes from RSS feeds, transcribes with [faster-whisper](https://github.com/SYSTRAN/faster-whisper), tracks state across runs.

---

## What it does

1. Reads podcast feeds from `feeds.txt`
2. Fetches RSS, shows episode list
3. Downloads selected episodes
4. Transcribes locally using Whisper models
5. Writes transcripts to organized folder structure
6. Tracks everything in a persistent SQLite ledger

---

## Core Features

### RSS Feed Sync (`podcast_sync.py`)

Interactive CLI for downloading and transcribing podcast episodes.

```bash
python podcast_sync.py
python podcast_sync.py --feeds feeds.txt --output-dir podcasts
```

**Episode selection modes:**
- All episodes
- All not yet transcribed
- Last N episodes
- Individual pick by number or range (`1,3,5-8`)

**Per-feed configuration in `feeds.txt`:**
```
https://example.com/feed.rss model=small language=de
https://example.com/other.rss language=en pipeline=full
```

Options: `model=`, `language=`, `pipeline=`

---

### Single-Pass Transcription

Default mode. One model, one pass.

```
feeds.txt: model=base language=de
```

Output per episode:
```
podcasts/feed-slug/episode-slug/
  YYYY-MM-DD_episode-slug.mp3
  YYYY-MM-DD_episode-slug.txt
  YYYY-MM-DD_episode-slug.srt
  YYYY-MM-DD_episode-slug.json
  YYYY-MM-DD_episode-slug.nfo
```

`.nfo` contains hardware info, model used, realtime ratio, word count.

---

### Multi-Pass Pipeline (`pipeline=full`)

Staged transcription for higher quality on difficult audio.

```
feeds.txt: ... pipeline=full
```

**Stages:**
1. **Preprocessing** — convert to 16kHz mono WAV (temp file, auto-deleted)
2. **First pass** — `base` model on full audio (fast)
3. **Scoring** — classify each segment: `green` / `yellow` / `red`
4. **Yellow re-pass** — re-transcribe difficult segments with `turbo`
5. **Red re-pass** — re-transcribe worst segments with `large-v3` (opt-in via `--enable-large-pass`)
6. **Output** — write `.txt` and `.json` with per-segment metadata

JSON output includes per segment: model used, difficulty, confidence scores, reason flags.

**Scoring heuristics:**
- `avg_logprob` (confidence)
- `no_speech_prob` (silence / noise)
- `compression_ratio` (repetition)
- Repeated words
- Short / long segment anomalies
- Known hallucination phrases

---

### Standalone Pipeline CLI (`transcribe_podcast`)

Runs the multi-pass pipeline on any audio file or folder.

```bash
transcribe_podcast audio.mp3
transcribe_podcast folder/ --output-dir out/ --export-srt --language de
transcribe_podcast audio.mp3 --dry-run   # score only, no re-transcription
transcribe_podcast --help
```

Key flags: `--enable-large-pass`, `--beam-size`, `--no-vad`, `--word-timestamps`, `--export-srt`, `--export-vtt`, `--dry-run`, `--verbose`, `--device`, `--compute-type`, `--yellow-logprob`, `--red-logprob`

---

### Simple Transcription CLI (`transcribe.py`)

Minimal single-file transcription.

```bash
python transcribe.py audio.mp3 --model small --formats txt,json,srt
```

---

## State DB & Resumable Processing

Persistent SQLite ledger at `podcasts/.podcast_transcriber_state.sqlite`.

Tracks every episode through all steps with full integrity verification.

### What's tracked per episode

| Field | Description |
|---|---|
| Feed metadata | URL, title, slug |
| Episode metadata | title, slug, GUID, audio URL, pub date |
| Config | model, language, pipeline mode |
| Paths | audio, txt, srt, json, nfo |
| Step statuses | download, transcription, metadata, nfo, overall |
| Integrity | SHA256 + size + mtime for all artifacts |
| Errors | last error message, retry count |
| Timing | started, updated, completed |

### Status values

| Status | Meaning |
|---|---|
| `pending` | not started |
| `running` | in progress |
| `done` | completed and verified |
| `failed` | error occurred |
| `partial` | some steps done |
| `stale` | audio changed after transcription |

### Resumability

- **Download skip**: audio exists + size + SHA256 match → skip download
- **Transcription skip**: transcript exists + SHA256 match + same model/language/audio → skip
- **Crash recovery**: any `running` state at startup → reset to `partial` automatically
- **Stale detection**: audio hash changes → transcript marked `stale`, re-transcribed next run
- **Batch safety**: one episode failure doesn't stop the batch; all errors recorded

### CLI flags

```bash
python podcast_sync.py --status            # grouped status overview
python podcast_sync.py --verify            # verify file existence + hashes
python podcast_sync.py --verify-hashes     # full SHA256 verification
python podcast_sync.py --retry-failed      # retry failed/partial/stale episodes
python podcast_sync.py --force-download    # re-download even if verified
python podcast_sync.py --force-transcribe  # re-transcribe even if done
python podcast_sync.py --state-file PATH   # custom DB location
```

### Status display

```
deconstructing-yourself
  ✓ [small] 2023-08-23  Reverse Meditation With Andrew Holecek
  ✓ [small] 2023-07-13  A Conversation With Kati Devaney

talk-ohne-gast
  ✓ [base] 2019-02-15  Ein Hoch Auf Den Muttizettel
  ✗ [base] 2019-01-18  Marteria  ← connection refused

Total: 5  4 done  1 failed
```

### Backfill existing podcasts

Import pre-existing transcripts into the state DB without re-processing:

```bash
python backfill_state.py              # import all
python backfill_state.py --dry-run    # preview only
```

Infers model from folder prefix (`base_`, `small_`, `large-v3_`, etc.), computes SHA256 for all artifacts.

---

## Models

| Model | Size | Notes |
|---|---|---|
| `tiny` | 75 MB | fast, lower quality |
| `base` | 145 MB | default first pass |
| `small` | 244 MB | better quality |
| `medium` | 769 MB | high quality |
| `turbo` | ~809 MB | fast, good quality |
| `large-v3` | ~1.5 GB | best quality |
| `distil-large-v3` | ~1.5 GB | distilled, fast |

Models download automatically on first use, cached in `.models/`.

---

## Output Formats

| File | Contents |
|---|---|
| `.txt` | Plain text, one line per segment |
| `.srt` | SubRip subtitles |
| `.json` | Segments with timestamps (+ confidence metadata for pipeline) |
| `.nfo` | Hardware info, model, word count, realtime ratio |

---

## Requirements

- Python 3.11+
- ffmpeg

```bash
make setup
source .venv/bin/activate
```

---

## Schema Extensibility

The DB includes a `pipeline_stages` table for future stages:
diarization, speaker identification, chapter extraction, embeddings, summaries, semantic indexing.

Same episode processed with different models = separate records. Identity: `(feed_url, episode_guid, audio_sha256, model, language)`.

---

## Tests

```bash
make test
# 80 tests covering: state DB, pipeline scorer, output writers,
# feed parsing, audio utils, transcriber, backfill, integrity checks
```
