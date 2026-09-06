# Local Speech-to-Text with Whisper

Transcribe audio files locally using [faster-whisper](https://github.com/SYSTRAN/faster-whisper). No cloud required. Switches to OpenAI API with one config change when needed.

## Requirements

- Python 3.10+
- ffmpeg (`sudo dnf install -y ffmpeg-free` on Fedora, or enable RPM Fusion for full ffmpeg)

## Setup

```bash
make setup
# or: ./setup.sh
```

## Usage

```bash
# Activate venv
source .venv/bin/activate

# Transcribe to txt (default)
python transcribe.py audio.mp3

# Specific model, multiple output formats
python transcribe.py audio.mp3 --model small --formats txt,json,srt

# Force language
python transcribe.py audio.mp3 --model base --language de

# Via make
make transcribe FILE=audio.mp3 ARGS="--model small --formats txt,srt"
```

## Models

| Model | Size | Speed (CPU i7) | Notes |
|---|---|---|---|
| `tiny` | 75 MB | ~10x realtime | Good for testing |
| `base` | 145 MB | ~7x realtime | Default, solid quality |
| `small` | 460 MB | ~4x realtime | Better accuracy |
| `medium` | 1.5 GB | ~2x realtime | High accuracy |
| `large-v3` | 3 GB | ~1x realtime | Best quality |

Models download automatically on first use and cache in `.models/`.

## Performance

**CPU (default):** Uses `int8` quantization. On an Intel i7, `base` model runs ~7x faster than realtime.

**GPU:** Pass `--device cuda` (NVIDIA) or set `device="cuda"` in config. Use `--compute float16` for better GPU performance.

## Output Formats

| Flag | File | Contents |
|---|---|---|
| `txt` | `audio.txt` | Plain text, one line per segment |
| `json` | `audio.json` | Array of `{start, end, text}` objects |
| `srt` | `audio.srt` | SubRip subtitles for video players |

## Backends

| Backend | Hardware | Install |
|---|---|---|
| `faster-whisper` | CPU (any), CUDA | default |
| `mlx` | Apple Silicon GPU | `pip install -e ".[mlx]"` |

`backend=auto` (the default) picks `mlx` on Apple Silicon when `mlx-whisper` is
installed, otherwise `faster-whisper`. Select it explicitly with `--backend`, or
per feed in `feeds.txt` via `backend=mlx`.

```bash
transcribe_podcast audio.mp3 --backend mlx
```

Caveats on the mlx backend:

- `turbo` and `distil-large-v3` have no mlx variant and fall back to
  faster-whisper automatically (with a log line).
- The full pipeline stays on faster-whisper unless a feed names a backend
  explicitly. `podcast_sync.py` overrides the models to `base`/`turbo`/
  `large-v3`, and `base` and `large-v3` do have mlx variants. Forwarding
  `auto` would move existing feeds onto the GPU silently, so `backend=mlx`
  in `feeds.txt` is required to opt in.
- mlx-whisper has no VAD and no beam search, so `--no-vad`, `--beam-size` and
  `--word-timestamps` are ignored there.
- Only `tiny`, `base`, `small`, `medium` and `large-v3` have mlx variants.
  Anything else (`large-v2`, the `.en` models, `turbo`, `distil-large-v3`)
  falls back to faster-whisper automatically.
- The state DB does not record which backend a run used, so `--retry-failed`
  reprocesses pipeline episodes with the default backend and prints a note
  when it does. Its quality metrics are reported per
  decoding window rather than per segment, making difficulty scoring coarser.

## Switching to OpenAI API

See `src/backend/api_stub.py` for instructions. The `Transcriber` protocol in `src/backend/__init__.py` is the only integration point.

## Podcast Sync

Sync and transcribe podcasts from RSS feeds.

### feeds.txt

One feed per line:

```
https://feeds.example.com/mypodcast model=small language=de
https://feeds.example.com/other
```

Options per feed (all optional):
- `model=small` — Whisper model to use (default: `small`)
- `language=de` — ISO 639-1 language code (default: auto-detect)
- `pipeline=full` — enable multi-pass transcription (see below)

### Run

```bash
make sync
# or
python podcast_sync.py
python podcast_sync.py --feeds my_feeds.txt --output-dir /data/podcasts
```

The CLI will prompt you to:
1. Pick a feed from `feeds.txt`
2. Choose: all / all new / last N / individual episodes

Output structure:
```
podcasts/
  podcast-slug/
    episode-slug/
      YYYY-MM-DD_episode-slug.mp3
      YYYY-MM-DD_episode-slug.txt
      YYYY-MM-DD_episode-slug.srt
      YYYY-MM-DD_episode-slug.json
      YYYY-MM-DD_episode-slug.nfo
```

### State DB & Resumable Processing

`podcast_sync.py` maintains a persistent SQLite ledger at:
```
podcasts/.podcast_transcriber_state.sqlite
```

This tracks every episode: download status, transcription status, file integrity (SHA256), and errors. Processing is fully resumable after crashes or interruptions.

#### CLI options

```bash
# Show status of all tracked episodes
python podcast_sync.py --status

# Verify artifact integrity (file existence + hash check)
python podcast_sync.py --verify

# Full SHA256 verification
python podcast_sync.py --verify-hashes

# Retry all failed/partial/stale episodes
python podcast_sync.py --retry-failed

# Re-download even if audio already verified
python podcast_sync.py --force-download

# Re-transcribe even if transcript already done
python podcast_sync.py --force-transcribe

# Custom state file location
python podcast_sync.py --state-file /path/to/state.sqlite
```

#### Status icons

| Icon | Meaning |
|------|---------|
| `✓` | done |
| `✗` | failed |
| `~` | partial (some steps done) |
| `↻` | stale (audio changed since transcription) |
| `▶` | running |
| `○` | pending |

#### Resumability behavior

- **Download**: skipped if audio file exists with matching SHA256
- **Transcription**: skipped if transcript exists with matching SHA256 and same model/language/audio
- **Crash recovery**: any `running` state at startup is reset to `partial` automatically
- **Stale detection**: if audio changes after transcription, transcript is marked stale and will be re-run
- **Batch failures**: one failing episode does not stop the rest; errors are recorded and a summary is printed

### Multi-pass Pipeline (`pipeline=full`)

```
feeds.txt: ... pipeline=full
```

Staged transcription:
1. **First pass** — `base` model on full audio (fast)
2. **Scoring** — classify each segment green/yellow/red by confidence
3. **Yellow re-pass** — `turbo` on difficult segments
4. **Red re-pass** — `large-v3` on worst segments (opt-in with `--enable-large-pass`)

Output: `.txt` and `.json` (JSON includes per-segment model, difficulty, confidence).

### Standalone Multi-pass CLI

```bash
transcribe_podcast audio.mp3 [options]
transcribe_podcast --help
```

## Run Tests

```bash
make test
```
