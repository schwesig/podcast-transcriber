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
      audio.mp3
      transcript.txt
      transcript.srt
```

## Run Tests

```bash
make test
```
