#!/usr/bin/env python3
"""
Usage:
  podcast_sync.py [--feeds feeds.txt] [--output-dir podcasts] [--state-file PATH]
  podcast_sync.py --status
  podcast_sync.py --verify [--verify-hashes]
  podcast_sync.py --retry-failed
"""
import argparse
import sys
import tempfile
import time
import traceback
from email.utils import parsedate
from pathlib import Path

from src.feeds import parse_feeds_file, parse_rss, FeedConfig, ParsedFeed, Episode
from src.downloader import download_audio
from src.config import TranscribeConfig
from src.backend import get_transcriber
from src.audio import prepare_audio
from src.output import write_txt, write_srt, write_metadata, write_nfo
from src.pipeline.config import PipelineConfig
from src.pipeline.stages import run_pipeline
from src.state_db import (
    StateDB, atomic_write_text,
    DONE, FAILED, PARTIAL, STALE,
    print_status, print_verify,
)


def _fmt_date(pub_date: str) -> str:
    try:
        t = parsedate(pub_date)
        if t:
            return f"{t[0]}-{t[1]:02}-{t[2]:02}"
    except Exception:
        pass
    return "          "


def pick_feed(configs: list[FeedConfig]) -> FeedConfig:
    print("\nAvailable feeds:")
    for i, cfg in enumerate(configs, 1):
        mode = f"pipeline={cfg.pipeline}" if cfg.pipeline else f"model={cfg.model}"
        print(f"  [{i}] {cfg.url}  {mode}  language={cfg.language or 'auto'}")
    while True:
        raw = input("\nSelect feed number: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(configs):
            return configs[int(raw) - 1]
        print("  Invalid choice, try again.")


def _episode_status(db: StateDB, feed_url: str, episode_guid: str, model: str, language: str) -> str:
    """Return overall_status from DB, or 'new' if not tracked yet.

    Tries exact match first (feed_url + guid + model + language).
    Falls back to guid-only match to handle backfill rows with empty feed_url.
    """
    row = db._conn.execute(
        """SELECT overall_status FROM episodes
           WHERE feed_url=? AND episode_guid=? AND model=? AND language=?
           ORDER BY id DESC LIMIT 1""",
        (feed_url, episode_guid, model, language),
    ).fetchone()
    if row:
        return row["overall_status"]
    # Fallback: guid only (covers backfill rows with empty feed_url)
    row = db._conn.execute(
        """SELECT overall_status FROM episodes
           WHERE episode_guid=? AND overall_status=?
           ORDER BY id DESC LIMIT 1""",
        (episode_guid, DONE),
    ).fetchone()
    return row["overall_status"] if row else "new"


def pick_episodes(
    episodes: list[Episode],
    db: StateDB | None = None,
    feed_url: str = "",
    model: str = "",
    language: str = "",
) -> tuple[list[Episode], bool]:
    """Returns (selected episodes, skip_existing flag)."""
    print("\nSelection mode:")
    print("  [1] All episodes")
    print("  [2] All not yet transcribed (skip existing)")
    print("  [3] Last N episodes")
    print("  [4] Pick individual episodes")
    print("  [5] Only missing or failed episodes")
    while True:
        mode = input("\nSelect mode: ").strip()
        if mode == "1":
            return episodes, False
        if mode == "2":
            return episodes, True
        if mode == "3":
            n = input("  How many? ").strip()
            if n.isdigit() and int(n) > 0:
                return episodes[:int(n)], False
            print("  Invalid number.")
        elif mode == "4":
            print("\nEpisodes:  [pos] #ep  date  title")
            for i, ep in enumerate(episodes, 1):
                date = _fmt_date(ep.pub_date)
                ep_num = f"#{ep.episode_number}" if ep.episode_number else "---"
                print(f"  [{i:3}] {ep_num:>5}  {date}  {ep.title}")
            raw = input("  Enter numbers (e.g. 1,3,5-8): ").strip()
            indices = []
            for part in raw.split(","):
                part = part.strip()
                if "-" in part:
                    bounds = part.split("-", 1)
                    if bounds[0].isdigit() and bounds[1].isdigit():
                        a, b = int(bounds[0]), int(bounds[1])
                        indices.extend(i - 1 for i in range(a, b + 1) if 1 <= i <= len(episodes))
                elif part.isdigit() and 1 <= int(part) <= len(episodes):
                    indices.append(int(part) - 1)
            seen = set()
            indices = [i for i in indices if not (i in seen or seen.add(i))]
            if indices:
                return [episodes[i] for i in indices], False
            print("  No valid selection.")
        elif mode == "5":
            if not db or not feed_url:
                print("  Status filter unavailable.")
                continue
            _done = {DONE}
            pending = [
                ep for ep in episodes
                if _episode_status(db, feed_url, ep.guid, model, language) not in _done
            ]
            if not pending:
                print("  All episodes already done — nothing to process.")
                continue
            print(f"\n  {len(pending)} missing/failed episodes:")
            for ep in pending:
                status = _episode_status(db, feed_url, ep.guid, model, language)
                date = _fmt_date(ep.pub_date)
                print(f"    [{status:8}]  {date}  {ep.title}")
            confirm = input(f"\n  Process all {len(pending)}? [Y/n]: ").strip().lower()
            if confirm in ("", "y", "yes"):
                return pending, False
            print("  Cancelled.")


def resolve_language(feed_config: FeedConfig, feed: ParsedFeed) -> str | None:
    if feed_config.language:
        return feed_config.language
    if feed.language:
        lang = feed.language.split("-")[0].lower()
        confirm = input(f"\n  Detected language '{lang}' from feed. Use it? [Y/n]: ").strip().lower()
        if confirm in ("", "y", "yes"):
            return lang
    answer = input("\n  Enter language code (e.g. de, en) or leave empty for auto-detect: ").strip()
    return answer if answer else None


def process_episode(
    ep: Episode,
    feed_config: FeedConfig,
    language: str | None,
    output_dir: Path,
    feed_slug: str,
    feed_title: str,
    skip_existing: bool,
    db: StateDB,
    force_download: bool = False,
    force_transcribe: bool = False,
) -> str:
    """Process one episode. Returns outcome: 'done' | 'skipped' | 'failed' | 'partial'."""
    ep_dir = output_dir / feed_slug / ep.dated_slug
    stem = ep.dated_slug
    audio_path = ep_dir / f"{stem}.mp3"
    txt_path = ep_dir / f"{stem}.txt"
    srt_path = ep_dir / f"{stem}.srt"
    json_path = ep_dir / f"{stem}.json"
    nfo_path = ep_dir / f"{stem}.nfo"

    model = feed_config.model if not feed_config.pipeline else ""
    pipeline_mode = feed_config.pipeline or ""

    episode_id = db.get_or_create(
        feed_url=feed_config.url,
        feed_title=feed_title,
        feed_slug=feed_slug,
        episode_title=ep.title,
        episode_slug=ep.slug,
        episode_guid=ep.guid,
        episode_audio_url=ep.audio_url,
        episode_pub_date=ep.pub_date,
        model=model,
        language=language or "",
        pipeline_mode=pipeline_mode,
        output_dir=str(output_dir),
        episode_dir=str(ep_dir),
        audio_path=str(audio_path),
        transcript_txt_path=str(txt_path),
        transcript_srt_path=str(srt_path),
        metadata_json_path=str(json_path),
        nfo_path=str(nfo_path),
    )

    # Check stale state
    db.detect_stale(episode_id)

    # Skip if fully done and not forced
    if skip_existing and db.is_episode_complete(episode_id) and not force_download and not force_transcribe:
        print(f"  [skip] {ep.title}")
        return "skipped"

    ep_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n>>> {ep.title}")

    # ── Download ──────────────────────────────────────────────────────────────
    if force_download or not db.should_skip_download(episode_id, audio_path):
        db.mark_download_running(episode_id)
        try:
            download_audio(ep.audio_url, audio_path)
            db.mark_download_done(episode_id, audio_path)
        except Exception as e:
            err = str(e)
            db.mark_download_failed(episode_id, err)
            print(f"  [download failed] {err}")
            return "failed"
    else:
        print(f"  [skip download] audio already verified")

    # ── Skip transcription check ──────────────────────────────────────────────
    if db.should_skip_transcription(episode_id, txt_path, force=force_transcribe):
        if skip_existing:
            print(f"  [skip transcription] already done")
            return "skipped"
        # Not skipping — ask user
        ans = input(f"  Transcript exists. Re-transcribe? [y/N]: ").strip().lower()
        if ans not in ("y", "yes"):
            print("  [skip] transcription")
            return "skipped"

    # ── Transcribe ────────────────────────────────────────────────────────────
    try:
        if feed_config.pipeline == "full":
            pipeline_cfg = PipelineConfig(
                first_pass_model="base",
                yellow_pass_model="turbo",
                red_pass_model="large-v3",
                language=language,
                output_dir=str(ep_dir),
                backend=feed_config.backend,
                vad=True,
                device="auto",
                compute_type="int8",
                model_cache_dir=".models",
            )
            print(f"  Transcribing with pipeline={feed_config.pipeline} language={language or 'auto'} ...")
            db.mark_transcription_running(episode_id)
            run_pipeline(audio_path, pipeline_cfg)
            db.mark_transcription_done(episode_id, txt_path)
            db.mark_metadata_done(episode_id)   # pipeline writes json
            db.mark_nfo_done(episode_id)
        else:
            cfg = TranscribeConfig(
                model=feed_config.model,
                device="auto",
                compute_type="int8",
                language=language,
                output_formats=["txt", "srt"],
                backend=feed_config.backend,
            )
            print(f"  Transcribing with model={cfg.model} language={language or 'auto'} ...")
            db.mark_transcription_running(episode_id)
            t0 = time.monotonic()
            with tempfile.TemporaryDirectory() as tmp:
                wav = prepare_audio(audio_path, Path(tmp))
                transcriber = get_transcriber(cfg)
                segments = transcriber.transcribe(wav)
            transcription_seconds = time.monotonic() - t0

            engine = getattr(transcriber, "engine_name", "faster-whisper")

            # Atomic writes
            write_txt(segments, txt_path)
            write_srt(segments, srt_path)
            write_metadata(feed_title, ep, json_path)
            write_nfo(audio_path, segments, transcription_seconds, cfg.model, nfo_path, engine=engine)

            db.mark_transcription_done(episode_id, txt_path, srt_path)
            db.mark_metadata_done(episode_id, json_path)
            db.mark_nfo_done(episode_id, nfo_path)

            print(f"  -> {txt_path}")
            print(f"  -> {srt_path}")
            print(f"  -> {json_path}")
            print(f"  -> {nfo_path}")

    except Exception as e:
        err = traceback.format_exc()
        db.mark_transcription_failed(episode_id, str(e))
        print(f"  [transcription failed] {e}")
        return "failed"

    return "done"


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync and transcribe podcasts from RSS feeds")
    parser.add_argument("--feeds", default="feeds.txt", help="Path to feeds.txt")
    parser.add_argument("--output-dir", default="podcasts", help="Output directory")
    parser.add_argument("--state-file", default=None, help="Path to state DB (default: <output-dir>/.podcast_transcriber_state.sqlite)")
    parser.add_argument("--force-download", action="store_true", help="Re-download even if already downloaded")
    parser.add_argument("--force-transcribe", action="store_true", help="Re-transcribe even if transcript exists")
    parser.add_argument("--retry-failed", action="store_true", help="Retry failed/partial/stale episodes (non-interactive)")
    parser.add_argument("--status", action="store_true", help="Show processing status overview")
    parser.add_argument("--verify", action="store_true", help="Verify artifact integrity")
    parser.add_argument("--verify-hashes", action="store_true", help="Full SHA256 verification")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    state_file = Path(args.state_file) if args.state_file else output_dir / ".podcast_transcriber_state.sqlite"
    output_dir.mkdir(parents=True, exist_ok=True)
    db = StateDB(state_file)

    # ── Read-only commands ────────────────────────────────────────────────────
    if args.status:
        print_status(db)
        return 0

    if args.verify or args.verify_hashes:
        print_verify(db, full_hash=args.verify_hashes)
        return 0

    # ── Retry failed ──────────────────────────────────────────────────────────
    if args.retry_failed:
        rows = db.list_failed_or_partial()
        if not rows:
            print("No failed/partial/stale episodes.")
            return 0
        print(f"Retrying {len(rows)} episode(s)...")
        counts = {"done": 0, "skipped": 0, "failed": 0, "partial": 0}
        for r in rows:
            # Reconstruct minimal feed_config from DB record
            fc = FeedConfig(
                url=r["feed_url"],
                model=r["model"] or "small",
                language=r["language"] or None,
                pipeline=r["pipeline_mode"] or None,
            )
            ep = Episode(
                title=r["episode_title"],
                guid=r["episode_guid"],
                audio_url=r["episode_audio_url"],
                pub_date=r["episode_pub_date"],
            )
            outcome = process_episode(
                ep, fc, r["language"] or None,
                Path(r["output_dir"]), r["feed_slug"], r["feed_title"],
                skip_existing=False, db=db,
                force_download=False, force_transcribe=True,
            )
            counts[outcome] = counts.get(outcome, 0) + 1
        _print_summary(counts)
        return 0

    # ── Normal interactive flow ───────────────────────────────────────────────
    feeds_path = Path(args.feeds)
    if not feeds_path.exists():
        print(f"ERROR: {feeds_path} not found.", file=sys.stderr)
        return 1

    configs = parse_feeds_file(feeds_path)
    if not configs:
        print("ERROR: No feeds found in feeds.txt", file=sys.stderr)
        return 1

    feed_config = pick_feed(configs)
    print(f"\nFetching feed: {feed_config.url}")
    feed = parse_rss(feed_config.url)
    print(f"  Found: {feed.title} ({len(feed.episodes)} episodes)")

    if not feed.episodes:
        print("  No downloadable episodes found.")
        return 0

    language = resolve_language(feed_config, feed)
    model = feed_config.model if not feed_config.pipeline else ""
    selected, skip_existing = pick_episodes(
        feed.episodes,
        db=db,
        feed_url=feed_config.url,
        model=model,
        language=language or "",
    )

    print(f"\nProcessing {len(selected)} episode(s) into {output_dir}/")
    counts = {"done": 0, "skipped": 0, "failed": 0, "partial": 0}
    for ep in selected:
        try:
            outcome = process_episode(
                ep, feed_config, language, output_dir,
                feed.slug, feed.title, skip_existing, db,
                force_download=args.force_download,
                force_transcribe=args.force_transcribe,
            )
            counts[outcome] = counts.get(outcome, 0) + 1
        except Exception as e:
            print(f"  [unexpected error] {ep.title}: {e}")
            counts["failed"] += 1

    _print_summary(counts)
    return 0


def _print_summary(counts: dict) -> None:
    print("\nDone.")
    parts = [f"{v} {k}" for k, v in counts.items() if v > 0]
    if parts:
        print("  " + "  ".join(parts))


if __name__ == "__main__":
    sys.exit(main())
