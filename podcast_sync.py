#!/usr/bin/env python3
"""
Usage:
  podcast_sync.py [--feeds feeds.txt] [--output-dir podcasts]
"""
import argparse
import sys
import tempfile
import time
from email.utils import parsedate
from pathlib import Path

from src.feeds import parse_feeds_file, parse_rss, FeedConfig, ParsedFeed, Episode
from src.downloader import download_audio
from src.sync_state import is_processed
from src.config import TranscribeConfig
from src.backend import get_transcriber
from src.audio import prepare_audio
from src.output import write_txt, write_srt, write_metadata, write_nfo
from src.pipeline.config import PipelineConfig
from src.pipeline.stages import run_pipeline


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


def pick_episodes(episodes: list[Episode]) -> tuple[list[Episode], bool]:
    """Returns (selected episodes, skip_existing flag)."""
    print("\nSelection mode:")
    print("  [1] All episodes")
    print("  [2] All not yet transcribed (skip existing)")
    print("  [3] Last N episodes")
    print("  [4] Pick individual episodes")
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
) -> None:
    ep_dir = output_dir / feed_slug / ep.slug
    if skip_existing and is_processed(ep_dir, ep.dated_slug):
        print(f"  [skip] {ep.title}")
        return

    stem = ep.dated_slug
    audio_path = ep_dir / f"{stem}.mp3"
    ep_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n>>> {ep.title}")
    download_audio(ep.audio_url, audio_path)

    if is_processed(ep_dir, ep.dated_slug):
        ans = input(f"  Transcript exists. Re-transcribe? [y/N]: ").strip().lower()
        if ans not in ("y", "yes"):
            print("  [skip] transcription")
            return

    if feed_config.pipeline in ("fast", "full"):
        pipeline_cfg = PipelineConfig(
            first_pass_model="base",
            yellow_pass_model="base" if feed_config.pipeline == "fast" else "turbo",
            red_pass_model="base" if feed_config.pipeline == "fast" else "large-v3",
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync and transcribe podcasts from RSS feeds")
    parser.add_argument("--feeds", default="feeds.txt", help="Path to feeds.txt (default: feeds.txt)")
    parser.add_argument("--output-dir", default="podcasts", help="Output directory (default: podcasts)")
    args = parser.parse_args()

    feeds_path = Path(args.feeds)
    if not feeds_path.exists():
        print(f"ERROR: {feeds_path} not found. Create it with one RSS URL per line.", file=sys.stderr)
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

    selected, skip_existing = pick_episodes(feed.episodes)
    language = resolve_language(feed_config, feed)
    output_dir = Path(args.output_dir)

    print(f"\nProcessing {len(selected)} episode(s) into {output_dir}/")
    for ep in selected:
        process_episode(ep, feed_config, language, output_dir, feed.slug, feed.title, skip_existing)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
