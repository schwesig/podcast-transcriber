#!/usr/bin/env python3
"""
Backfill state DB from existing podcasts/ folder structure.

Scans all episode directories, computes SHA256 hashes for existing files,
and registers each episode as 'done' in the state DB.

Usage:
  python backfill_state.py
  python backfill_state.py --podcasts-dir podcasts --state-file podcasts/.podcast_transcriber_state.sqlite --dry-run
"""
import argparse
import re
import sys
from pathlib import Path

from src.state_db import StateDB, sha256_file, file_stats, DONE, PENDING


def _infer_model_and_feed(feed_dir_name: str) -> tuple[str, str]:
    """
    Infer model and feed slug from folder name.
    Patterns:
      base_talk-ohne-gast   -> model=base,  feed_slug=talk-ohne-gast
      small_talk-ohne-gast  -> model=small, feed_slug=talk-ohne-gast
      deconstructing-yourself -> model=small (unknown), feed_slug=deconstructing-yourself
    """
    known_models = {"tiny", "base", "small", "medium", "large-v2", "large-v3", "turbo", "distil-large-v3"}
    for m in known_models:
        prefix = f"{m}_"
        if feed_dir_name.startswith(prefix):
            return m, feed_dir_name[len(prefix):]
    return "small", feed_dir_name  # default


def _parse_dated_slug(filename_stem: str) -> tuple[str, str]:
    """
    Split '2023-08-23_reverse-meditation-with-andrew-holecek'
    into ('2023-08-23', 'reverse-meditation-with-andrew-holecek').
    """
    m = re.match(r'^(\d{4}-\d{2}-\d{2})_(.+)$', filename_stem)
    if m:
        return m.group(1), m.group(2)
    return "", filename_stem


def backfill(podcasts_dir: Path, state_file: Path, dry_run: bool = False) -> None:
    db = StateDB(state_file)
    counts = {"inserted": 0, "skipped": 0, "errors": 0}

    for feed_dir in sorted(podcasts_dir.iterdir()):
        if not feed_dir.is_dir() or feed_dir.name.startswith("."):
            continue

        model, feed_slug = _infer_model_and_feed(feed_dir.name)
        feed_title = feed_slug.replace("-", " ").title()

        for ep_dir in sorted(feed_dir.iterdir()):
            if not ep_dir.is_dir():
                continue

            # Find mp3 + txt files
            mp3_files = list(ep_dir.glob("*.mp3"))
            txt_files = list(ep_dir.glob("*.txt"))
            if not txt_files:
                continue  # no transcript = skip

            # Use the first txt as primary
            txt_path = txt_files[0]
            stem = txt_path.stem
            date_str, ep_slug = _parse_dated_slug(stem)

            audio_path = mp3_files[0] if mp3_files else Path("")
            srt_path = ep_dir / f"{stem}.srt"
            json_path = ep_dir / f"{stem}.json"
            nfo_path = ep_dir / f"{stem}.nfo"

            episode_title = ep_slug.replace("-", " ").title()
            pub_date = f"{date_str}T00:00:00" if date_str else ""

            print(f"  [{model}] {feed_slug} / {stem}")

            if dry_run:
                counts["inserted"] += 1
                continue

            try:
                episode_id = db.get_or_create(
                    feed_url="",           # unknown from disk scan
                    feed_title=feed_title,
                    feed_slug=feed_slug,
                    episode_title=episode_title,
                    episode_slug=ep_slug,
                    episode_guid=stem,     # use stem as stable guid
                    episode_audio_url="",
                    episode_pub_date=pub_date,
                    model=model,
                    language="",
                    pipeline_mode="",
                    output_dir=str(podcasts_dir),
                    episode_dir=str(ep_dir),
                    audio_path=str(audio_path),
                    transcript_txt_path=str(txt_path),
                    transcript_srt_path=str(srt_path),
                    metadata_json_path=str(json_path),
                    nfo_path=str(nfo_path),
                )

                # Compute hashes and mark steps done
                if audio_path.exists():
                    sha, size, mtime = file_stats(audio_path)
                    from src.state_db import _now
                    with db._tx():
                        db._conn.execute(
                            """UPDATE episodes SET download_status=?, audio_sha256=?,
                               audio_size_bytes=?, audio_mtime=?, updated_at=? WHERE id=?""",
                            (DONE, sha, size, mtime, _now(), episode_id),
                        )

                db.mark_transcription_done(
                    episode_id,
                    txt_path if txt_path.exists() else None,
                    srt_path if srt_path.exists() else None,
                )
                db.mark_metadata_done(episode_id, json_path if json_path.exists() else None)
                db.mark_nfo_done(episode_id, nfo_path if nfo_path.exists() else None)

                counts["inserted"] += 1
            except Exception as e:
                print(f"    ERROR: {e}")
                counts["errors"] += 1

    print(f"\nBackfill complete: {counts['inserted']} inserted  {counts['skipped']} skipped  {counts['errors']} errors")
    if not dry_run:
        print(f"State DB: {state_file}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill state DB from existing podcasts folder")
    parser.add_argument("--podcasts-dir", default="podcasts", help="Podcasts output directory")
    parser.add_argument("--state-file", default=None, help="State DB path")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be imported without writing")
    args = parser.parse_args()

    podcasts_dir = Path(args.podcasts_dir)
    if not podcasts_dir.exists():
        print(f"ERROR: {podcasts_dir} not found", file=sys.stderr)
        return 1

    state_file = Path(args.state_file) if args.state_file else podcasts_dir / ".podcast_transcriber_state.sqlite"

    if args.dry_run:
        print(f"[dry-run] scanning {podcasts_dir} ...")
    else:
        print(f"Backfilling state DB from {podcasts_dir} ...")

    backfill(podcasts_dir, state_file, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
