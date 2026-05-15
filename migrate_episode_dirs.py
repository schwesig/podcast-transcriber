#!/usr/bin/env python3
"""
Migrate episode directories from slug-only to dated-slug naming.

Before: podcasts/talk-ohne-gast/von-oben-eingegossen/
After:  podcasts/talk-ohne-gast/2024-11-06_von-oben-eingegossen/

Also updates all paths stored in the state DB.

Usage:
  python migrate_episode_dirs.py [--podcasts-dir podcasts] [--state-file PATH] [--dry-run]
"""
import argparse
import re
import shutil
import sqlite3
from pathlib import Path


_DATE_PREFIX = re.compile(r"^\d{4}-\d{2}-\d{2}_")


def _dated_slug_from_files(ep_dir: Path) -> str | None:
    """Infer dated_slug from files inside the directory (they already have dates)."""
    for f in ep_dir.iterdir():
        if f.suffix in (".txt", ".mp3", ".srt", ".json", ".nfo"):
            stem = f.stem
            if _DATE_PREFIX.match(stem):
                return stem
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--podcasts-dir", default="podcasts")
    parser.add_argument("--state-file", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    podcasts_dir = Path(args.podcasts_dir)
    state_file = Path(args.state_file) if args.state_file else podcasts_dir / ".podcast_transcriber_state.sqlite"

    if not podcasts_dir.exists():
        print(f"podcasts dir not found: {podcasts_dir}")
        return 1

    renames: list[tuple[Path, Path]] = []

    for feed_dir in sorted(podcasts_dir.iterdir()):
        if not feed_dir.is_dir() or feed_dir.name.startswith("."):
            continue
        for ep_dir in sorted(feed_dir.iterdir()):
            if not ep_dir.is_dir():
                continue
            # Already dated?
            if _DATE_PREFIX.match(ep_dir.name):
                continue
            dated = _dated_slug_from_files(ep_dir)
            if not dated:
                print(f"  [skip] {ep_dir} — no dated files found")
                continue
            new_dir = ep_dir.parent / dated
            renames.append((ep_dir, new_dir))

    if not renames:
        print("Nothing to migrate.")
        return 0

    print(f"{'[dry-run] ' if args.dry_run else ''}Migrating {len(renames)} directories:")
    for old, new in renames:
        print(f"  {old.relative_to(podcasts_dir.parent)}  →  {new.name}")

    if args.dry_run:
        print("\n[dry-run] No changes made.")
        return 0

    # Rename directories
    for old, new in renames:
        if new.exists():
            print(f"  [skip] target exists: {new}")
            continue
        shutil.move(str(old), str(new))

    # Update state DB
    if state_file.exists():
        conn = sqlite3.connect(str(state_file))
        conn.row_factory = sqlite3.Row
        updated = 0
        for old, new in renames:
            old_str = str(old)
            new_str = str(new)
            # Update episode_dir and all path columns
            conn.execute(
                """UPDATE episodes SET
                   episode_dir = REPLACE(episode_dir, ?, ?),
                   audio_path = REPLACE(audio_path, ?, ?),
                   transcript_txt_path = REPLACE(transcript_txt_path, ?, ?),
                   transcript_srt_path = REPLACE(transcript_srt_path, ?, ?),
                   metadata_json_path = REPLACE(metadata_json_path, ?, ?),
                   nfo_path = REPLACE(nfo_path, ?, ?)
                   WHERE episode_dir LIKE ?""",
                (old_str, new_str,
                 old_str, new_str,
                 old_str, new_str,
                 old_str, new_str,
                 old_str, new_str,
                 old_str, new_str,
                 f"{old_str}%"),
            )
            updated += conn.execute(
                "SELECT changes()"
            ).fetchone()[0]
        conn.commit()
        conn.close()
        print(f"\nUpdated {updated} DB rows.")
    else:
        print(f"\n[warn] State DB not found: {state_file} — skipped.")

    print("Migration complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
