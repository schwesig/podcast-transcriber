#!/usr/bin/env python3
"""
Remove duplicate 'failed' rows caused by UNIQUE constraint errors in get_or_create.

These are phantom rows — the real episode row exists with status=done.
Safe to delete: they have no transcription artifacts, only error state.

Usage:
  python cleanup_duplicate_errors.py [--state-file PATH] [--dry-run]
"""
import argparse
import sqlite3
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-file", default="podcasts/.podcast_transcriber_state.sqlite")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    db_path = Path(args.state_file)
    if not db_path.exists():
        print(f"DB not found: {db_path}")
        return 1

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        """SELECT id, episode_title, last_error FROM episodes
           WHERE overall_status='failed'
           AND last_error LIKE '%UNIQUE constraint%'"""
    ).fetchall()

    if not rows:
        print("No duplicate error rows found.")
        conn.close()
        return 0

    print(f"Found {len(rows)} duplicate error rows:")
    for r in rows:
        print(f"  id={r['id']}  {r['episode_title'][:60]}")

    if args.dry_run:
        print("\n[dry-run] No changes made.")
        conn.close()
        return 0

    ids = [r["id"] for r in rows]
    conn.execute(f"DELETE FROM episodes WHERE id IN ({','.join('?' * len(ids))})", ids)
    conn.commit()
    print(f"\nDeleted {len(ids)} rows.")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
