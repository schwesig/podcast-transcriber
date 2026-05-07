"""
Persistent processing ledger for podcast transcription.

Tracks per-episode state: download, transcription, metadata, integrity.
State identity: (feed_url, episode_guid, audio_sha256, model, language).
Same episode + different model = separate record.

Default DB location: podcasts/.podcast_transcriber_state.sqlite
"""
import hashlib
import os
import shutil
import sqlite3
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ── status constants ──────────────────────────────────────────────────────────
PENDING = "pending"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
PARTIAL = "partial"
STALE = "stale"

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS episodes (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Feed metadata
    feed_url            TEXT    NOT NULL,
    feed_title          TEXT    NOT NULL DEFAULT '',
    feed_slug           TEXT    NOT NULL DEFAULT '',

    -- Episode metadata
    episode_title       TEXT    NOT NULL DEFAULT '',
    episode_slug        TEXT    NOT NULL DEFAULT '',
    episode_guid        TEXT    NOT NULL DEFAULT '',
    episode_audio_url   TEXT    NOT NULL DEFAULT '',
    episode_pub_date    TEXT    NOT NULL DEFAULT '',

    -- Processing configuration
    model               TEXT    NOT NULL DEFAULT '',
    language            TEXT    NOT NULL DEFAULT '',
    pipeline_mode       TEXT    NOT NULL DEFAULT '',

    -- Filesystem paths
    output_dir          TEXT    NOT NULL DEFAULT '',
    episode_dir         TEXT    NOT NULL DEFAULT '',
    audio_path          TEXT    NOT NULL DEFAULT '',
    transcript_txt_path TEXT    NOT NULL DEFAULT '',
    transcript_srt_path TEXT    NOT NULL DEFAULT '',
    metadata_json_path  TEXT    NOT NULL DEFAULT '',
    nfo_path            TEXT    NOT NULL DEFAULT '',

    -- Per-step statuses
    download_status     TEXT    NOT NULL DEFAULT 'pending',
    transcription_status TEXT   NOT NULL DEFAULT 'pending',
    metadata_status     TEXT    NOT NULL DEFAULT 'pending',
    nfo_status          TEXT    NOT NULL DEFAULT 'pending',
    overall_status      TEXT    NOT NULL DEFAULT 'pending',

    -- Timing
    started_at          TEXT,
    updated_at          TEXT,
    completed_at        TEXT,

    -- Error tracking
    last_error          TEXT,
    retry_count         INTEGER NOT NULL DEFAULT 0,

    -- Audio integrity
    audio_sha256        TEXT    NOT NULL DEFAULT '',
    audio_size_bytes    INTEGER NOT NULL DEFAULT 0,
    audio_mtime         REAL    NOT NULL DEFAULT 0.0,

    -- Transcript integrity
    transcript_txt_sha256  TEXT NOT NULL DEFAULT '',
    transcript_srt_sha256  TEXT NOT NULL DEFAULT '',
    metadata_json_sha256   TEXT NOT NULL DEFAULT '',
    nfo_sha256             TEXT NOT NULL DEFAULT '',

    -- Unique processing identity
    UNIQUE(feed_url, episode_guid, audio_sha256, model, language)
);

-- Future pipeline stages (diarization, embeddings, etc.) can add rows here
-- without touching the episodes table schema.
CREATE TABLE IF NOT EXISTS pipeline_stages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id  INTEGER NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    stage_name  TEXT    NOT NULL,
    status      TEXT    NOT NULL DEFAULT 'pending',
    started_at  TEXT,
    completed_at TEXT,
    error       TEXT,
    metadata    TEXT,   -- JSON blob for stage-specific data
    UNIQUE(episode_id, stage_name)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    """Return hex SHA256 of file contents. Returns '' on error."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return ""


def file_stats(path: Path) -> tuple[str, int, float]:
    """Return (sha256, size_bytes, mtime) or ('', 0, 0.0) on error."""
    try:
        st = path.stat()
        return sha256_file(path), st.st_size, st.st_mtime
    except OSError:
        return "", 0, 0.0


def atomic_write_text(path: Path, content: str) -> None:
    """Write text to path atomically via temp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        shutil.move(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_bytes(path: Path, content: bytes) -> None:
    """Write bytes to path atomically via temp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
        shutil.move(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class StateDB:
    """Durable processing ledger backed by SQLite."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._cleanup_stale_running()

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def _tx(self):
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # ── Stale running cleanup ─────────────────────────────────────────────────

    def _cleanup_stale_running(self) -> None:
        """On startup: any 'running' record from a previous crashed run → 'partial'."""
        with self._tx():
            self._conn.execute(
                "UPDATE episodes SET overall_status=?, updated_at=? WHERE overall_status=?",
                (PARTIAL, _now(), RUNNING),
            )
            self._conn.execute(
                "UPDATE episodes SET download_status=?, updated_at=? WHERE download_status=?",
                (PARTIAL, _now(), RUNNING),
            )
            self._conn.execute(
                "UPDATE episodes SET transcription_status=?, updated_at=? WHERE transcription_status=?",
                (PARTIAL, _now(), RUNNING),
            )

    # ── Record creation ───────────────────────────────────────────────────────

    def get_or_create(
        self,
        *,
        feed_url: str,
        feed_title: str,
        feed_slug: str,
        episode_title: str,
        episode_slug: str,
        episode_guid: str,
        episode_audio_url: str,
        episode_pub_date: str,
        model: str,
        language: str,
        pipeline_mode: str,
        output_dir: str,
        episode_dir: str,
        audio_path: str,
        transcript_txt_path: str,
        transcript_srt_path: str,
        metadata_json_path: str,
        nfo_path: str,
    ) -> int:
        """Return id of existing record or create new one. audio_sha256='' until downloaded."""
        now = _now()
        with self._tx():
            row = self._conn.execute(
                """SELECT id FROM episodes
                   WHERE feed_url=? AND episode_guid=? AND audio_sha256='' AND model=? AND language=?""",
                (feed_url, episode_guid, model, language),
            ).fetchone()
            if row:
                # Update paths in case output_dir changed
                self._conn.execute(
                    """UPDATE episodes SET
                       feed_title=?, feed_slug=?, episode_title=?, episode_slug=?,
                       episode_audio_url=?, episode_pub_date=?, pipeline_mode=?,
                       output_dir=?, episode_dir=?, audio_path=?,
                       transcript_txt_path=?, transcript_srt_path=?,
                       metadata_json_path=?, nfo_path=?, updated_at=?
                       WHERE id=?""",
                    (feed_title, feed_slug, episode_title, episode_slug,
                     episode_audio_url, episode_pub_date, pipeline_mode,
                     output_dir, episode_dir, audio_path,
                     transcript_txt_path, transcript_srt_path,
                     metadata_json_path, nfo_path, now, row["id"]),
                )
                return row["id"]

            cur = self._conn.execute(
                """INSERT INTO episodes (
                   feed_url, feed_title, feed_slug,
                   episode_title, episode_slug, episode_guid,
                   episode_audio_url, episode_pub_date,
                   model, language, pipeline_mode,
                   output_dir, episode_dir, audio_path,
                   transcript_txt_path, transcript_srt_path,
                   metadata_json_path, nfo_path,
                   started_at, updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (feed_url, feed_title, feed_slug,
                 episode_title, episode_slug, episode_guid,
                 episode_audio_url, episode_pub_date,
                 model, language, pipeline_mode,
                 output_dir, episode_dir, audio_path,
                 transcript_txt_path, transcript_srt_path,
                 metadata_json_path, nfo_path,
                 now, now),
            )
            return cur.lastrowid

    # ── Download tracking ─────────────────────────────────────────────────────

    def mark_download_running(self, episode_id: int) -> None:
        with self._tx():
            self._conn.execute(
                "UPDATE episodes SET download_status=?, overall_status=?, updated_at=? WHERE id=?",
                (RUNNING, RUNNING, _now(), episode_id),
            )

    def mark_download_done(self, episode_id: int, audio_path: Path) -> None:
        sha, size, mtime = file_stats(audio_path)
        now = _now()
        with self._tx():
            # Check if audio hash changed vs previously completed transcription
            row = self._conn.execute(
                "SELECT audio_sha256, transcription_status FROM episodes WHERE id=?",
                (episode_id,),
            ).fetchone()
            transcription_status = row["transcription_status"] if row else PENDING
            old_sha = row["audio_sha256"] if row else ""
            if old_sha and old_sha != sha and transcription_status == DONE:
                transcription_status = STALE

            self._conn.execute(
                """UPDATE episodes SET
                   download_status=?, audio_sha256=?, audio_size_bytes=?, audio_mtime=?,
                   transcription_status=?, updated_at=?
                   WHERE id=?""",
                (DONE, sha, size, mtime, transcription_status, now, episode_id),
            )
            self._update_overall(episode_id)

    def mark_download_failed(self, episode_id: int, error: str) -> None:
        with self._tx():
            self._conn.execute(
                """UPDATE episodes SET download_status=?, overall_status=?, last_error=?,
                   retry_count=retry_count+1, updated_at=? WHERE id=?""",
                (FAILED, FAILED, error, _now(), episode_id),
            )

    # ── Transcription tracking ────────────────────────────────────────────────

    def mark_transcription_running(self, episode_id: int) -> None:
        with self._tx():
            self._conn.execute(
                "UPDATE episodes SET transcription_status=?, overall_status=?, updated_at=? WHERE id=?",
                (RUNNING, RUNNING, _now(), episode_id),
            )

    def mark_transcription_done(
        self,
        episode_id: int,
        txt_path: Optional[Path] = None,
        srt_path: Optional[Path] = None,
    ) -> None:
        txt_sha = sha256_file(txt_path) if txt_path and txt_path.exists() else ""
        srt_sha = sha256_file(srt_path) if srt_path and srt_path.exists() else ""
        with self._tx():
            self._conn.execute(
                """UPDATE episodes SET transcription_status=?,
                   transcript_txt_sha256=?, transcript_srt_sha256=?,
                   updated_at=? WHERE id=?""",
                (DONE, txt_sha, srt_sha, _now(), episode_id),
            )
            self._update_overall(episode_id)

    def mark_transcription_failed(self, episode_id: int, error: str) -> None:
        with self._tx():
            self._conn.execute(
                """UPDATE episodes SET transcription_status=?, overall_status=?,
                   last_error=?, retry_count=retry_count+1, updated_at=? WHERE id=?""",
                (FAILED, FAILED, error, _now(), episode_id),
            )

    # ── Metadata / NFO tracking ───────────────────────────────────────────────

    def mark_metadata_done(self, episode_id: int, json_path: Optional[Path] = None) -> None:
        sha = sha256_file(json_path) if json_path and json_path.exists() else ""
        with self._tx():
            self._conn.execute(
                "UPDATE episodes SET metadata_status=?, metadata_json_sha256=?, updated_at=? WHERE id=?",
                (DONE, sha, _now(), episode_id),
            )
            self._update_overall(episode_id)

    def mark_nfo_done(self, episode_id: int, nfo_path: Optional[Path] = None) -> None:
        sha = sha256_file(nfo_path) if nfo_path and nfo_path.exists() else ""
        with self._tx():
            self._conn.execute(
                "UPDATE episodes SET nfo_status=?, nfo_sha256=?, updated_at=? WHERE id=?",
                (DONE, sha, _now(), episode_id),
            )
            self._update_overall(episode_id)

    # ── Generic ───────────────────────────────────────────────────────────────

    def mark_failed(self, episode_id: int, error: str) -> None:
        with self._tx():
            self._conn.execute(
                """UPDATE episodes SET overall_status=?, last_error=?,
                   retry_count=retry_count+1, updated_at=? WHERE id=?""",
                (FAILED, error, _now(), episode_id),
            )

    def mark_partial(self, episode_id: int) -> None:
        with self._tx():
            self._conn.execute(
                "UPDATE episodes SET overall_status=?, updated_at=? WHERE id=?",
                (PARTIAL, _now(), episode_id),
            )

    def mark_stale(self, episode_id: int) -> None:
        with self._tx():
            self._conn.execute(
                "UPDATE episodes SET overall_status=?, updated_at=? WHERE id=?",
                (STALE, _now(), episode_id),
            )

    def mark_completed(self, episode_id: int) -> None:
        with self._tx():
            self._conn.execute(
                "UPDATE episodes SET overall_status=?, completed_at=?, updated_at=? WHERE id=?",
                (DONE, _now(), _now(), episode_id),
            )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _update_overall(self, episode_id: int) -> None:
        """Derive overall_status from individual step statuses."""
        row = self._conn.execute(
            "SELECT download_status, transcription_status, metadata_status, nfo_status FROM episodes WHERE id=?",
            (episode_id,),
        ).fetchone()
        if not row:
            return
        statuses = [row["download_status"], row["transcription_status"],
                    row["metadata_status"], row["nfo_status"]]
        if any(s == FAILED for s in statuses):
            overall = FAILED
        elif any(s == STALE for s in statuses):
            overall = STALE
        elif any(s == RUNNING for s in statuses):
            overall = RUNNING
        elif all(s == DONE for s in statuses):
            overall = DONE
        elif any(s in (DONE, PARTIAL) for s in statuses):
            overall = PARTIAL
        else:
            overall = PENDING
        completed_at = _now() if overall == DONE else None
        self._conn.execute(
            "UPDATE episodes SET overall_status=?, completed_at=COALESCE(completed_at, ?), updated_at=? WHERE id=?",
            (overall, completed_at, _now(), episode_id),
        )

    # ── Skip / resumability logic ─────────────────────────────────────────────

    def should_skip_download(self, episode_id: int, audio_path: Path) -> bool:
        """True if audio file exists with matching hash and download is done."""
        row = self._conn.execute(
            "SELECT download_status, audio_sha256, audio_size_bytes FROM episodes WHERE id=?",
            (episode_id,),
        ).fetchone()
        if not row or row["download_status"] != DONE:
            return False
        if not audio_path.exists():
            return False
        if row["audio_sha256"] and row["audio_size_bytes"]:
            # Fast check: size match first, then full hash
            try:
                if audio_path.stat().st_size != row["audio_size_bytes"]:
                    return False
            except OSError:
                return False
            current_sha = sha256_file(audio_path)
            return current_sha == row["audio_sha256"]
        return audio_path.exists()

    def should_skip_transcription(
        self,
        episode_id: int,
        txt_path: Path,
        force: bool = False,
    ) -> bool:
        """True if transcription is done, txt exists and hashes match."""
        if force:
            return False
        row = self._conn.execute(
            "SELECT transcription_status, transcript_txt_sha256, audio_sha256 FROM episodes WHERE id=?",
            (episode_id,),
        ).fetchone()
        if not row or row["transcription_status"] != DONE:
            return False
        if not txt_path.exists():
            return False
        stored_sha = row["transcript_txt_sha256"]
        if stored_sha:
            return sha256_file(txt_path) == stored_sha
        return True

    # ── Integrity verification ────────────────────────────────────────────────

    def verify_artifacts(self, episode_id: int, *, full_hash: bool = False) -> dict:
        """
        Check all tracked artifacts. Returns dict with status per artifact.
        If full_hash=True, always recompute SHA256. Otherwise use mtime/size heuristic.
        """
        row = self._conn.execute("SELECT * FROM episodes WHERE id=?", (episode_id,)).fetchone()
        if not row:
            return {"error": "not found"}

        results = {}

        def _check(label: str, path_str: str, stored_sha: str) -> str:
            if not path_str:
                return "no_path"
            p = Path(path_str)
            if not p.exists():
                return "missing"
            if not stored_sha:
                return "no_hash"
            if full_hash:
                return "ok" if sha256_file(p) == stored_sha else "hash_mismatch"
            return "ok"

        results["audio"] = _check("audio", row["audio_path"], row["audio_sha256"])
        results["transcript_txt"] = _check("txt", row["transcript_txt_path"], row["transcript_txt_sha256"])
        results["transcript_srt"] = _check("srt", row["transcript_srt_path"], row["transcript_srt_sha256"])
        results["metadata_json"] = _check("json", row["metadata_json_path"], row["metadata_json_sha256"])
        results["nfo"] = _check("nfo", row["nfo_path"], row["nfo_sha256"])
        return results

    def is_episode_complete(self, episode_id: int) -> bool:
        row = self._conn.execute(
            "SELECT overall_status FROM episodes WHERE id=?", (episode_id,)
        ).fetchone()
        return row is not None and row["overall_status"] == DONE

    # ── Query / status ────────────────────────────────────────────────────────

    def get_record(self, episode_id: int) -> Optional[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM episodes WHERE id=?", (episode_id,)
        ).fetchone()

    def list_all(self) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM episodes ORDER BY feed_slug, episode_pub_date DESC"
        ).fetchall()

    def list_by_status(self, *statuses: str) -> list[sqlite3.Row]:
        placeholders = ",".join("?" * len(statuses))
        return self._conn.execute(
            f"SELECT * FROM episodes WHERE overall_status IN ({placeholders}) "
            "ORDER BY feed_slug, episode_pub_date DESC",
            statuses,
        ).fetchall()

    def list_failed_or_partial(self) -> list[sqlite3.Row]:
        return self.list_by_status(FAILED, PARTIAL, STALE)

    # ── Stale detection ───────────────────────────────────────────────────────

    def detect_stale(self, episode_id: int) -> None:
        """
        Recheck files on disk. Mark stale/partial if artifacts are missing
        or audio hash changed vs stored transcription.
        """
        row = self._conn.execute("SELECT * FROM episodes WHERE id=?", (episode_id,)).fetchone()
        if not row:
            return

        audio_p = Path(row["audio_path"]) if row["audio_path"] else None
        txt_p = Path(row["transcript_txt_path"]) if row["transcript_txt_path"] else None

        # If transcription was done but txt is now missing → partial
        if row["transcription_status"] == DONE and txt_p and not txt_p.is_file():
            self.mark_partial(episode_id)
            with self._tx():
                self._conn.execute(
                    "UPDATE episodes SET transcription_status=?, updated_at=? WHERE id=?",
                    (PARTIAL, _now(), episode_id),
                )
            return

        # If audio changed since transcription → stale
        if (row["transcription_status"] == DONE
                and audio_p and audio_p.exists()
                and row["audio_sha256"]):
            current = sha256_file(audio_p)
            if current != row["audio_sha256"]:
                with self._tx():
                    self._conn.execute(
                        "UPDATE episodes SET transcription_status=?, updated_at=? WHERE id=?",
                        (STALE, _now(), episode_id),
                    )
                    self._update_overall(episode_id)


def print_status(db: StateDB) -> None:
    """Print a human-readable grouped status overview."""
    rows = db.list_all()
    if not rows:
        print("No episodes tracked yet.")
        return

    by_feed: dict[str, list] = {}
    for r in rows:
        by_feed.setdefault(r["feed_slug"] or r["feed_url"], []).append(r)

    status_icons = {
        DONE: "✓", FAILED: "✗", PARTIAL: "~", STALE: "↻",
        RUNNING: "▶", PENDING: "○",
    }
    counts = {s: 0 for s in (DONE, FAILED, PARTIAL, STALE, RUNNING, PENDING)}

    for feed, episodes in by_feed.items():
        print(f"\n{feed}")
        for r in episodes:
            icon = status_icons.get(r["overall_status"], "?")
            model = r["model"] or r["pipeline_mode"] or "?"
            date = r["episode_pub_date"][:10] if r["episode_pub_date"] else "----"
            title = r["episode_title"][:55]
            err = f"  ← {r['last_error'][:60]}" if r["last_error"] and r["overall_status"] == FAILED else ""
            print(f"  {icon} [{model}] {date}  {title}{err}")
            counts[r["overall_status"]] = counts.get(r["overall_status"], 0) + 1

    print(f"\nTotal: {len(rows)}  ", end="")
    parts = [f"{v} {k}" for k, v in counts.items() if v > 0]
    print("  ".join(parts))


def print_verify(db: StateDB, full_hash: bool = False) -> None:
    """Verify all tracked artifacts and print results."""
    rows = db.list_all()
    if not rows:
        print("No episodes tracked.")
        return

    issues = 0
    for r in rows:
        results = db.verify_artifacts(r["id"], full_hash=full_hash)
        bad = {k: v for k, v in results.items() if v not in ("ok", "no_path", "no_hash")}
        if bad:
            issues += 1
            print(f"  [issue] {r['feed_slug']} / {r['episode_title'][:50]}")
            for k, v in bad.items():
                print(f"    {k}: {v}")

    if issues == 0:
        print(f"All {len(rows)} episodes verified OK.")
    else:
        print(f"\n{issues}/{len(rows)} episodes have integrity issues.")
