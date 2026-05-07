"""Tests for src/state_db.py"""
import pytest
from pathlib import Path

from src.state_db import (
    StateDB,
    PENDING, RUNNING, DONE, FAILED, PARTIAL, STALE,
    sha256_file, atomic_write_text,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def db(tmp_path):
    return StateDB(tmp_path / "state.sqlite")


def _ep_kwargs(**overrides):
    base = dict(
        feed_url="https://example.com/feed",
        feed_title="Test Podcast",
        feed_slug="test-podcast",
        episode_title="Episode 1",
        episode_slug="episode-1",
        episode_guid="guid-ep1",
        episode_audio_url="https://example.com/ep1.mp3",
        episode_pub_date="2024-01-01",
        model="small",
        language="de",
        pipeline_mode="",
        output_dir="/tmp/podcasts",
        episode_dir="/tmp/podcasts/test-podcast/episode-1",
        audio_path="/tmp/podcasts/test-podcast/episode-1/2024-01-01_episode-1.mp3",
        transcript_txt_path="/tmp/podcasts/test-podcast/episode-1/2024-01-01_episode-1.txt",
        transcript_srt_path="/tmp/podcasts/test-podcast/episode-1/2024-01-01_episode-1.srt",
        metadata_json_path="/tmp/podcasts/test-podcast/episode-1/2024-01-01_episode-1.json",
        nfo_path="/tmp/podcasts/test-podcast/episode-1/2024-01-01_episode-1.nfo",
    )
    base.update(overrides)
    return base


# ── DB creation ───────────────────────────────────────────────────────────────

def test_db_created(tmp_path):
    db_path = tmp_path / "state.sqlite"
    db = StateDB(db_path)
    assert db_path.exists()


def test_db_survives_reopen(tmp_path):
    db_path = tmp_path / "state.sqlite"
    db1 = StateDB(db_path)
    eid = db1.get_or_create(**_ep_kwargs())
    db1.close()
    db2 = StateDB(db_path)
    row = db2.get_record(eid)
    assert row["episode_title"] == "Episode 1"


# ── Record creation ───────────────────────────────────────────────────────────

def test_get_or_create_returns_id(db):
    eid = db.get_or_create(**_ep_kwargs())
    assert isinstance(eid, int)
    assert eid > 0


def test_get_or_create_idempotent(db):
    eid1 = db.get_or_create(**_ep_kwargs())
    eid2 = db.get_or_create(**_ep_kwargs())
    assert eid1 == eid2


def test_initial_status_pending(db):
    eid = db.get_or_create(**_ep_kwargs())
    row = db.get_record(eid)
    assert row["overall_status"] == PENDING
    assert row["download_status"] == PENDING
    assert row["transcription_status"] == PENDING


# ── Download lifecycle ────────────────────────────────────────────────────────

def test_mark_download_running(db):
    eid = db.get_or_create(**_ep_kwargs())
    db.mark_download_running(eid)
    row = db.get_record(eid)
    assert row["download_status"] == RUNNING
    assert row["overall_status"] == RUNNING


def test_mark_download_done(db, tmp_path):
    eid = db.get_or_create(**_ep_kwargs())
    audio = tmp_path / "ep.mp3"
    audio.write_bytes(b"fakeaudio")
    db.mark_download_running(eid)
    db.mark_download_done(eid, audio)
    row = db.get_record(eid)
    assert row["download_status"] == DONE
    assert row["audio_sha256"] != ""
    assert row["audio_size_bytes"] == 9


def test_mark_download_failed(db):
    eid = db.get_or_create(**_ep_kwargs())
    db.mark_download_running(eid)
    db.mark_download_failed(eid, "connection refused")
    row = db.get_record(eid)
    assert row["download_status"] == FAILED
    assert row["overall_status"] == FAILED
    assert row["last_error"] == "connection refused"
    assert row["retry_count"] == 1


# ── Transcription lifecycle ───────────────────────────────────────────────────

def test_mark_transcription_done(db, tmp_path):
    eid = db.get_or_create(**_ep_kwargs())
    audio = tmp_path / "ep.mp3"
    audio.write_bytes(b"audio")
    txt = tmp_path / "ep.txt"
    txt.write_text("transcript")
    db.mark_download_done(eid, audio)
    db.mark_transcription_running(eid)
    db.mark_transcription_done(eid, txt)
    row = db.get_record(eid)
    assert row["transcription_status"] == DONE
    assert row["transcript_txt_sha256"] != ""


def test_transcription_failed_stores_error(db):
    eid = db.get_or_create(**_ep_kwargs())
    db.mark_transcription_running(eid)
    db.mark_transcription_failed(eid, "OOM")
    row = db.get_record(eid)
    assert row["transcription_status"] == FAILED
    assert row["last_error"] == "OOM"
    assert row["retry_count"] == 1


def test_transcription_failed_resumable(db, tmp_path):
    """After failure, should_skip_transcription returns False → can retry."""
    eid = db.get_or_create(**_ep_kwargs())
    txt = tmp_path / "ep.txt"
    db.mark_transcription_failed(eid, "crash")
    assert db.should_skip_transcription(eid, txt) is False


# ── Overall status derivation ─────────────────────────────────────────────────

def test_overall_done_when_all_done(db, tmp_path):
    eid = db.get_or_create(**_ep_kwargs())
    audio = tmp_path / "ep.mp3"
    audio.write_bytes(b"x")
    txt = tmp_path / "ep.txt"
    txt.write_text("t")
    db.mark_download_done(eid, audio)
    db.mark_transcription_done(eid, txt)
    db.mark_metadata_done(eid, None)
    db.mark_nfo_done(eid, None)
    assert db.is_episode_complete(eid)


def test_overall_partial_when_some_done(db, tmp_path):
    eid = db.get_or_create(**_ep_kwargs())
    audio = tmp_path / "ep.mp3"
    audio.write_bytes(b"x")
    db.mark_download_done(eid, audio)
    row = db.get_record(eid)
    assert row["overall_status"] == PARTIAL


# ── Skip logic ────────────────────────────────────────────────────────────────

def test_should_skip_download_after_done(db, tmp_path):
    eid = db.get_or_create(**_ep_kwargs())
    audio = tmp_path / "ep.mp3"
    audio.write_bytes(b"audio content")
    db.mark_download_done(eid, audio)
    assert db.should_skip_download(eid, audio) is True


def test_should_not_skip_download_if_file_missing(db, tmp_path):
    eid = db.get_or_create(**_ep_kwargs())
    audio = tmp_path / "ep.mp3"
    audio.write_bytes(b"audio content")
    db.mark_download_done(eid, audio)
    audio.unlink()
    assert db.should_skip_download(eid, audio) is False


def test_should_not_skip_download_if_file_changed(db, tmp_path):
    eid = db.get_or_create(**_ep_kwargs())
    audio = tmp_path / "ep.mp3"
    audio.write_bytes(b"original")
    db.mark_download_done(eid, audio)
    audio.write_bytes(b"different content")
    assert db.should_skip_download(eid, audio) is False


def test_should_skip_transcription_when_done(db, tmp_path):
    eid = db.get_or_create(**_ep_kwargs())
    audio = tmp_path / "ep.mp3"
    audio.write_bytes(b"x")
    txt = tmp_path / "ep.txt"
    txt.write_text("transcript")
    db.mark_download_done(eid, audio)
    db.mark_transcription_done(eid, txt)
    assert db.should_skip_transcription(eid, txt) is True


def test_force_transcribe_overrides_skip(db, tmp_path):
    eid = db.get_or_create(**_ep_kwargs())
    audio = tmp_path / "ep.mp3"
    audio.write_bytes(b"x")
    txt = tmp_path / "ep.txt"
    txt.write_text("transcript")
    db.mark_download_done(eid, audio)
    db.mark_transcription_done(eid, txt)
    assert db.should_skip_transcription(eid, txt, force=True) is False


# ── Different models → separate records ──────────────────────────────────────

def test_same_episode_different_models_separate_records(db):
    eid_small = db.get_or_create(**_ep_kwargs(model="small"))
    eid_large = db.get_or_create(**_ep_kwargs(model="large-v3"))
    assert eid_small != eid_large


def test_different_languages_separate_records(db):
    eid_de = db.get_or_create(**_ep_kwargs(language="de"))
    eid_en = db.get_or_create(**_ep_kwargs(language="en"))
    assert eid_de != eid_en


# ── Stale detection ───────────────────────────────────────────────────────────

def test_stale_when_audio_changes(db, tmp_path):
    audio = tmp_path / "ep.mp3"
    txt = tmp_path / "ep.txt"
    eid = db.get_or_create(**_ep_kwargs(
        audio_path=str(audio),
        transcript_txt_path=str(txt),
    ))
    audio.write_bytes(b"original audio")
    txt.write_text("transcript")
    db.mark_download_done(eid, audio)
    db.mark_transcription_done(eid, txt)

    # Now audio file changes content
    audio.write_bytes(b"completely different audio content")
    db.detect_stale(eid)
    row = db.get_record(eid)
    assert row["transcription_status"] == STALE


def test_partial_when_txt_missing(db, tmp_path):
    eid = db.get_or_create(**_ep_kwargs())
    audio = tmp_path / "ep.mp3"
    audio.write_bytes(b"audio")
    txt = tmp_path / "ep.txt"
    txt.write_text("transcript")
    db.mark_download_done(eid, audio)
    db.mark_transcription_done(eid, txt)
    txt.unlink()
    db.detect_stale(eid)
    row = db.get_record(eid)
    assert row["overall_status"] == PARTIAL


# ── Crash recovery ────────────────────────────────────────────────────────────

def test_running_becomes_partial_on_reopen(tmp_path):
    db_path = tmp_path / "state.sqlite"
    db1 = StateDB(db_path)
    eid = db1.get_or_create(**_ep_kwargs())
    db1.mark_download_running(eid)
    db1.close()

    # Simulate crash + reopen
    db2 = StateDB(db_path)
    row = db2.get_record(eid)
    assert row["download_status"] == PARTIAL  # not stuck as RUNNING
    assert row["overall_status"] == PARTIAL


# ── Hash verification ─────────────────────────────────────────────────────────

def test_verify_artifacts_missing(db, tmp_path):
    eid = db.get_or_create(**_ep_kwargs(
        audio_path=str(tmp_path / "missing.mp3"),
        transcript_txt_path=str(tmp_path / "missing.txt"),
    ))
    audio = tmp_path / "missing.mp3"
    audio.write_bytes(b"audio")
    db.mark_download_done(eid, audio)
    audio.unlink()
    results = db.verify_artifacts(eid, full_hash=True)
    assert results["audio"] == "missing"


def test_verify_artifacts_ok(db, tmp_path):
    eid = db.get_or_create(**_ep_kwargs(
        audio_path=str(tmp_path / "ep.mp3"),
    ))
    audio = tmp_path / "ep.mp3"
    audio.write_bytes(b"audio content")
    db.mark_download_done(eid, audio)
    results = db.verify_artifacts(eid, full_hash=True)
    assert results["audio"] == "ok"


# ── Atomic write ──────────────────────────────────────────────────────────────

def test_atomic_write_text(tmp_path):
    path = tmp_path / "out.txt"
    atomic_write_text(path, "hello world")
    assert path.read_text() == "hello world"


def test_atomic_write_text_creates_parents(tmp_path):
    path = tmp_path / "a" / "b" / "c.txt"
    atomic_write_text(path, "test")
    assert path.read_text() == "test"


# ── SHA256 ────────────────────────────────────────────────────────────────────

def test_sha256_file(tmp_path):
    p = tmp_path / "file.bin"
    p.write_bytes(b"hello")
    sha = sha256_file(p)
    assert len(sha) == 64  # hex sha256
    assert sha == sha256_file(p)


def test_sha256_missing_file(tmp_path):
    assert sha256_file(tmp_path / "nope.bin") == ""


# ── list_by_status ────────────────────────────────────────────────────────────

def test_list_failed(db):
    eid = db.get_or_create(**_ep_kwargs())
    db.mark_download_failed(eid, "error")
    rows = db.list_failed_or_partial()
    assert any(r["id"] == eid for r in rows)


def test_list_all_empty(db):
    assert db.list_all() == []


def test_list_all_returns_rows(db, tmp_path):
    db.get_or_create(**_ep_kwargs())
    db.get_or_create(**_ep_kwargs(episode_guid="guid2", model="large-v3"))
    assert len(db.list_all()) == 2
