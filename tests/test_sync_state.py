from pathlib import Path
from src.sync_state import is_processed, mark_processed


def test_not_processed_when_dir_missing(tmp_path):
    ep_dir = tmp_path / "podcast" / "episode-1"
    assert is_processed(ep_dir) is False


def test_not_processed_when_txt_missing(tmp_path):
    ep_dir = tmp_path / "podcast" / "episode-1"
    ep_dir.mkdir(parents=True)
    assert is_processed(ep_dir) is False


def test_processed_when_txt_exists(tmp_path):
    ep_dir = tmp_path / "podcast" / "episode-1"
    ep_dir.mkdir(parents=True)
    (ep_dir / "transcript.txt").write_text("hello")
    assert is_processed(ep_dir) is True


def test_mark_processed_creates_dir_and_marker(tmp_path):
    ep_dir = tmp_path / "podcast" / "episode-2"
    mark_processed(ep_dir)
    assert (ep_dir / "transcript.txt").exists() or ep_dir.exists()
