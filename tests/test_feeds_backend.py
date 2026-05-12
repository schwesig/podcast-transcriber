"""Tests for backend= key in feeds.txt parsing."""
import textwrap
from pathlib import Path

import pytest

from src.feeds import parse_feeds_file


@pytest.fixture
def tmp_feeds(tmp_path):
    def _write(content: str) -> Path:
        p = tmp_path / "feeds.txt"
        p.write_text(textwrap.dedent(content))
        return p
    return _write


def test_default_backend_is_auto(tmp_feeds):
    p = tmp_feeds("https://example.com/feed.rss model=small language=de\n")
    configs = parse_feeds_file(p)
    assert configs[0].backend == "auto"


def test_explicit_mlx_backend(tmp_feeds):
    p = tmp_feeds("https://example.com/feed.rss model=small backend=mlx\n")
    configs = parse_feeds_file(p)
    assert configs[0].backend == "mlx"


def test_explicit_faster_whisper_backend(tmp_feeds):
    p = tmp_feeds("https://example.com/feed.rss backend=faster-whisper\n")
    configs = parse_feeds_file(p)
    assert configs[0].backend == "faster-whisper"


def test_backend_alongside_other_keys(tmp_feeds):
    p = tmp_feeds("https://example.com/feed.rss model=base language=en backend=mlx pipeline=full\n")
    configs = parse_feeds_file(p)
    fc = configs[0]
    assert fc.model == "base"
    assert fc.language == "en"
    assert fc.backend == "mlx"
    assert fc.pipeline == "full"
