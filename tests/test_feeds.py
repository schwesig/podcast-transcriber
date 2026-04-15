# tests/test_feeds.py
from pathlib import Path
from src.feeds import FeedConfig, parse_feeds_file, parse_rss, slugify

FIXTURE_XML = Path("tests/fixtures/sample_feed.xml")


def test_slugify_basic():
    assert slugify("Test Podcast: Hello World!") == "test-podcast-hello-world"


def test_slugify_german():
    assert slugify("Macht & Wechsel") == "macht-wechsel"


def test_parse_feeds_file(tmp_path):
    feeds_file = tmp_path / "feeds.txt"
    feeds_file.write_text(
        "https://example.com/feed1.xml model=small language=de\n"
        "https://example.com/feed2.xml\n"
        "# this is a comment\n"
        "\n"
    )
    configs = parse_feeds_file(feeds_file)
    assert len(configs) == 2
    assert configs[0].url == "https://example.com/feed1.xml"
    assert configs[0].model == "small"
    assert configs[0].language == "de"
    assert configs[1].url == "https://example.com/feed2.xml"
    assert configs[1].model == "small"   # default
    assert configs[1].language is None


def test_parse_rss_title():
    feed = parse_rss(str(FIXTURE_XML))
    assert feed.title == "Test Podcast"


def test_parse_rss_language():
    feed = parse_rss(str(FIXTURE_XML))
    assert feed.language == "de"


def test_parse_rss_episodes():
    feed = parse_rss(str(FIXTURE_XML))
    assert len(feed.episodes) == 2
    ep = feed.episodes[0]
    assert ep.title == "Episode 1: Pilot"
    assert ep.guid == "ep-001"
    assert ep.audio_url == "https://example.com/ep1.mp3"


def test_parse_rss_episode_slug():
    feed = parse_rss(str(FIXTURE_XML))
    assert feed.episodes[0].slug == "episode-1-pilot"


def test_parse_feeds_file_pipeline(tmp_path):
    feeds_file = tmp_path / "feeds.txt"
    feeds_file.write_text(
        "https://example.com/feed1.xml model=small language=de pipeline=full\n"
        "https://example.com/feed2.xml pipeline=fast\n"
        "https://example.com/feed3.xml model=small language=en\n"
    )
    configs = parse_feeds_file(feeds_file)
    assert configs[0].pipeline == "full"
    assert configs[1].pipeline == "fast"
    assert configs[2].pipeline is None
