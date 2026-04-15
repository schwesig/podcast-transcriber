import re
from dataclasses import dataclass, field
from email.utils import parsedate
from pathlib import Path
from typing import Optional

import feedparser


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


@dataclass
class FeedConfig:
    url: str
    model: str = "small"
    language: Optional[str] = None
    pipeline: Optional[str] = None


@dataclass
class Episode:
    title: str
    guid: str
    audio_url: str
    pub_date: str = ""
    episode_number: Optional[str] = None
    duration: Optional[str] = None
    summary: str = ""
    shownotes: str = ""

    @property
    def slug(self) -> str:
        return slugify(self.title)

    @property
    def dated_slug(self) -> str:
        try:
            t = parsedate(self.pub_date)
            if t:
                return f"{t[0]}-{t[1]:02}-{t[2]:02}_{self.slug}"
        except Exception:
            pass
        return self.slug


@dataclass
class ParsedFeed:
    title: str
    language: Optional[str]
    episodes: list[Episode] = field(default_factory=list)

    @property
    def slug(self) -> str:
        return slugify(self.title)


def parse_feeds_file(path: Path) -> list[FeedConfig]:
    configs = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        url = parts[0]
        kwargs: dict = {"model": "small", "language": None, "pipeline": None}
        for part in parts[1:]:
            if "=" in part:
                k, v = part.split("=", 1)
                if k == "model":
                    kwargs["model"] = v
                elif k == "language":
                    kwargs["language"] = v
                elif k == "pipeline":
                    kwargs["pipeline"] = v
        configs.append(FeedConfig(url=url, **kwargs))
    return configs


def parse_rss(url_or_path: str) -> ParsedFeed:
    d = feedparser.parse(url_or_path)
    title = d.feed.get("title", "Unknown Podcast")
    language = d.feed.get("language", None)
    episodes = []
    for entry in d.entries:
        audio_url = ""
        for link in entry.get("links", []):
            if link.get("type", "").startswith("audio"):
                audio_url = link.get("href", "")
                break
        if not audio_url:
            enclosures = entry.get("enclosures", [])
            if enclosures:
                audio_url = enclosures[0].get("href", "")
        if not audio_url:
            continue
        # duration: itunes:duration or enclosure length
        duration = entry.get("itunes_duration", None)
        if not duration:
            enclosures = entry.get("enclosures", [])
            if enclosures and enclosures[0].get("length"):
                duration = enclosures[0]["length"]

        # shownotes: content:encoded > summary > description
        shownotes = ""
        for content in entry.get("content", []):
            if content.get("value"):
                shownotes = content["value"]
                break
        if not shownotes:
            shownotes = entry.get("summary", "")

        episodes.append(Episode(
            title=entry.get("title", "Untitled"),
            guid=entry.get("id", entry.get("title", "")),
            audio_url=audio_url,
            pub_date=entry.get("published", ""),
            episode_number=entry.get("itunes_episode", None),
            duration=duration,
            summary=entry.get("itunes_summary", entry.get("summary", "")),
            shownotes=shownotes,
        ))
    return ParsedFeed(title=title, language=language, episodes=episodes)
