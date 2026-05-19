#!/usr/bin/env python3
"""
Fetch all Talk ohne Gast episodes via fyyd API and build a podcast RSS feed.

Usage:
  python scrape_fyyd_tog.py [--output tog_fyyd.xml] [--podcast-id 53875]
"""
import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

import requests


FYYD_API = "https://api.fyyd.de/0.2/podcast/episodes"
PODCAST_ID = 53875
PAGE_SIZE = 50


def fetch_all_episodes(podcast_id: int) -> list[dict]:
    episodes = []
    page = 0

    while True:
        print(f"\r  Fetching page {page} ({len(episodes)} eps)...", end="", flush=True)
        r = requests.get(FYYD_API, params={
            "podcast_id": podcast_id,
            "count": PAGE_SIZE,
            "page": page,
        }, timeout=30)
        r.raise_for_status()
        data = r.json()

        if data.get("status") != 1:
            print(f"\nAPI error: {data.get('msg')}", file=sys.stderr)
            break

        batch = data["data"].get("episodes", [])
        if not batch:
            break

        episodes.extend(batch)

        meta = data["meta"]["paging"]
        if page >= meta["last_page"]:
            break
        page += 1
        time.sleep(0.3)  # be polite

    print(f"\r  Done: {len(episodes)} episodes fetched.          ")
    return episodes


def fmt_rfc2822(iso_date: str) -> str:
    """Convert ISO 8601 to RFC 2822 for RSS pubDate."""
    try:
        dt = datetime.fromisoformat(iso_date)
        return dt.strftime("%a, %d %b %Y %H:%M:%S %z")
    except Exception:
        return iso_date


def build_rss(episodes: list[dict], output_path: Path) -> None:
    now = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S +0000")

    items = []
    for ep in episodes:
        title = escape(ep.get("title", ""))
        desc = escape(ep.get("description", "") or ep.get("title", ""))
        mp3 = ep.get("enclosure") or ep.get("url", "")
        pub = fmt_rfc2822(ep.get("pubdate", "")) if ep.get("pubdate") else now
        guid = f"fyyd-{ep['id']}"
        link = ep.get("url_fyyd", f"https://fyyd.de/episode/{ep['id']}")
        duration = ep.get("duration_string", "")
        img = ep.get("imgURL", "")

        enclosure = f'<enclosure url="{escape(mp3)}" type="{escape(ep.get("content_type","audio/mpeg"))}"/>' if mp3 else ""
        itunes_dur = f"<itunes:duration>{escape(duration)}</itunes:duration>" if duration else ""
        itunes_img = f'<itunes:image href="{escape(img)}"/>' if img else ""

        items.append(f"""    <item>
      <title>{title}</title>
      <description>{desc}</description>
      <pubDate>{pub}</pubDate>
      <guid isPermaLink="false">{guid}</guid>
      <link>{link}</link>
      {enclosure}
      {itunes_dur}
      {itunes_img}
    </item>""")

    rss = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"
  xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd"
  xmlns:content="http://purl.org/rss/1.0/modules/content/">
  <channel>
    <title>Talk ohne Gast</title>
    <description>Talk ohne Gast mit Till Reiners und Moritz Neumeier</description>
    <link>https://www.fritz.de/programm/podcasts/talk-ohne-gast.html</link>
    <language>de</language>
    <lastBuildDate>{now}</lastBuildDate>
    <itunes:author>Till Reiners, Moritz Neumeier</itunes:author>
    <itunes:explicit>no</itunes:explicit>
{chr(10).join(items)}
  </channel>
</rss>"""

    output_path.write_text(rss, encoding="utf-8")
    print(f"RSS saved: {output_path} ({len(episodes)} episodes)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="tog_fyyd.xml")
    parser.add_argument("--podcast-id", type=int, default=PODCAST_ID)
    args = parser.parse_args()

    episodes = fetch_all_episodes(args.podcast_id)

    if not episodes:
        print("No episodes found.", file=sys.stderr)
        return 1

    build_rss(episodes, Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
