"""Collect conflict-related news metadata from Google News RSS.

This is a lightweight fallback when GDELT rate-limits repeated historical
queries. It records article metadata only: title, publication date, source,
URL, region, and query.
"""

from __future__ import annotations

import json
import os
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"


def load_json(path: Path) -> dict | list:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def iter_windows(start_date: date, end_date: date, days: int = 45):
    current = start_date
    while current <= end_date:
        window_end = min(current + timedelta(days=days - 1), end_date)
        yield current, window_end
        current = window_end + timedelta(days=1)


def build_search_query(region_terms: list[str], conflict_terms: list[str], start_date: date, end_date: date) -> str:
    region_query = " OR ".join(f'"{term}"' for term in region_terms[:3])
    conflict_query = " OR ".join(conflict_terms)
    return f"({region_query}) ({conflict_query}) after:{start_date:%Y-%m-%d} before:{end_date:%Y-%m-%d}"


def batch_keywords(keywords: list[str], group_size: int = 6) -> list[list[str]]:
    """Split a long keyword list into smaller groups that fit Google News RSS query limits."""
    return [keywords[i : i + group_size] for i in range(0, len(keywords), group_size)]


def fetch_rss(query: str) -> list[dict]:
    url = f"{GOOGLE_NEWS_RSS}?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    root = ET.fromstring(response.content)
    rows: list[dict] = []
    for item in root.findall("./channel/item"):
        source = item.find("source")
        rows.append(
            {
                "title": item.findtext("title"),
                "publication_date": item.findtext("pubDate"),
                "source": source.text if source is not None else None,
                "domain": source.attrib.get("url") if source is not None else None,
                "url": item.findtext("link"),
                "language": "English",
                "query": query,
            }
        )
    return rows


def collect_region_news(region: dict, config: dict) -> pd.DataFrame:
    start_date = parse_date(os.getenv("PROJECT_START_DATE", config["start_date"]))
    end_date = parse_date(os.getenv("PROJECT_END_DATE", config["end_date"]))
    keyword_batches = batch_keywords(config["news_keywords"], group_size=6)
    rows: list[dict] = []

    for window_start, window_end in iter_windows(start_date, end_date):
        for batch_index, conflict_terms in enumerate(keyword_batches, start=1):
            query = build_search_query(region["news_terms"], conflict_terms, window_start, window_end)
            print(
                f"Downloading Google News RSS for {region['name']} | {window_start} to {window_end} "
                f"| batch {batch_index}/{len(keyword_batches)}",
                flush=True,
            )
            try:
                articles = fetch_rss(query)
            except requests.RequestException as error:
                print(f"  RSS request error: {error}", flush=True)
                continue
            except ET.ParseError as error:
                print(f"  RSS parse error: {error}", flush=True)
                continue

            for article in articles:
                article["region"] = region["name"]
                rows.append(article)

            time.sleep(1.5)

    return pd.DataFrame(rows)


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    config = load_json(CONFIG_DIR / "project_config.json")
    regions_file = os.getenv("REGIONS_FILE", "regions.json")
    regions = load_json(CONFIG_DIR / regions_file)

    # Optional: override news_keywords with a different keyword file
    keywords_file = os.getenv("KEYWORDS_FILE")
    if keywords_file:
        keywords_payload = load_json(CONFIG_DIR / keywords_file)
        if isinstance(keywords_payload, dict):
            keyword_field = os.getenv("KEYWORDS_FIELD", "news_keywords")
            combined: list[str] = []
            for field in keyword_field.split(","):
                value = keywords_payload.get(field.strip())
                if isinstance(value, list):
                    combined.extend(value)
            config = {**config, "news_keywords": combined or config["news_keywords"]}
        elif isinstance(keywords_payload, list):
            config = {**config, "news_keywords": keywords_payload}

    frames = [collect_region_news(region, config) for region in regions]
    frames = [frame for frame in frames if not frame.empty]

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_name = os.getenv("NEWS_OUTPUT_NAME", "conflict_news_google_rss_raw.csv")
    output_path = RAW_DATA_DIR / output_name

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined.drop_duplicates(subset=["url"], inplace=True)
        combined.to_csv(output_path, index=False)
        print(f"Saved {len(combined):,} RSS news rows to {output_path}", flush=True)
    else:
        print("No RSS news rows were downloaded.", flush=True)


if __name__ == "__main__":
    main()
