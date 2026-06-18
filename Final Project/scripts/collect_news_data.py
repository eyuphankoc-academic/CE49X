"""Collect conflict-related news metadata from the free GDELT Doc API."""

from __future__ import annotations

import json
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
REQUEST_PAUSE_SECONDS = 5
MAX_RETRIES = 4


def load_json(path: Path) -> dict | list:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def gdelt_timestamp(value: date) -> str:
    return f"{value:%Y%m%d}000000"


def build_query(region_terms: list[str], conflict_terms: list[str]) -> str:
    region_query = " OR ".join(f'"{term}"' for term in region_terms)
    conflict_query = " OR ".join(conflict_terms)
    return f"({region_query}) ({conflict_query})"


def fetch_gdelt_articles(query: str, start_date: date, end_date: date, max_records: int = 100) -> list[dict]:
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": max_records,
        "sort": "hybridrel",
        "startdatetime": gdelt_timestamp(start_date),
        "enddatetime": gdelt_timestamp(end_date),
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(GDELT_DOC_API, params=params, timeout=90)
        except requests.RequestException as error:
            wait_seconds = REQUEST_PAUSE_SECONDS * attempt
            print(f"  Request error: {error}. Waiting {wait_seconds} seconds.", flush=True)
            time.sleep(wait_seconds)
            continue

        if response.status_code == 429:
            wait_seconds = REQUEST_PAUSE_SECONDS * attempt * 3
            print(f"  Rate limited by GDELT. Waiting {wait_seconds} seconds.", flush=True)
            time.sleep(wait_seconds)
            continue

        try:
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as error:
            wait_seconds = REQUEST_PAUSE_SECONDS * attempt
            print(f"  GDELT response error: {error}. Waiting {wait_seconds} seconds.", flush=True)
            time.sleep(wait_seconds)
            continue

        return payload.get("articles", [])

    print("  Skipping window after repeated GDELT errors.", flush=True)
    return []


def collect_region_news(region: dict, config: dict) -> pd.DataFrame:
    start_date = parse_date(os.getenv("PROJECT_START_DATE", config["start_date"]))
    end_date = parse_date(os.getenv("PROJECT_END_DATE", config["end_date"]))
    keywords = config["news_keywords"]
    query = build_query(region["news_terms"], keywords)
    rows: list[dict] = []

    current = start_date
    while current <= end_date:
        window_end = min(current + timedelta(days=30), end_date)
        print(f"Downloading GDELT news for {region['name']} | {current} to {window_end}", flush=True)
        articles = fetch_gdelt_articles(query, current, window_end)

        for article in articles:
            rows.append(
                {
                    "region": region["name"],
                    "title": article.get("title"),
                    "publication_date": article.get("seendate"),
                    "source": article.get("sourceCountry"),
                    "domain": article.get("domain"),
                    "url": article.get("url"),
                    "language": article.get("language"),
                    "query": query,
                }
            )

        time.sleep(REQUEST_PAUSE_SECONDS)
        current = window_end + timedelta(days=1)

    return pd.DataFrame(rows)


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    config = load_json(CONFIG_DIR / "project_config.json")
    regions = load_json(CONFIG_DIR / "regions.json")

    frames = [collect_region_news(region, config) for region in regions]
    frames = [frame for frame in frames if not frame.empty]

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_name = os.getenv("NEWS_OUTPUT_NAME", "conflict_news_raw.csv")
    output_path = RAW_DATA_DIR / output_name

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined.drop_duplicates(subset=["url"], inplace=True)
        combined.to_csv(output_path, index=False)
        print(f"Saved {len(combined):,} news rows to {output_path}", flush=True)
    else:
        print("No GDELT news rows were downloaded.", flush=True)


if __name__ == "__main__":
    main()
