"""Collect NASA FIRMS thermal anomaly data in 5-day chunks."""

from __future__ import annotations

import io
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
FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"


def load_json(path: Path) -> dict | list:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def iter_date_chunks(start_date: date, end_date: date, day_range: int):
    current = start_date
    while current <= end_date:
        remaining_days = (end_date - current).days + 1
        current_range = min(day_range, remaining_days)
        yield current, current_range
        current += timedelta(days=current_range)


def format_bbox(bbox: list[float]) -> str:
    return ",".join(str(value) for value in bbox)


def build_firms_url(map_key: str, source: str, area: str, day_range: int, start: date) -> str:
    return f"{FIRMS_BASE_URL}/{map_key}/{source}/{area}/{day_range}/{start:%Y-%m-%d}"


def fetch_firms_csv(url: str) -> pd.DataFrame:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    text = response.text.strip()

    if not text or text.lower().startswith(("invalid", "error")):
        return pd.DataFrame()

    return pd.read_csv(io.StringIO(text))


def collect_region(region: dict, config: dict, map_key: str) -> list[pd.DataFrame]:
    start_date = parse_date(os.getenv("PROJECT_START_DATE", config["start_date"]))
    end_date = parse_date(os.getenv("PROJECT_END_DATE", config["end_date"]))
    day_range = int(config["firms_day_range"])
    source = os.getenv("FIRMS_SOURCE", config["firms_source"])
    area = format_bbox(region["bbox"])
    frames: list[pd.DataFrame] = []

    for chunk_start, chunk_range in iter_date_chunks(start_date, end_date, day_range):
        print(f"Downloading {region['name']} | {chunk_start} | {chunk_range} days", flush=True)
        url = build_firms_url(map_key, source, area, chunk_range, chunk_start)
        try:
            frame = fetch_firms_csv(url)
        except requests.RequestException as error:
            print(f"  Request error: {error}", flush=True)
            continue

        if not frame.empty:
            frame["region"] = region["name"]
            frame["region_description"] = region["description"]
            frames.append(frame)

        time.sleep(0.3)

    return frames


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    map_key = os.getenv("FIRMS_MAP_KEY")
    if not map_key:
        raise ValueError("FIRMS_MAP_KEY is missing. Add it to the .env file.")

    config = load_json(CONFIG_DIR / "project_config.json")
    regions_file = os.getenv("REGIONS_FILE", "regions.json")
    regions = load_json(CONFIG_DIR / regions_file)

    all_frames: list[pd.DataFrame] = []
    for region in regions:
        all_frames.extend(collect_region(region, config, map_key))

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_name = os.getenv("FIRMS_OUTPUT_NAME", "firms_thermal_raw.csv")
    output_path = RAW_DATA_DIR / output_name

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined.drop_duplicates(inplace=True)
        combined.to_csv(output_path, index=False)
        print(f"Saved {len(combined):,} FIRMS rows to {output_path}", flush=True)
    else:
        print("No FIRMS rows were downloaded. Check source, dates, regions, or API key.", flush=True)


if __name__ == "__main__":
    main()
