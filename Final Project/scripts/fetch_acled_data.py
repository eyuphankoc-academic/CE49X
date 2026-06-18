"""Fetch ACLED conflict events for all project conflict regions.

Register at https://developer.acleddata.com/ and add to .env:
    ACLED_API_KEY=your_key_here
    ACLED_EMAIL=your_registered_email

Usage:
    python scripts/fetch_acled_data.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

ACLED_URL = "https://api.acleddata.com/acled/read"
PAGE_SIZE = 5000
SLEEP_BETWEEN_REQUESTS = 0.5

# Countries to query per project region.
# ACLED uses country name strings exactly as in their database.
REGION_COUNTRIES: dict[str, list[str]] = {
    "Ukraine_Black_Sea":       ["Ukraine"],
    "Red_Sea_Yemen":           ["Yemen"],
    "Gaza_Israel_Lebanon":     ["Israel", "Palestine", "Lebanon"],
    "Persian_Gulf_Iraq_Iran":  ["Iraq", "Iran"],
    "Sudan_Red_Sea_Corridor":  ["Sudan"],
    "Libya_Mediterranean":     ["Libya"],
}

# Event types we care about — the ones that produce thermal anomalies
RELEVANT_EVENT_TYPES = [
    "Battles",
    "Explosions/Remote violence",
    "Violence against civilians",
    "Strategic developments",
]


def fetch_country(api_key: str, email: str, country: str,
                  start_date: str, end_date: str) -> list[dict]:
    """Page through ACLED API results for one country + date range."""
    all_events: list[dict] = []
    offset = 0

    while True:
        params = {
            "key": api_key,
            "email": email,
            "country": country,
            "event_date": f"{start_date}|{end_date}",
            "event_date_where": "BETWEEN",
            "limit": PAGE_SIZE,
            "offset": offset,
        }
        resp = requests.get(ACLED_URL, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()

        batch = payload.get("data", [])
        all_events.extend(batch)
        fetched = len(all_events)
        print(f"    {country}: offset={offset:,} -> got {len(batch)} events (running total: {fetched:,})")

        if len(batch) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    return all_events


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("ACLED_API_KEY", "").strip()
    email   = os.getenv("ACLED_EMAIL", "").strip()

    if not api_key or api_key == "replace_with_your_acled_api_key":
        raise SystemExit(
            "\nACLED credentials missing.\n"
            "Add to .env:\n"
            "  ACLED_API_KEY=your_key_here\n"
            "  ACLED_EMAIL=your_registered_email\n"
            "Register free at: https://developer.acleddata.com/"
        )

    start_date = os.getenv("PROJECT_START_DATE", "2024-01-01")
    end_date   = "2026-05-26"

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_records: list[dict] = []

    for region, countries in REGION_COUNTRIES.items():
        print(f"\nFetching ACLED events for region: {region}")
        for country in countries:
            events = fetch_country(api_key, email, country, start_date, end_date)
            for ev in events:
                ev["project_region"] = region
            all_records.extend(events)

    if not all_records:
        print("\nNo events fetched. Check your ACLED_API_KEY and ACLED_EMAIL in .env.")
        return

    df = pd.DataFrame(all_records)
    if "event_date" in df.columns:
        df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    for col in ("latitude", "longitude", "fatalities"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    out_path = RAW_DATA_DIR / "acled_events_raw.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df):,} ACLED events -> {out_path}")

    if "event_type" in df.columns and "project_region" in df.columns:
        print("\nEvents by region + type:")
        summary = (
            df.groupby(["project_region", "event_type"])
            .size()
            .rename("count")
            .reset_index()
            .sort_values(["project_region", "count"], ascending=[True, False])
        )
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
