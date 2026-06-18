"""Continuously pull new GDELT v2 15-minute export files as they are published.

GDELT publishes a new file every 15 minutes.  This script polls
http://data.gdeltproject.org/gdeltv2/lastupdate.txt to detect new files,
downloads and filters them, then appends to both output CSVs.

Run this after (or alongside) fetch_gdelt_events.py to keep data current.
It runs forever -- Ctrl+C to stop.

Outputs appended to (created if missing):
  data/raw/gdelt_conflict_events_v2_raw.csv   -- all 18/19/20 events
  data/raw/gdelt_fire_relevant_events_v2.csv  -- 184/185/193 fire events only

Usage:
    python scripts/fetch_gdelt_live.py
    python scripts/fetch_gdelt_live.py --poll-interval 900
"""

from __future__ import annotations

import io
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parents[1]
RAW_DATA_DIR  = PROJECT_ROOT / "data" / "raw"
OUTPUT_ALL    = RAW_DATA_DIR / "gdelt_conflict_events_v2_raw.csv"
OUTPUT_FIRE   = RAW_DATA_DIR / "gdelt_fire_relevant_events_v2.csv"
CACHE_DIR     = RAW_DATA_DIR / "gdelt_cache_v2"
SEEN_FILE     = CACHE_DIR / "_live_seen.txt"   # tracks already-processed URLs

LASTUPDATE_URL = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"
TIMEOUT        = 60
POLL_INTERVAL  = 900   # seconds (15 minutes)

# ── GDELT v2 column indices (0-indexed) ───────────────────────────────────────
C_DATE      = 1
C_BASE_CODE = 27
C_ROOT_CODE = 28
C_COUNTRY   = 53
C_LAT       = 56
C_LON       = 57
C_ADDED     = 59
C_URL       = 60

CONFLICT_ROOT_CODES = {18, 19, 20}
FIRE_BASE_CODES     = {184, 185, 193}

FIPS_TO_REGION: dict[str, str] = {
    "UP": "Ukraine_Black_Sea",
    "RS": "Ukraine_Black_Sea",
    "YM": "Red_Sea_Yemen",
    "IS": "Gaza_Israel_Lebanon",
    "WE": "Gaza_Israel_Lebanon",
    "GZ": "Gaza_Israel_Lebanon",
    "LE": "Gaza_Israel_Lebanon",
    "IZ": "Persian_Gulf_Iraq_Iran",
    "IR": "Persian_Gulf_Iraq_Iran",
    "SU": "Sudan_Red_Sea_Corridor",
    "LY": "Libya_Mediterranean",
}


def load_seen() -> set[str]:
    if SEEN_FILE.exists():
        return set(SEEN_FILE.read_text(encoding="utf-8").splitlines())
    return set()


def mark_seen(url: str) -> None:
    with SEEN_FILE.open("a", encoding="utf-8") as f:
        f.write(url + "\n")


def latest_export_urls() -> list[str]:
    """Parse lastupdate.txt and return only .export.CSV.zip URLs."""
    try:
        resp = requests.get(LASTUPDATE_URL, timeout=TIMEOUT)
        resp.raise_for_status()
    except Exception as exc:
        print(f"  [poll] lastupdate.txt fetch failed: {exc}", flush=True)
        return []
    urls = []
    for line in resp.text.splitlines():
        parts = line.strip().split()
        if len(parts) == 3 and parts[2].endswith(".export.CSV.zip"):
            urls.append(parts[2])
    return urls


def download_and_filter(url: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Download one export file; return (all_events, fire_events) or None."""
    try:
        resp = requests.get(url, timeout=TIMEOUT)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
    except Exception as exc:
        print(f"  [download] {url.split('/')[-1]}: {exc}", flush=True)
        return None

    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            with zf.open(zf.namelist()[0]) as f:
                raw = pd.read_csv(
                    f, sep="\t", header=None, dtype=str, low_memory=False
                )
    except Exception as exc:
        print(f"  [parse] {url.split('/')[-1]}: {exc}", flush=True)
        return None

    raw[C_ROOT_CODE] = pd.to_numeric(raw[C_ROOT_CODE], errors="coerce")
    raw[C_BASE_CODE] = pd.to_numeric(raw[C_BASE_CODE], errors="coerce")

    filtered = raw[
        raw[C_ROOT_CODE].isin(CONFLICT_ROOT_CODES) &
        raw[C_COUNTRY].isin(FIPS_TO_REGION)
    ].copy()

    if filtered.empty:
        return pd.DataFrame(), pd.DataFrame()

    out = pd.DataFrame({
        "event_date":      pd.to_datetime(
                               filtered[C_DATE].astype(str),
                               format="%Y%m%d", errors="coerce"
                           ),
        "date_added":      filtered[C_ADDED] if C_ADDED < len(filtered.columns) else pd.NA,
        "country_fips":    filtered[C_COUNTRY],
        "project_region":  filtered[C_COUNTRY].map(FIPS_TO_REGION),
        "event_root_code": filtered[C_ROOT_CODE].astype("Int64"),
        "event_base_code": filtered[C_BASE_CODE].astype("Int64"),
        "latitude":        pd.to_numeric(filtered[C_LAT], errors="coerce"),
        "longitude":       pd.to_numeric(filtered[C_LON], errors="coerce"),
        "source_url":      filtered[C_URL] if C_URL < len(filtered.columns) else "",
    })
    out = out.dropna(subset=["event_date", "latitude", "longitude"])

    fire = out[out["event_base_code"].isin(FIRE_BASE_CODES)].copy()
    return out, fire


def append_to_csv(path: Path, df: pd.DataFrame) -> None:
    if df.empty:
        return
    write_header = not path.exists() or path.stat().st_size == 0
    df.to_csv(path, mode="a", header=write_header, index=False)


def process_url(url: str, seen: set[str]) -> int:
    """Download, filter, and append one file. Returns number of new rows added."""
    if url in seen:
        return 0

    result = download_and_filter(url)
    mark_seen(url)
    seen.add(url)

    if result is None:
        return 0

    all_df, fire_df = result
    append_to_csv(OUTPUT_ALL,  all_df)
    append_to_csv(OUTPUT_FIRE, fire_df)

    ts = url.split("/")[-1].replace(".export.CSV.zip", "")
    now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    print(
        f"  [{now}] {ts}  all={len(all_df)}  fire={len(fire_df)}",
        flush=True,
    )
    return len(all_df)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-interval", type=int, default=POLL_INTERVAL,
                        help="Seconds between polls (default 900 = 15 min)")
    args = parser.parse_args()
    interval = args.poll_interval

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    seen = load_seen()
    total_appended = 0

    print("GDELT v2 live poller started (Ctrl+C to stop)")
    print(f"  Poll interval : {interval}s")
    print(f"  Output (all)  : {OUTPUT_ALL}")
    print(f"  Output (fire) : {OUTPUT_FIRE}")
    print(f"  Already seen  : {len(seen)} URLs\n")

    while True:
        urls = latest_export_urls()
        for url in urls:
            n = process_url(url, seen)
            total_appended += n

        next_poll = datetime.now(timezone.utc)
        print(
            f"  next poll in {interval}s  |  total appended this session: {total_appended:,}",
            flush=True,
        )
        time.sleep(interval)


if __name__ == "__main__":
    main()
