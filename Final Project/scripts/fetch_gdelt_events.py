"""Download every GDELT v2 15-minute event file for project regions (2024-01-01 to 2026-05-29).

GDELT v2 publishes one .export.CSV.zip every 15 minutes -- no API key needed.
Source: http://data.gdeltproject.org/gdeltv2/

Coverage
--------
2024-01-01 to 2026-05-29  =  ~84,480 files (880 days x 96 slots/day)
Filtered to CAMEO codes 18/19/20 + our 11 FIPS countries.  Each filtered file
is cached so re-runs skip already-done work.

Estimated wall time (first run, 8 workers): ~90-150 min.
Subsequent runs: < 1 min (all cache hits).

Two outputs
-----------
gdelt_conflict_events_v2_raw.csv
    All violent events (root codes 18/19/20).  Broad superset.

gdelt_fire_relevant_events_v2.csv
    Subset filtered to EventBaseCode {184, 185, 193} -- events that produce
    large enough thermal signatures to appear in FIRMS satellite data:
      184  Fight with artillery and tanks    (building/infrastructure fires)
      185  Employ aerial weapons             (airstrikes, drone strikes)
      193  Conduct bombing                   (IEDs, explosions, factory/ship hits)
    Small-arms fire, assassinations, blockades etc. are excluded because they
    do not create the sustained high-FRP anomalies FIRMS detects.

GDELT v2 column indices (0-indexed, tab-separated, 61 cols total):
  1   SQLDATE               event date YYYYMMDD
  27  EventBaseCode         CAMEO 3-digit base code  (NEW -- stored in cache)
  28  EventRootCode         CAMEO 2-digit root code
  53  ActionGeo_CountryCode FIPS 2-letter country
  56  ActionGeo_Lat         latitude
  57  ActionGeo_Long        longitude
  59  DATEADDED             15-minute publish timestamp YYYYMMDDHHMMSS
  60  SOURCEURL             source article

Usage:
    python scripts/fetch_gdelt_events.py
    python scripts/fetch_gdelt_events.py --workers 12
    python scripts/fetch_gdelt_events.py --start 2025-01-01 --end 2025-06-30
"""

from __future__ import annotations

import argparse
import io
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT      = Path(__file__).resolve().parents[1]
RAW_DATA_DIR      = PROJECT_ROOT / "data" / "raw"
OUTPUT_ALL        = RAW_DATA_DIR / "gdelt_conflict_events_v2_raw.csv"
OUTPUT_FIRE       = RAW_DATA_DIR / "gdelt_fire_relevant_events_v2.csv"
CACHE_DIR         = RAW_DATA_DIR / "gdelt_cache_v2"

# ── GDELT v2 ──────────────────────────────────────────────────────────────────
GDELT_URL    = "http://data.gdeltproject.org/gdeltv2/{ts}.export.CSV.zip"
TIMEOUT      = 60    # seconds per request
WORKER_SLEEP = 0.15  # polite delay per worker after each download

# ── GDELT v2 column indices (0-indexed) ───────────────────────────────────────
C_DATE      = 1
C_BASE_CODE = 27   # EventBaseCode -- 3-digit CAMEO
C_ROOT_CODE = 28   # EventRootCode -- 2-digit CAMEO
C_COUNTRY   = 53
C_LAT       = 56
C_LON       = 57
C_ADDED     = 59
C_URL       = 60

# ── Broad cache filter: root codes 18/19/20 ───────────────────────────────────
# Keeps everything violent so the cache is a useful superset.
CONFLICT_ROOT_CODES = {18, 19, 20}

# ── Fire-relevant base codes (produce large thermal anomalies in FIRMS) ────────
# 184 artillery/tanks, 185 aerial/drone strikes, 193 bombing/explosions.
# These are the events that burn buildings, factories, ships, power plants.
FIRE_BASE_CODES = {184, 185, 193}

# ── Project regions (FIPS 10-4 -> region name) ────────────────────────────────
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

DEFAULT_START = date(2024, 1, 1)
DEFAULT_END   = date(2026, 5, 29)


def all_timestamps(start: date, end: date) -> list[str]:
    """Return YYYYMMDDHHMMSS strings for every 15-min slot between start and end."""
    stamps: list[str] = []
    d = start
    while d <= end:
        prefix = d.strftime("%Y%m%d")
        for hour in range(24):
            for minute in (0, 15, 30, 45):
                stamps.append(f"{prefix}{hour:02d}{minute:02d}00")
        d += timedelta(days=1)
    return stamps


def fetch_one(ts: str) -> pd.DataFrame | None:
    """Download, filter, and cache one 15-minute GDELT v2 export file."""
    cache_path = CACHE_DIR / f"{ts}.csv"

    if cache_path.exists():
        if cache_path.stat().st_size > 10:
            try:
                df = pd.read_csv(cache_path, dtype=str)
                if "event_base_code" in df.columns:
                    return df  # cache is current
                # Old cache entry missing event_base_code -- fall through to re-download
            except Exception:
                pass
        else:
            return None  # empty marker = no events for this slot

    url = GDELT_URL.format(ts=ts)
    try:
        resp = requests.get(url, timeout=TIMEOUT)
        if resp.status_code == 404:
            cache_path.touch()
            return None
        resp.raise_for_status()
    except Exception:
        return None  # network error -- retry on re-run, no cache written

    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            with zf.open(zf.namelist()[0]) as f:
                raw = pd.read_csv(
                    f, sep="\t", header=None, dtype=str, low_memory=False
                )
    except Exception:
        return None

    # Filter 1: CAMEO root code in broad conflict set
    raw[C_ROOT_CODE] = pd.to_numeric(raw[C_ROOT_CODE], errors="coerce")
    filtered = raw[raw[C_ROOT_CODE].isin(CONFLICT_ROOT_CODES)].copy()

    # Filter 2: action geography in our project countries
    if filtered.empty or C_COUNTRY >= len(filtered.columns):
        cache_path.touch()
        return None
    filtered = filtered[filtered[C_COUNTRY].isin(FIPS_TO_REGION)].copy()

    if filtered.empty:
        cache_path.touch()
        return None

    # Parse base code (3-digit) -- stored so fire-relevant filter works later
    raw[C_BASE_CODE] = pd.to_numeric(raw[C_BASE_CODE], errors="coerce")
    filtered[C_BASE_CODE] = pd.to_numeric(filtered[C_BASE_CODE], errors="coerce")

    out = pd.DataFrame(
        {
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
        }
    )
    out = out.dropna(subset=["event_date", "latitude", "longitude"])

    if out.empty:
        cache_path.touch()
        return None

    out.to_csv(cache_path, index=False)
    time.sleep(WORKER_SLEEP)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch GDELT v2 15-min conflict events.")
    parser.add_argument("--start",   default=DEFAULT_START.isoformat())
    parser.add_argument("--end",     default=DEFAULT_END.isoformat())
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    start   = date.fromisoformat(args.start)
    end     = date.fromisoformat(args.end)
    workers = args.workers

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    stamps    = all_timestamps(start, end)
    total     = len(stamps)
    cached    = sum(1 for s in stamps if (CACHE_DIR / f"{s}.csv").exists())
    remaining = total - cached

    print("GDELT v2 15-minute conflict event fetcher")
    print(f"  Date range : {start} to {end}")
    print(f"  Total slots: {total:,}  |  cached: {cached:,}  |  to download: {remaining:,}")
    print(f"  Workers    : {workers}")
    if remaining > 0:
        est = remaining / workers * (WORKER_SLEEP + 0.7) / 60
        print(f"  Est. time  : {est:.0f}-{est * 1.8:.0f} min")
    print(f"  Cache dir  : {CACHE_DIR}\n")

    done_count = [0]
    lock = threading.Lock()
    t0 = time.time()
    all_frames: list[pd.DataFrame] = []

    def on_done(df: pd.DataFrame | None) -> None:
        if df is not None and not df.empty:
            all_frames.append(df)
        with lock:
            done_count[0] += 1
            n = done_count[0]
            if n % 1000 == 0 or n == total:
                elapsed = time.time() - t0
                rate    = n / elapsed if elapsed > 0 else 1
                eta_min = (total - n) / rate / 60
                events  = sum(len(f) for f in all_frames)
                print(
                    f"  [{n:,}/{total:,}] {n/total*100:5.1f}%  "
                    f"events: {events:,}  ETA: {eta_min:.0f} min",
                    flush=True,
                )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_one, ts): ts for ts in stamps}
        for future in as_completed(futures):
            on_done(future.result())

    print()
    if not all_frames:
        print("No conflict events found in range.")
        return

    combined = pd.concat(all_frames, ignore_index=True)
    combined["event_date"] = pd.to_datetime(combined["event_date"], errors="coerce")

    # Deduplicate (same event picked up by multiple 15-min windows)
    before   = len(combined)
    combined = combined.drop_duplicates(
        subset=["event_date", "latitude", "longitude", "event_root_code"]
    )
    combined = combined.sort_values(["event_date", "project_region"]).reset_index(drop=True)
    print(f"Deduplication: {before:,} -> {len(combined):,} unique events")

    # ── Output 1: full broad dataset ──────────────────────────────────────────
    combined.to_csv(OUTPUT_ALL, index=False)
    print(f"Saved {len(combined):,} events (all 18/19/20)  -> {OUTPUT_ALL}")

    # ── Output 2: fire-relevant subset ────────────────────────────────────────
    # Old cache files (before this update) don't have event_base_code -- mark
    # them NaN so they stay in the broad file but are excluded from fire filter.
    if "event_base_code" in combined.columns:
        combined["event_base_code"] = pd.to_numeric(
            combined["event_base_code"], errors="coerce"
        )
        fire = combined[combined["event_base_code"].isin(FIRE_BASE_CODES)].copy()
    else:
        fire = combined.iloc[0:0].copy()  # empty -- no base codes available

    fire.to_csv(OUTPUT_FIRE, index=False)
    print(f"Saved {len(fire):,} events (fire-relevant 184/185/193) -> {OUTPUT_FIRE}\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    combined["year"] = combined["event_date"].dt.year
    print("All events by region and year:")
    pivot = (
        combined.groupby(["project_region", "year"])
        .size()
        .unstack(fill_value=0)
    )
    print(pivot.to_string())

    print("\nCAMEO root code breakdown (all events):")
    labels = {18: "Assault (18)", 19: "Fight (19)", 20: "Mass violence (20)"}
    print(combined["event_root_code"].map(labels).value_counts().to_string())

    if not fire.empty:
        print("\nFire-relevant base code breakdown:")
        base_labels = {
            184: "Artillery/tanks (184)",
            185: "Aerial/drone strike (185)",
            193: "Bombing/explosion (193)",
        }
        fire["event_base_code"] = pd.to_numeric(fire["event_base_code"], errors="coerce")
        print(fire["event_base_code"].map(base_labels).value_counts().to_string())

        print("\nFire-relevant events by region and year:")
        fire["year"] = pd.to_datetime(fire["event_date"], errors="coerce").dt.year
        print(
            fire.groupby(["project_region", "year"])
            .size()
            .unstack(fill_value=0)
            .to_string()
        )


if __name__ == "__main__":
    main()
