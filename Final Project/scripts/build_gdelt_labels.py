"""
build_gdelt_labels.py — Create point-level conflict labels from GDELT v2 events.

Method 2 label: conflict_gdelt_nearby
  For each FIRMS detection (lat, lon, date):
    Search GDELT conflict events within RADIUS_KM and ±DAYS_WINDOW days.
    Label = 1 if any qualifying event found, else 0.

Filtering applied to GDELT events before the join:
  1. Date range  : event_date must be within project window (2024-01-01 → 2026-05-29)
  2. Bounding box: ActionGeo coords must fall inside one of our 6 conflict bboxes
                   (removes all-Russia events, sea-only pings, zero/null coords)
  3. Coordinate precision: drop events whose lat OR lon rounds to a .0 or .5 grid
                   with > CENTROID_THRESHOLD duplicates on the same day
                   (these are city/country centroid placeholders, not actual locations)
  4. Deduplication: keep one event per unique (event_date, lat_1dp, lon_1dp, region)
                   so the same explosion reported by 50 outlets counts once

Usage:
    python scripts/build_gdelt_labels.py
    python scripts/build_gdelt_labels.py --radius 50 --days 3 --sample 250000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# ── config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED    = PROJECT_ROOT / "data" / "processed"
RAW          = PROJECT_ROOT / "data" / "raw"

GDELT_CSV    = RAW / "gdelt_conflict_events_v2_raw.csv"   # full merged output
GDELT_CACHE  = RAW / "gdelt_cache_v2"                     # per-slot CSVs (fallback)
FIRMS_CSV    = PROCESSED / "firms_clean.csv"
MONITORING   = PROCESSED / "daily_region_monitoring_summary.csv"
OUTPUT_CSV   = PROCESSED / "firms_method2_labeled.csv"

RANDOM_SEED  = 49

# ── bounding boxes: (region, lat_min, lat_max, lon_min, lon_max) ──────────────
BBOXES = [
    ("Ukraine_Black_Sea",       38.5, 54.5,  25.5, 44.0),
    ("Red_Sea_Yemen",            8.5, 25.0,  30.5, 47.0),
    ("Gaza_Israel_Lebanon",     27.5, 36.5,  31.5, 39.0),
    ("Persian_Gulf_Iraq_Iran",  21.5, 39.5,  36.5, 61.5),
    ("Sudan_Red_Sea_Corridor",   6.5, 25.0,  20.0, 40.0),
    ("Libya_Mediterranean",     17.5, 35.5,   7.5, 27.5),
]


def in_bbox(lat: float, lon: float) -> bool:
    for _, lat_min, lat_max, lon_min, lon_max in BBOXES:
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return True
    return False


# ── step 1: load GDELT events ─────────────────────────────────────────────────

def load_gdelt() -> pd.DataFrame:
    """Always load from per-slot cache files — they are atomic and never modified.

    The merged output CSV is written incrementally while downloading and may contain
    partial rows or URL fields with commas (making it unsafe to read mid-download).
    Cache files are written once per slot and are always complete.
    """
    if not GDELT_CACHE.exists():
        raise FileNotFoundError(
            f"Cache dir not found: {GDELT_CACHE}\n"
            "Run:  python scripts/fetch_gdelt_events.py"
        )

    cache_files = sorted(f for f in GDELT_CACHE.glob("*.csv") if f.stat().st_size > 10)
    if not cache_files:
        raise FileNotFoundError("No cached GDELT slot files found. Still downloading?")

    print(f"Reading {len(cache_files):,} GDELT cache files...")
    frames = []
    for i, f in enumerate(cache_files):
        try:
            frames.append(pd.read_csv(f, dtype=str))
        except Exception:
            pass
        if (i + 1) % 2000 == 0:
            print(f"  {i+1:,} / {len(cache_files):,} files read...", end="\r")

    df = pd.concat(frames, ignore_index=True)
    print(f"  Loaded {len(df):,} rows from {len(frames):,} cache files")

    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df["latitude"]   = pd.to_numeric(df["latitude"],   errors="coerce")
    df["longitude"]  = pd.to_numeric(df["longitude"],  errors="coerce")
    return df


# ── step 2: filter GDELT events for fire-probable, precise events ─────────────

def filter_gdelt(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    print(f"\nFiltering GDELT events ({n0:,} rows)...")

    # 2a. Date range
    df = df[
        (df["event_date"] >= pd.Timestamp("2024-01-01")) &
        (df["event_date"] <= pd.Timestamp("2026-05-29"))
    ].copy()
    print(f"  After date filter          : {len(df):,}  (dropped {n0 - len(df):,})")

    # 2b. Drop null / zero coordinates
    n = len(df)
    df = df.dropna(subset=["latitude", "longitude"])
    df = df[(df["latitude"] != 0) | (df["longitude"] != 0)]
    print(f"  After null/zero coord drop : {len(df):,}  (dropped {n - len(df):,})")

    # 2c. Bounding box — keeps only events inside our 6 conflict regions
    n = len(df)
    mask = df.apply(lambda r: in_bbox(r["latitude"], r["longitude"]), axis=1)
    df = df[mask].copy()
    print(f"  After bounding box filter  : {len(df):,}  (dropped {n - len(df):,})")

    # 2d. Remove country/city centroid placeholders.
    #     GDELT rounds unknown locations to ±0.5 or ±0.0 grids.
    #     If the same rounded (lat, lon) appears > 30 times on the same date
    #     it's almost certainly a centroid, not a real event location.
    n = len(df)
    df["lat_r"] = df["latitude"].round(1)
    df["lon_r"] = df["longitude"].round(1)
    daily_counts = (
        df.groupby(["event_date", "lat_r", "lon_r"])["latitude"]
        .transform("size")
    )
    df = df[daily_counts <= 30].copy()
    df = df.drop(columns=["lat_r", "lon_r"])
    print(f"  After centroid removal     : {len(df):,}  (dropped {n - len(df):,})")

    # 2e. Deduplicate: same event reported by multiple articles.
    #     Keep one row per (event_date, lat 1dp, lon 1dp, region).
    n = len(df)
    df["lat_1"] = df["latitude"].round(1)
    df["lon_1"] = df["longitude"].round(1)
    df = df.drop_duplicates(subset=["event_date", "lat_1", "lon_1", "project_region"])
    df = df.drop(columns=["lat_1", "lon_1"])
    print(f"  After deduplication        : {len(df):,}  (dropped {n - len(df):,})")

    print(f"\nFinal GDELT events for matching: {len(df):,}")
    print("Events by region:")
    print(df["project_region"].value_counts().to_string())
    print(f"\nDate range: {df['event_date'].min().date()} to {df['event_date'].max().date()}")
    return df.reset_index(drop=True)


# ── step 3: load FIRMS sample with Method 1 label ─────────────────────────────

def load_firms(n_sample: int) -> pd.DataFrame:
    print(f"\nLoading FIRMS data (sample={n_sample:,})...")
    monitoring = pd.read_csv(
        MONITORING,
        usecols=["region", "acq_date", "conflict_news_nearby"],
    )
    monitoring["acq_date"] = pd.to_datetime(monitoring["acq_date"])
    monitoring["conflict_news_nearby"] = (
        monitoring["conflict_news_nearby"].fillna(0).astype(int)
    )

    chunks, loaded = [], 0
    for chunk in pd.read_csv(FIRMS_CSV, chunksize=500_000, low_memory=False):
        remaining = n_sample - loaded
        if remaining <= 0:
            break
        if len(chunk) > remaining:
            chunk = chunk.sample(n=remaining, random_state=RANDOM_SEED)
        chunks.append(chunk)
        loaded += len(chunk)
    firms = pd.concat(chunks, ignore_index=True)
    firms["acq_date"] = pd.to_datetime(firms["acq_date"], errors="coerce")
    firms = firms.merge(monitoring, on=["region", "acq_date"], how="left")
    firms["conflict_news_nearby"] = firms["conflict_news_nearby"].fillna(0).astype(int)
    firms["latitude"]  = pd.to_numeric(firms["latitude"],  errors="coerce")
    firms["longitude"] = pd.to_numeric(firms["longitude"], errors="coerce")
    firms = firms.dropna(subset=["latitude", "longitude", "acq_date"])
    print(f"  FIRMS sample: {len(firms):,} rows")
    print(f"  Method 1 positive rate: {firms['conflict_news_nearby'].mean():.1%}")
    return firms.reset_index(drop=True)


# ── step 4: spatial-temporal join ─────────────────────────────────────────────

EARTH_RADIUS_KM = 6371.0


def assign_gdelt_labels(
    firms: pd.DataFrame,
    gdelt: pd.DataFrame,
    radius_km: float,
    days_window: int,
) -> np.ndarray:
    """
    For each FIRMS row, label=1 if any GDELT event falls within
    radius_km AND ±days_window days of the detection date.

    Strategy: iterate over unique detection dates; for each date
    build a cKDTree from GDELT events in the ±days_window window;
    query all FIRMS points for that date in one shot.
    """
    labels = np.zeros(len(firms), dtype=np.int8)
    radius_rad = radius_km / EARTH_RADIUS_KM

    gdelt_dates = gdelt["event_date"].values
    gdelt_lat   = np.radians(gdelt["latitude"].values)
    gdelt_lon   = np.radians(gdelt["longitude"].values)

    unique_dates = firms["acq_date"].dt.normalize().unique()
    print(f"\nRunning spatial-temporal join:")
    print(f"  Radius  : {radius_km} km")
    print(f"  Window  : ±{days_window} days")
    print(f"  Dates   : {len(unique_dates):,}")

    for i, d in enumerate(sorted(unique_dates)):
        if (i + 1) % 50 == 0 or (i + 1) == len(unique_dates):
            print(f"  Progress: {i+1:,}/{len(unique_dates):,} dates", end="\r")

        lo = d - np.timedelta64(days_window, "D")
        hi = d + np.timedelta64(days_window, "D")
        in_window = (gdelt_dates >= lo) & (gdelt_dates <= hi)

        if not in_window.any():
            continue

        # Build KDTree for events in this time window
        evt_lat = gdelt_lat[in_window]
        evt_lon = gdelt_lon[in_window]
        evt_coords = np.column_stack([evt_lat, evt_lon])
        tree = cKDTree(evt_coords)

        # Query FIRMS points for this date
        day_mask = firms["acq_date"].dt.normalize() == d
        day_idx  = np.where(day_mask)[0]
        if len(day_idx) == 0:
            continue

        firms_lat = np.radians(firms.loc[day_idx, "latitude"].values)
        firms_lon = np.radians(firms.loc[day_idx, "longitude"].values)
        query_pts = np.column_stack([firms_lat, firms_lon])

        # count_neighbors > 0 means at least one event within radius
        hits = tree.query_ball_point(query_pts, radius_rad)
        for j, h in zip(day_idx, hits):
            if len(h) > 0:
                labels[j] = 1

    print(f"\n  Done. Method 2 positive rate: {labels.mean():.1%}")
    return labels


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--radius",  type=float, default=50,
                        help="Spatial radius in km (default 50)")
    parser.add_argument("--days",    type=int,   default=3,
                        help="Time window in ±days (default 3)")
    parser.add_argument("--sample",  type=int,   default=250_000,
                        help="FIRMS rows to label (default 250,000)")
    args = parser.parse_args()

    gdelt = load_gdelt()
    gdelt = filter_gdelt(gdelt)

    firms = load_firms(args.sample)

    labels = assign_gdelt_labels(firms, gdelt, args.radius, args.days)
    firms["conflict_gdelt_nearby"] = labels

    # Agreement stats
    m1 = firms["conflict_news_nearby"].values
    m2 = firms["conflict_gdelt_nearby"].values
    agree     = (m1 == m2).mean()
    both_pos  = ((m1 == 1) & (m2 == 1)).mean()
    m1_only   = ((m1 == 1) & (m2 == 0)).mean()
    m2_only   = ((m1 == 0) & (m2 == 1)).mean()
    both_neg  = ((m1 == 0) & (m2 == 0)).mean()

    print(f"\n{'='*55}")
    print("Label agreement — Method 1 vs Method 2")
    print(f"{'='*55}")
    print(f"  Overall agreement   : {agree:.1%}")
    print(f"  Both = 1 (conflict) : {both_pos:.1%}")
    print(f"  Both = 0 (no conf.) : {both_neg:.1%}")
    print(f"  Method1=1, Method2=0: {m1_only:.1%}  (M1 over-labels relative to M2)")
    print(f"  Method1=0, Method2=1: {m2_only:.1%}  (M2 finds events M1 misses)")
    print(f"{'='*55}")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    firms.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved labeled dataset ({len(firms):,} rows) -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
