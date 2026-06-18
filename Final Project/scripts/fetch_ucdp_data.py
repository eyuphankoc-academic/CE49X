"""Download UCDP GED (Georeferenced Event Dataset) and filter for project regions.

No API token, no registration — uses UCDP's public bulk CSV download.
Dataset: GED v24.1  (events 1989-2023, ~30 MB zip)
Source:  https://ucdp.uu.se/downloads/

The zip is downloaded once and cached at data/raw/ucdp_ged241.zip.
Subsequent runs skip the download if the file already exists.

Usage:
    python scripts/fetch_ucdp_data.py
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

GED_ZIP_URL   = "https://ucdp.uu.se/downloads/ged/ged241-csv.zip"
GED_ZIP_CACHE = RAW_DATA_DIR / "ucdp_ged241.zip"
GED_CSV_NAME  = "GEDEvent_v24_1.csv"
OUTPUT_PATH   = RAW_DATA_DIR / "ucdp_events_raw.csv"

# Only keep events from these years (overlap with our FIRMS data window).
# GED 24.1 covers up to 2023; earlier years give useful baseline context.
FILTER_YEARS = {2021, 2022, 2023}

# Country names exactly as they appear in UCDP + project region mapping.
REGION_COUNTRIES: dict[str, list[str]] = {
    "Ukraine_Black_Sea":       ["Ukraine"],
    "Red_Sea_Yemen":           ["Yemen (North Yemen)"],
    "Gaza_Israel_Lebanon":     ["Israel", "Lebanon"],
    "Persian_Gulf_Iraq_Iran":  ["Iraq", "Iran"],
    "Sudan_Red_Sea_Corridor":  ["Sudan", "South Sudan"],
    "Libya_Mediterranean":     ["Libya"],
}

# Columns we actually need downstream.
KEEP_COLS = [
    "id", "year", "conflict_name",
    "side_a", "side_b",
    "where_description", "adm_1", "adm_2",
    "latitude", "longitude",
    "country", "country_id",
    "date_start", "date_end",
    "deaths_a", "deaths_b", "deaths_civilians", "deaths_unknown",
    "best", "high", "low",
    "type_of_violence",
]


def download_zip() -> bytes:
    if GED_ZIP_CACHE.exists():
        print(f"Using cached zip: {GED_ZIP_CACHE}")
        return GED_ZIP_CACHE.read_bytes()

    print(f"Downloading UCDP GED 24.1 (~30 MB) ...")
    resp = requests.get(GED_ZIP_URL, timeout=180, stream=True)
    resp.raise_for_status()

    chunks: list[bytes] = []
    downloaded = 0
    for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MB chunks
        chunks.append(chunk)
        downloaded += len(chunk)
        print(f"  {downloaded / 1e6:.1f} MB downloaded", end="\r", flush=True)

    data = b"".join(chunks)
    print(f"\n  Download complete: {len(data) / 1e6:.1f} MB")
    GED_ZIP_CACHE.write_bytes(data)
    print(f"  Cached at: {GED_ZIP_CACHE}")
    return data


def load_ged(zip_bytes: bytes) -> pd.DataFrame:
    print("Reading CSV from zip...")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        with z.open(GED_CSV_NAME) as f:
            df = pd.read_csv(f, low_memory=False)
    print(f"  Total rows in GED 24.1: {len(df):,}  ({df['year'].min()}–{df['year'].max()})")
    return df


def filter_and_label(df: pd.DataFrame) -> pd.DataFrame:
    # Keep only relevant years
    df = df[df["year"].isin(FILTER_YEARS)].copy()
    print(f"  Rows after year filter ({sorted(FILTER_YEARS)}): {len(df):,}")

    # Build reverse lookup: country -> region
    country_to_region: dict[str, str] = {}
    for region, countries in REGION_COUNTRIES.items():
        for c in countries:
            country_to_region[c] = region

    df = df[df["country"].isin(country_to_region)].copy()
    df["project_region"] = df["country"].map(country_to_region)
    print(f"  Rows after country filter: {len(df):,}")

    # Keep only the columns we need (ignore any missing)
    keep = [c for c in KEEP_COLS + ["project_region"] if c in df.columns]
    df = df[keep].copy()

    # Parse dates
    for col in ("date_start", "date_end"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Unified event_date column used by evaluate_with_external.py
    if "date_start" in df.columns:
        df["event_date"] = df["date_start"]

    for col in ("latitude", "longitude", "best", "deaths_civilians"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.reset_index(drop=True)


def main() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    zip_bytes = download_zip()
    df_raw = load_ged(zip_bytes)
    df = filter_and_label(df_raw)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df):,} UCDP events -> {OUTPUT_PATH}")

    print("\nEvents by region and year:")
    pivot = (
        df.groupby(["project_region", "year"])
        .size()
        .unstack(fill_value=0)
    )
    print(pivot.to_string())

    if "best" in df.columns:
        by_region = df.groupby("project_region")["best"].sum().astype(int)
        print("\nEstimated deaths by region (best estimate):")
        print(by_region.sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
