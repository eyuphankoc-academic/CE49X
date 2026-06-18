"""Clean and summarize CE49X final project raw datasets.

Outputs:
    data/processed/firms_clean.csv
    data/processed/firms_model_sample.csv
    data/processed/firms_daily_region_summary.csv
    data/processed/news_clean.csv
    data/processed/news_daily_region_summary.csv
    data/processed/daily_region_monitoring_summary.csv
    reports/data_cleaning_report.md
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

FIRMS_RAW_FILES = [
    ("firms_thermal_raw.csv", "VIIRS_SNPP_SP", "2024_standard_processed", "conflict"),
    ("firms_thermal_2025_2026_raw.csv", "VIIRS_SNPP_SP", "2025_2026_standard_processed", "conflict"),
    ("firms_thermal_2026_apr_may_nrt_raw.csv", "VIIRS_SNPP_NRT", "2026_apr_may_near_real_time", "conflict"),
    ("firms_thermal_nonconflict_2024_raw.csv", "VIIRS_SNPP_SP", "2024_nonconflict_reference", "nonconflict"),
    ("firms_thermal_nonconflict_2025_2026_raw.csv", "VIIRS_SNPP_SP", "2025_2026_nonconflict_reference", "nonconflict"),
    ("firms_thermal_nonconflict_2026_apr_may_nrt_raw.csv", "VIIRS_SNPP_NRT", "2026_apr_may_nonconflict_nrt", "nonconflict"),
]

NEWS_RAW_FILES = [
    ("conflict_news_raw.csv", "GDELT", "conflict"),
    ("conflict_news_2025_2026_google_rss_raw.csv", "Google News RSS", "conflict"),
    ("conflict_news_2024_expanded_rss_raw.csv", "Google News RSS expanded", "conflict"),
    ("conflict_news_2025_2026_expanded_rss_raw.csv", "Google News RSS expanded", "conflict"),
    ("news_nonconflict_fire_and_conflict_rss_raw.csv", "Google News RSS non-conflict reference", "nonconflict"),
]

FIRMS_OUTPUT = PROCESSED_DATA_DIR / "firms_clean.csv"
FIRMS_SAMPLE_OUTPUT = PROCESSED_DATA_DIR / "firms_model_sample.csv"
FIRMS_DAILY_OUTPUT = PROCESSED_DATA_DIR / "firms_daily_region_summary.csv"
NEWS_OUTPUT = PROCESSED_DATA_DIR / "news_clean.csv"
NEWS_DAILY_OUTPUT = PROCESSED_DATA_DIR / "news_daily_region_summary.csv"
DAILY_MONITORING_OUTPUT = PROCESSED_DATA_DIR / "daily_region_monitoring_summary.csv"
CLEANING_REPORT_OUTPUT = REPORTS_DIR / "data_cleaning_report.md"

FIRMS_CHUNK_SIZE = 250_000
MODEL_SAMPLE_MAX_ROWS = 250_000
RANDOM_SEED = 49


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [column.strip().lower() for column in frame.columns]
    return frame


def parse_acquisition_datetime(frame: pd.DataFrame) -> pd.Series:
    dates = pd.to_datetime(frame["acq_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    times = (
        pd.to_numeric(frame["acq_time"], errors="coerce")
        .fillna(-1)
        .astype(int)
        .astype(str)
        .str.zfill(4)
    )
    hours = times.str.slice(0, 2)
    minutes = times.str.slice(2, 4)
    return pd.to_datetime(dates + " " + hours + ":" + minutes, errors="coerce", utc=True)


def is_usable_confidence(confidence: pd.Series) -> pd.Series:
    """Keep nearly all detections so we don't miss real conflict thermal signatures.

    We intentionally retain even low-confidence detections (numeric >= 0 or text == "low")
    because in conflict monitoring, a false negative (missed event) is worse than a false
    positive (extra signal the ML model can learn to filter).
    """
    normalized = confidence.astype(str).str.strip().str.lower()
    numeric = pd.to_numeric(normalized, errors="coerce")
    numeric_ok = numeric.ge(0)
    text_ok = ~normalized.isin({"", "nan", "none"})
    return numeric_ok.fillna(False) | (numeric.isna() & text_ok)


def clean_firms_chunk(frame: pd.DataFrame, source: str, source_period: str, source_file: str,
                       category: str = "conflict") -> pd.DataFrame:
    frame = normalize_columns(frame)
    if "type" not in frame.columns:
        frame["type"] = pd.NA

    required = ["latitude", "longitude", "bright_ti4", "acq_date", "acq_time", "confidence", "frp", "region"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{source_file} is missing required columns: {missing}")

    for column in ["latitude", "longitude", "bright_ti4", "bright_ti5", "frp", "scan", "track"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["acq_date"] = pd.to_datetime(frame["acq_date"], errors="coerce")
    frame["acquisition_datetime_utc"] = parse_acquisition_datetime(frame)
    frame["confidence"] = frame["confidence"].astype(str).str.strip().str.lower()
    frame["daynight"] = frame["daynight"].astype(str).str.strip().str.upper()
    frame["region"] = frame["region"].astype(str).str.strip()
    frame["firms_source"] = source
    frame["source_period"] = source_period
    frame["source_file"] = source_file
    frame["region_category"] = category

    valid = (
        frame["latitude"].between(-90, 90)
        & frame["longitude"].between(-180, 180)
        & frame["acq_date"].notna()
        & frame["acquisition_datetime_utc"].notna()
        & frame["bright_ti4"].notna()
        & frame["frp"].ge(0)
        & is_usable_confidence(frame["confidence"])
    )
    frame = frame.loc[valid].copy()

    frame.drop_duplicates(
        subset=[
            "region",
            "latitude",
            "longitude",
            "acq_date",
            "acq_time",
            "satellite",
            "instrument",
            "bright_ti4",
            "frp",
        ],
        inplace=True,
    )

    keep_columns = [
        "region",
        "region_category",
        "latitude",
        "longitude",
        "acq_date",
        "acq_time",
        "acquisition_datetime_utc",
        "satellite",
        "instrument",
        "confidence",
        "bright_ti4",
        "bright_ti5",
        "frp",
        "daynight",
        "scan",
        "track",
        "type",
        "firms_source",
        "source_period",
        "source_file",
        "region_description",
    ]
    return frame[[column for column in keep_columns if column in frame.columns]]


def make_firms_daily_summary(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["acq_date"] = pd.to_datetime(frame["acq_date"]).dt.date

    summary = (
        frame.groupby(["region", "region_category", "acq_date"], as_index=False)
        .agg(
            firms_detection_count=("latitude", "size"),
            firms_mean_bright_ti4=("bright_ti4", "mean"),
            firms_max_bright_ti4=("bright_ti4", "max"),
            firms_mean_frp=("frp", "mean"),
            firms_max_frp=("frp", "max"),
            firms_total_frp=("frp", "sum"),
            firms_mean_latitude=("latitude", "mean"),
            firms_mean_longitude=("longitude", "mean"),
        )
    )

    daynight = (
        frame.assign(daynight=frame["daynight"].where(frame["daynight"].isin(["D", "N"]), "UNKNOWN"))
        .pivot_table(
            index=["region", "region_category", "acq_date"],
            columns="daynight", values="latitude", aggfunc="size", fill_value=0,
        )
        .reset_index()
        .rename(columns={"D": "firms_day_count", "N": "firms_night_count", "UNKNOWN": "firms_unknown_daynight_count"})
    )

    return summary.merge(daynight, on=["region", "region_category", "acq_date"], how="left")


def clean_firms_data() -> tuple[pd.DataFrame, dict[str, int]]:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if FIRMS_OUTPUT.exists():
        FIRMS_OUTPUT.unlink()

    first_write = True
    daily_frames: list[pd.DataFrame] = []
    sample_frames: list[pd.DataFrame] = []
    counts: dict[str, int] = {"raw_rows": 0, "clean_rows": 0}

    for file_name, source, source_period, category in FIRMS_RAW_FILES:
        path = RAW_DATA_DIR / file_name
        if not path.exists():
            print(f"Skipping missing FIRMS file: {file_name}", flush=True)
            continue

        for chunk_number, chunk in enumerate(pd.read_csv(path, chunksize=FIRMS_CHUNK_SIZE), start=1):
            counts["raw_rows"] += len(chunk)
            clean = clean_firms_chunk(chunk, source, source_period, file_name, category)
            counts["clean_rows"] += len(clean)

            clean.to_csv(FIRMS_OUTPUT, mode="w" if first_write else "a", header=first_write, index=False)
            first_write = False

            if not clean.empty:
                daily_frames.append(make_firms_daily_summary(clean))
                sample_size = min(len(clean), 15_000)
                sample_frames.append(clean.sample(n=sample_size, random_state=RANDOM_SEED + chunk_number))

            print(f"Cleaned {file_name} chunk {chunk_number}: {len(chunk):,} raw -> {len(clean):,} clean", flush=True)

    daily = pd.concat(daily_frames, ignore_index=True)
    daily = (
        daily.groupby(["region", "region_category", "acq_date"], as_index=False)
        .agg(
            firms_detection_count=("firms_detection_count", "sum"),
            firms_mean_bright_ti4=("firms_mean_bright_ti4", "mean"),
            firms_max_bright_ti4=("firms_max_bright_ti4", "max"),
            firms_mean_frp=("firms_mean_frp", "mean"),
            firms_max_frp=("firms_max_frp", "max"),
            firms_total_frp=("firms_total_frp", "sum"),
            firms_mean_latitude=("firms_mean_latitude", "mean"),
            firms_mean_longitude=("firms_mean_longitude", "mean"),
            firms_day_count=("firms_day_count", "sum"),
            firms_night_count=("firms_night_count", "sum"),
        )
        .sort_values(["region", "acq_date"])
    )
    daily.to_csv(FIRMS_DAILY_OUTPUT, index=False)

    sample = pd.concat(sample_frames, ignore_index=True)
    if len(sample) > MODEL_SAMPLE_MAX_ROWS:
        sample = sample.sample(n=MODEL_SAMPLE_MAX_ROWS, random_state=RANDOM_SEED)
    sample.to_csv(FIRMS_SAMPLE_OUTPUT, index=False)

    return daily, counts


def clean_title(title: object) -> str | pd.NA:
    if pd.isna(title):
        return pd.NA
    cleaned = re.sub(r"\s+", " ", str(title)).strip()
    return cleaned if cleaned else pd.NA


def parse_publication_dates(series: pd.Series) -> pd.Series:
    gdelt_mask = series.astype(str).str.match(r"^\d{8}T\d{6}Z$", na=False)
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns, UTC]")
    parsed.loc[gdelt_mask] = pd.to_datetime(series.loc[gdelt_mask], format="%Y%m%dT%H%M%SZ", errors="coerce", utc=True)
    parsed.loc[~gdelt_mask] = pd.to_datetime(series.loc[~gdelt_mask], errors="coerce", utc=True)
    return parsed


def load_keyword_lists() -> tuple[list[str], list[str]]:
    """Return (conflict_keywords, fire_keywords) used to tag each article title."""
    import json as _json
    config_dir = PROJECT_ROOT / "config"
    with (config_dir / "project_config.json").open(encoding="utf-8") as f:
        conflict_kw = [k.lower() for k in _json.load(f)["news_keywords"]]
    fire_kw: list[str] = []
    fire_path = config_dir / "fire_keywords.json"
    if fire_path.exists():
        with fire_path.open(encoding="utf-8") as f:
            fire_kw = [k.lower() for k in _json.load(f).get("fire_keywords", [])]
    return conflict_kw, fire_kw


def tag_articles_by_keyword(news: pd.DataFrame) -> pd.DataFrame:
    """Add `has_conflict_kw`, `has_fire_kw`, `is_conflict_article` columns by scanning titles.

    A title is treated as a *conflict article* only when it contains at least one
    conflict keyword and NO fire keyword — this avoids hybrid headlines like
    'troops sent to fight wildfire' from polluting the conflict signal.
    """
    conflict_kw, fire_kw = load_keyword_lists()
    titles = news["title"].astype(str).str.lower()

    def any_kw(vocab: list[str]) -> pd.Series:
        if not vocab:
            return pd.Series(False, index=titles.index)
        pattern = r"\b(?:" + "|".join(re.escape(k) for k in vocab) + r")\b"
        return titles.str.contains(pattern, regex=True, na=False)

    news = news.copy()
    news["has_conflict_kw"] = any_kw(conflict_kw)
    news["has_fire_kw"]     = any_kw(fire_kw)
    news["is_conflict_article"] = news["has_conflict_kw"] & (~news["has_fire_kw"])
    return news


def clean_news_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    frames: list[pd.DataFrame] = []
    counts: dict[str, int] = {"raw_rows": 0, "clean_rows": 0}

    for file_name, source_name, category in NEWS_RAW_FILES:
        path = RAW_DATA_DIR / file_name
        if not path.exists():
            print(f"Skipping missing news file: {file_name}", flush=True)
            continue

        frame = normalize_columns(pd.read_csv(path))
        frame["news_source_dataset"] = source_name
        frame["region_category"] = category
        counts["raw_rows"] += len(frame)
        frames.append(frame)

    news = pd.concat(frames, ignore_index=True)
    news["title"] = news["title"].apply(clean_title)
    news["publication_datetime_utc"] = parse_publication_dates(news["publication_date"])
    news["publication_date_clean"] = news["publication_datetime_utc"].dt.date
    news["region"] = news["region"].astype(str).str.strip()
    news["source"] = news["source"].astype(str).str.strip().replace({"nan": pd.NA, "": pd.NA})
    news["domain"] = news["domain"].astype(str).str.strip().replace({"nan": pd.NA, "": pd.NA})
    news["url"] = news["url"].astype(str).str.strip()

    news = news[
        news["region"].notna()
        & news["title"].notna()
        & news["publication_datetime_utc"].notna()
        & news["url"].notna()
        & news["url"].ne("")
    ].copy()
    news.drop_duplicates(subset=["url"], inplace=True)
    news.drop_duplicates(subset=["region", "title", "publication_date_clean"], inplace=True)

    # Tag each article by keyword content so we can label uniformly across regions.
    news = tag_articles_by_keyword(news)

    keep_columns = [
        "region",
        "region_category",
        "title",
        "publication_datetime_utc",
        "publication_date_clean",
        "source",
        "domain",
        "url",
        "language",
        "news_source_dataset",
        "query",
        "has_conflict_kw",
        "has_fire_kw",
        "is_conflict_article",
    ]
    news = news[[column for column in keep_columns if column in news.columns]].sort_values(
        ["region", "publication_datetime_utc", "title"]
    )
    counts["clean_rows"] = len(news)
    news.to_csv(NEWS_OUTPUT, index=False)

    daily = (
        news.groupby(["region", "region_category", "publication_date_clean"], as_index=False)
        .agg(
            news_article_count=("title", "size"),
            news_unique_source_count=("domain", "nunique"),
            news_conflict_article_count=("is_conflict_article", "sum"),
            news_fire_article_count=("has_fire_kw", "sum"),
        )
        .rename(columns={"publication_date_clean": "acq_date"})
        .sort_values(["region", "acq_date"])
    )
    daily["news_conflict_article_count"] = daily["news_conflict_article_count"].astype(int)
    daily["news_fire_article_count"] = daily["news_fire_article_count"].astype(int)
    daily.to_csv(NEWS_DAILY_OUTPUT, index=False)

    return news, daily, counts


def create_monitoring_summary(firms_daily: pd.DataFrame, news_daily: pd.DataFrame) -> pd.DataFrame:
    """Build the day-level monitoring summary with DATA-DRIVEN labels.

    The label `conflict_news_nearby` is computed UNIFORMLY across every region
    (both conflict and non-conflict reference regions). It depends solely on the
    number of *conflict-keyword-only* articles published in that region around
    the detection date — not on the region's predefined category.

    This means:
    - A genuine missile attack in Australia, if reported, will spike
      `news_conflict_article_count` and the model can flip the label to 1.
    - A purely wildfire week in Ukraine, with no war headlines, can still get
      labelled 0.
    """
    firms_daily = firms_daily.copy()
    news_daily = news_daily.copy()
    firms_daily["acq_date"] = pd.to_datetime(firms_daily["acq_date"])
    news_daily["acq_date"] = pd.to_datetime(news_daily["acq_date"])

    # Merge ALL news (conflict-region and non-conflict-region) — keep article
    # counts but treat them uniformly going forward.
    news_daily = news_daily.drop(columns=["region_category"])

    monitoring = firms_daily.merge(news_daily, on=["region", "acq_date"], how="left")
    for col in ["news_article_count", "news_unique_source_count",
                "news_conflict_article_count", "news_fire_article_count"]:
        monitoring[col] = monitoring[col].fillna(0).astype(int)

    monitoring = monitoring.sort_values(["region", "acq_date"]).reset_index(drop=True)

    # Total news (used in some plots), kept for backwards compatibility.
    monitoring["news_count_7d"] = (
        monitoring.groupby("region")["news_article_count"]
        .transform(lambda values: values.rolling(window=7, min_periods=1).sum())
    )

    # *** The label-driving signal: conflict-only article rolling sums ***
    monitoring["conflict_news_count_29d_centered"] = (
        monitoring.groupby("region")["news_conflict_article_count"]
        .transform(lambda values: values.rolling(window=29, min_periods=1, center=True).sum())
    )
    monitoring["fire_news_count_29d_centered"] = (
        monitoring.groupby("region")["news_fire_article_count"]
        .transform(lambda values: values.rolling(window=29, min_periods=1, center=True).sum())
    )
    monitoring["news_count_29d_centered"] = (
        monitoring.groupby("region")["news_article_count"]
        .transform(lambda values: values.rolling(window=29, min_periods=1, center=True).sum())
    )

    # Global median of conflict-article activity across ALL regions (uniform threshold)
    global_median = monitoring["conflict_news_count_29d_centered"].median()
    monitoring["global_conflict_news_median"] = global_median
    monitoring["conflict_news_nearby"] = (
        monitoring["conflict_news_count_29d_centered"] > global_median
    ).astype(int)

    monitoring.to_csv(DAILY_MONITORING_OUTPUT, index=False)
    return monitoring


def write_report(
    firms_counts: dict[str, int],
    firms_daily: pd.DataFrame,
    news_counts: dict[str, int],
    news_clean: pd.DataFrame,
    monitoring: pd.DataFrame,
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Data Cleaning Report",
        "",
        "## FIRMS Thermal Data",
        f"- Raw rows processed: {firms_counts['raw_rows']:,}",
        f"- Clean rows retained: {firms_counts['clean_rows']:,}",
        f"- Daily region summary rows: {len(firms_daily):,}",
        f"- Date range: {firms_daily['acq_date'].min()} to {firms_daily['acq_date'].max()}",
        "",
        "## News Data",
        f"- Raw rows processed: {news_counts['raw_rows']:,}",
        f"- Clean rows retained: {news_counts['clean_rows']:,}",
        f"- Date range: {news_clean['publication_date_clean'].min()} to {news_clean['publication_date_clean'].max()}",
        "",
        "## Modeling Summary",
        f"- Daily monitoring rows: {len(monitoring):,}",
        f"- Days with nearby conflict news: {int(monitoring['conflict_news_nearby'].sum()):,}",
        "",
        "## Cleaning Choices",
        "- Removed low-confidence FIRMS detections (`l` / `low`) and invalid geographic, date, brightness, or FRP values.",
        "- Combined standard processed FIRMS data with near-real-time FIRMS data for April-May 2026.",
        "- Removed duplicate news URLs and duplicate same-region/same-date titles.",
        "- Created daily region-level summaries to make notebook analysis and machine learning manageable.",
    ]
    CLEANING_REPORT_OUTPUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Cleaning FIRMS data...", flush=True)
    firms_daily, firms_counts = clean_firms_data()

    print("Cleaning news data...", flush=True)
    news_clean, news_daily, news_counts = clean_news_data()

    print("Creating daily monitoring summary...", flush=True)
    monitoring = create_monitoring_summary(firms_daily, news_daily)

    write_report(firms_counts, firms_daily, news_counts, news_clean, monitoring)

    print("Cleaning complete.", flush=True)
    print(f"FIRMS clean rows: {firms_counts['clean_rows']:,}", flush=True)
    print(f"News clean rows: {news_counts['clean_rows']:,}", flush=True)
    print(f"Daily monitoring rows: {len(monitoring):,}", flush=True)


if __name__ == "__main__":
    main()
