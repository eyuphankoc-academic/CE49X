"""Set up the PostgreSQL database inside Docker and load all project data.

PREREQUISITES — run this once before executing:
    docker run --name ce49x-postgres \
        -e POSTGRES_USER=ce49x \
        -e POSTGRES_HOST_AUTH_METHOD=trust \
        -e POSTGRES_DB=conflict_monitoring \
        -p 5432:5432 \
        -d postgres:16

Then run this script:
    python scripts/setup_database.py

The script will create the 4 required tables and load all processed data.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"

DB_URL = "postgresql://ce49x@localhost:5432/conflict_monitoring"
CONTAINER_NAME = "ce49x-postgres"
CHUNK_SIZE = 50_000


# ── Docker helpers ─────────────────────────────────────────────────────────────

def docker_is_running() -> bool:
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name={CONTAINER_NAME}", "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=15,
        )
        return CONTAINER_NAME in result.stdout
    except FileNotFoundError:
        return False


def start_container() -> None:
    result = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name={CONTAINER_NAME}", "--format", "{{.Names}}"],
        capture_output=True, text=True, timeout=15,
    )
    if CONTAINER_NAME in result.stdout:
        print(f"Container '{CONTAINER_NAME}' exists, starting...")
        subprocess.run(["docker", "start", CONTAINER_NAME], check=True)
    else:
        print(f"Creating container '{CONTAINER_NAME}'...")
        subprocess.run([
            "docker", "run", "--name", CONTAINER_NAME,
            "-e", "POSTGRES_USER=ce49x",
            "-e", "POSTGRES_HOST_AUTH_METHOD=trust",
            "-e", "POSTGRES_DB=conflict_monitoring",
            "-p", "5432:5432",
            "-d", "postgres:16",
        ], check=True)

    print("Waiting for PostgreSQL to accept connections...")
    for attempt in range(30):
        time.sleep(2)
        result = subprocess.run(
            ["docker", "exec", CONTAINER_NAME, "pg_isready", "-U", "ce49x"],
            capture_output=True, text=True,
        )
        if "accepting connections" in result.stdout:
            print("PostgreSQL is ready.")
            return
    raise RuntimeError("PostgreSQL did not become ready within 60 seconds.")


# ── Table schemas ──────────────────────────────────────────────────────────────

CREATE_FIRMS_DETECTIONS = """
CREATE TABLE IF NOT EXISTS firms_detections (
    id              SERIAL PRIMARY KEY,
    region          TEXT NOT NULL,
    latitude        DOUBLE PRECISION NOT NULL,
    longitude       DOUBLE PRECISION NOT NULL,
    acq_date        DATE NOT NULL,
    acq_time        INTEGER,
    acquisition_datetime_utc TIMESTAMPTZ,
    satellite       TEXT,
    instrument      TEXT,
    confidence      TEXT,
    bright_ti4      DOUBLE PRECISION,
    bright_ti5      DOUBLE PRECISION,
    frp             DOUBLE PRECISION,
    daynight        CHAR(1),
    firms_source    TEXT,
    source_period   TEXT
);
"""

CREATE_NEWS_ARTICLES = """
CREATE TABLE IF NOT EXISTS news_articles (
    id                      SERIAL PRIMARY KEY,
    region                  TEXT NOT NULL,
    title                   TEXT,
    publication_datetime_utc TIMESTAMPTZ,
    publication_date_clean  DATE,
    source                  TEXT,
    domain                  TEXT,
    url                     TEXT UNIQUE,
    language                TEXT,
    news_source_dataset     TEXT
);
"""

CREATE_THERMAL_EVENTS = """
CREATE TABLE IF NOT EXISTS thermal_events (
    id                      SERIAL PRIMARY KEY,
    region                  TEXT NOT NULL,
    acq_date                DATE NOT NULL,
    firms_detection_count   INTEGER,
    firms_mean_bright_ti4   DOUBLE PRECISION,
    firms_max_bright_ti4    DOUBLE PRECISION,
    firms_mean_frp          DOUBLE PRECISION,
    firms_max_frp           DOUBLE PRECISION,
    firms_total_frp         DOUBLE PRECISION,
    firms_mean_latitude     DOUBLE PRECISION,
    firms_mean_longitude    DOUBLE PRECISION,
    firms_day_count         INTEGER,
    firms_night_count       INTEGER
);
"""

CREATE_EVENT_MATCHES = """
CREATE TABLE IF NOT EXISTS event_matches (
    id                          SERIAL PRIMARY KEY,
    region                      TEXT NOT NULL,
    acq_date                    DATE NOT NULL,
    firms_detection_count       INTEGER,
    firms_mean_bright_ti4       DOUBLE PRECISION,
    firms_max_bright_ti4        DOUBLE PRECISION,
    firms_mean_frp              DOUBLE PRECISION,
    firms_max_frp               DOUBLE PRECISION,
    firms_total_frp             DOUBLE PRECISION,
    firms_mean_latitude         DOUBLE PRECISION,
    firms_mean_longitude        DOUBLE PRECISION,
    firms_day_count             INTEGER,
    firms_night_count           INTEGER,
    news_article_count          INTEGER,
    news_unique_source_count    INTEGER,
    news_count_7d               INTEGER,
    conflict_news_nearby        SMALLINT
);
"""


# ── Load & push data ───────────────────────────────────────────────────────────

def load_firms_detections(engine) -> None:
    path = PROCESSED / "firms_model_sample.csv"
    print(f"Loading firms_detections from {path.name}...")
    keep_cols = [
        "region", "latitude", "longitude", "acq_date", "acq_time",
        "acquisition_datetime_utc", "satellite", "instrument", "confidence",
        "bright_ti4", "bright_ti5", "frp", "daynight", "firms_source", "source_period",
    ]
    df = pd.read_csv(path, usecols=[c for c in keep_cols if c in pd.read_csv(path, nrows=0).columns])
    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce").dt.date
    df["acquisition_datetime_utc"] = pd.to_datetime(df.get("acquisition_datetime_utc"), errors="coerce")
    df["acq_time"] = pd.to_numeric(df.get("acq_time"), errors="coerce").astype("Int64")
    df["daynight"] = df["daynight"].astype(str).str.strip().str.upper().str[:1].replace("N", pd.NA)
    df = df.dropna(subset=["latitude", "longitude", "acq_date"])

    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE firms_detections RESTART IDENTITY CASCADE;"))
    for i in range(0, len(df), CHUNK_SIZE):
        df.iloc[i:i + CHUNK_SIZE].to_sql(
            "firms_detections", engine, if_exists="append", index=False,
        )
        print(f"  firms_detections: {min(i + CHUNK_SIZE, len(df)):,}/{len(df):,} rows inserted")
    print(f"firms_detections done: {len(df):,} rows")


def load_news_articles(engine) -> None:
    path = PROCESSED / "news_clean.csv"
    print(f"Loading news_articles from {path.name}...")
    df = pd.read_csv(path)
    df["publication_datetime_utc"] = pd.to_datetime(df.get("publication_datetime_utc"), errors="coerce")
    df["publication_date_clean"] = pd.to_datetime(df.get("publication_date_clean"), errors="coerce").dt.date
    keep_cols = [
        "region", "title", "publication_datetime_utc", "publication_date_clean",
        "source", "domain", "url", "language", "news_source_dataset",
    ]
    df = df[[c for c in keep_cols if c in df.columns]].dropna(subset=["url"])
    df = df.drop_duplicates(subset=["url"])

    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE news_articles RESTART IDENTITY CASCADE;"))
    for i in range(0, len(df), 500):
        df.iloc[i:i + 500].to_sql("news_articles", engine, if_exists="append", index=False)
    print(f"news_articles done: {len(df):,} rows")


def load_thermal_events(engine) -> None:
    path = PROCESSED / "firms_daily_region_summary.csv"
    print(f"Loading thermal_events from {path.name}...")
    df = pd.read_csv(path)
    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce").dt.date
    for col in ["firms_day_count", "firms_night_count", "firms_detection_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE thermal_events RESTART IDENTITY CASCADE;"))
    for i in range(0, len(df), 500):
        df.iloc[i:i + 500].to_sql("thermal_events", engine, if_exists="append", index=False)
    print(f"thermal_events done: {len(df):,} rows")


def load_event_matches(engine) -> None:
    path = PROCESSED / "daily_region_monitoring_summary.csv"
    print(f"Loading event_matches from {path.name}...")
    df = pd.read_csv(path)
    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce").dt.date
    for col in ["firms_detection_count", "firms_day_count", "firms_night_count",
                "news_article_count", "news_count_7d", "conflict_news_nearby"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE event_matches RESTART IDENTITY CASCADE;"))
    for i in range(0, len(df), 500):
        df.iloc[i:i + 500].to_sql("event_matches", engine, if_exists="append", index=False)
    print(f"event_matches done: {len(df):,} rows")


# ── Verify ─────────────────────────────────────────────────────────────────────

def verify(engine) -> None:
    tables = ["firms_detections", "news_articles", "thermal_events", "event_matches"]
    print("\nVerification:")
    with engine.connect() as conn:
        for table in tables:
            count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            print(f"  {table}: {count:,} rows")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    if not docker_is_running():
        print("Docker container not running. Starting it now...")
        start_container()

    print(f"\nConnecting to: {DB_URL}")
    engine = create_engine(DB_URL, future=True)

    print("Creating tables...")
    with engine.begin() as conn:
        # Drop and recreate to ensure schema matches current definition
        for tbl in ["event_matches", "thermal_events", "news_articles", "firms_detections"]:
            conn.execute(text(f"DROP TABLE IF EXISTS {tbl} CASCADE;"))
        for ddl in [CREATE_FIRMS_DETECTIONS, CREATE_NEWS_ARTICLES,
                    CREATE_THERMAL_EVENTS, CREATE_EVENT_MATCHES]:
            conn.execute(text(ddl))
    print("Tables created.")

    load_firms_detections(engine)
    load_news_articles(engine)
    load_thermal_events(engine)
    load_event_matches(engine)

    verify(engine)
    print("\nDatabase setup complete.")


if __name__ == "__main__":
    main()
