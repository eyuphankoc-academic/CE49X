"""Train scalable ML models on up to 4M FIRMS detection rows.

Replaces or supplements the notebook's 250K training with models designed
to handle millions of rows efficiently:

  HistGradientBoostingClassifier  — sklearn's fast gradient boosting (LightGBM-style).
                                    Handles 4M+ rows natively; supports missing values.

  RandomForestClassifier          — parallel tree ensemble (n_jobs=-1).
                                    Good generalisation; memory-heavier than HistGBM.

The label (conflict_news_nearby) is merged from daily_region_monitoring_summary.csv,
the same label used in the notebook.

Usage:
    python scripts/train_scalable_models.py
    python scripts/train_scalable_models.py --max-rows 2000000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED    = PROJECT_ROOT / "data" / "processed"

FEATURES = [
    "latitude", "longitude",
    "bright_ti4", "frp", "log_frp",
    "scan", "track",
    "is_night", "month", "year", "region_enc",
]
TARGET         = "conflict_news_nearby"
RANDOM_SEED    = 49
CHUNK_SIZE     = 500_000
DECISION_THRESHOLD = 0.30


# ── data loading ──────────────────────────────────────────────────────────────

def load_and_label(max_rows: int) -> pd.DataFrame:
    print("Loading daily monitoring labels...")
    monitoring = pd.read_csv(
        PROCESSED / "daily_region_monitoring_summary.csv",
        usecols=["region", "acq_date", "conflict_news_nearby"],
    )
    monitoring["acq_date"] = pd.to_datetime(monitoring["acq_date"])
    monitoring[TARGET] = monitoring[TARGET].fillna(0).astype(int)

    firms_path = PROCESSED / "firms_clean.csv"
    if not firms_path.exists():
        raise FileNotFoundError(
            "firms_clean.csv not found — run python scripts/clean_data.py first."
        )

    print(f"Loading FIRMS clean data (cap: {max_rows:,} rows) in {CHUNK_SIZE:,}-row chunks...")
    chunks: list[pd.DataFrame] = []
    loaded = 0

    for chunk in pd.read_csv(firms_path, chunksize=CHUNK_SIZE, low_memory=False):
        remaining = max_rows - loaded
        if remaining <= 0:
            break
        if len(chunk) > remaining:
            chunk = chunk.sample(n=remaining, random_state=RANDOM_SEED)
        chunks.append(chunk)
        loaded += len(chunk)
        print(f"  Loaded {loaded:,} / {max_rows:,} rows")

    df = pd.concat(chunks, ignore_index=True)
    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
    df = df.merge(monitoring, on=["region", "acq_date"], how="left")
    df[TARGET] = df[TARGET].fillna(0).astype(int)
    print(f"Total rows after merge: {len(df):,}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month"]    = df["acq_date"].dt.month
    df["year"]     = df["acq_date"].dt.year
    df["is_night"] = (df["daynight"].astype(str).str.upper() == "N").astype(int)
    df["log_frp"]  = np.log1p(pd.to_numeric(df["frp"],        errors="coerce").fillna(0))
    df["frp"]      = pd.to_numeric(df["frp"],        errors="coerce").fillna(0)
    df["bright_ti4"] = pd.to_numeric(df["bright_ti4"], errors="coerce")
    df["scan"]     = pd.to_numeric(df["scan"],      errors="coerce")
    df["track"]    = pd.to_numeric(df["track"],     errors="coerce")

    le = LabelEncoder()
    df["region_enc"] = le.fit_transform(df["region"].astype(str))
    return df.dropna(subset=FEATURES)


# ── model trainers ────────────────────────────────────────────────────────────

def run_hist_gradient_boosting(X_train: np.ndarray, y_train: np.ndarray,
                                X_test: np.ndarray, y_test: np.ndarray) -> None:
    print("\n" + "─" * 60)
    print("HistGradientBoostingClassifier")
    print("  Fast gradient boosting — handles millions of rows natively.")
    print("─" * 60)

    t0 = time.time()
    clf = HistGradientBoostingClassifier(
        max_iter=300,
        max_leaf_nodes=63,
        min_samples_leaf=50,
        learning_rate=0.05,
        random_state=RANDOM_SEED,
        class_weight="balanced",
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
    )
    clf.fit(X_train, y_train)
    elapsed = time.time() - t0

    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= DECISION_THRESHOLD).astype(int)
    auc     = roc_auc_score(y_test, y_proba)

    print(f"  Iterations used : {clf.n_iter_}")
    print(f"  Training time   : {elapsed:.1f}s")
    print(f"  Test AUC        : {auc:.4f}")
    print(classification_report(y_test, y_pred,
                                target_names=["No conflict", "Conflict"], digits=3))


def run_random_forest(X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray) -> None:
    print("\n" + "─" * 60)
    print("RandomForestClassifier  (n_jobs=-1, all CPU cores)")
    print("  Parallel tree ensemble — good generalisation at scale.")
    print("─" * 60)

    t0 = time.time()
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=100,
        n_jobs=-1,
        random_state=RANDOM_SEED,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)
    elapsed = time.time() - t0

    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= DECISION_THRESHOLD).astype(int)
    auc     = roc_auc_score(y_test, y_proba)

    print(f"  Training time   : {elapsed:.1f}s")
    print(f"  Test AUC        : {auc:.4f}")
    print(classification_report(y_test, y_pred,
                                target_names=["No conflict", "Conflict"], digits=3))

    importances = pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("  Top feature importances:")
    for feat, imp in importances.head(6).items():
        print(f"    {feat:<20} {imp:.4f}")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train scalable models on large FIRMS data.")
    parser.add_argument("--max-rows", type=int, default=4_000_000,
                        help="Maximum rows to load from firms_clean.csv (default: 4,000,000)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = load_and_label(max_rows=args.max_rows)
    df = engineer_features(df)

    ml = df[FEATURES + [TARGET]].dropna()
    print(f"\nRows after feature engineering : {len(ml):,}")
    print(f"Label distribution:\n{ml[TARGET].value_counts().to_string()}")
    print(f"Positive rate: {ml[TARGET].mean():.1%}")

    X = ml[FEATURES].values
    y = ml[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f"\nTraining set : {len(X_train):,} rows (80%)")
    print(f"Test set     : {len(X_test):,} rows (20%)")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    run_hist_gradient_boosting(X_train_s, y_train, X_test_s, y_test)
    run_random_forest(X_train_s, y_train, X_test_s, y_test)

    print("\nDone.")


if __name__ == "__main__":
    main()
