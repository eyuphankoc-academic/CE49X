"""Train an ensemble of models on different 250K subsets, then average predictions.

Why this works:
  A single model trained on 200K rows (80% of the 250K sample) sees only a
  fraction of the full dataset.  Training N models on N DIFFERENT 250K subsets,
  then averaging their probability outputs (soft voting), is a form of manual
  bagging.  Each model sees different examples; errors tend to cancel out.
  The ensemble typically beats any single subset model on the held-out test set.

Design:
  1. All ~5.9M rows (firms_clean.csv) are loaded and labelled.
  2. A fixed 20% of ALL rows is reserved as the test set — no model ever trains
     on this.
  3. N_SUBSETS models are each trained on a DIFFERENT random 250K draw from
     the remaining 80% training pool.
  4. Predictions on the test set are averaged (soft vote).
  5. Results are compared: baseline single model vs. ensemble.

Usage:
    python scripts/train_ensemble_subsets.py
    python scripts/train_ensemble_subsets.py --n-subsets 12 --subset-size 300000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
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
TARGET             = "conflict_news_nearby"
RANDOM_SEED        = 49
CHUNK_SIZE         = 500_000
DECISION_THRESHOLD = 0.30


# ── data loading ──────────────────────────────────────────────────────────────

def load_all_labelled() -> pd.DataFrame:
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

    print("Loading ALL firms_clean.csv rows (this may take a minute)...")
    chunks: list[pd.DataFrame] = []
    total = 0
    for chunk in pd.read_csv(firms_path, chunksize=CHUNK_SIZE, low_memory=False):
        chunks.append(chunk)
        total += len(chunk)
        print(f"  Loaded {total:,} rows...")

    df = pd.concat(chunks, ignore_index=True)
    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
    df = df.merge(monitoring, on=["region", "acq_date"], how="left")
    df[TARGET] = df[TARGET].fillna(0).astype(int)
    print(f"Total rows with labels: {len(df):,}")
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


# ── helpers ───────────────────────────────────────────────────────────────────

def make_classifier(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=200,
        max_leaf_nodes=63,
        min_samples_leaf=50,
        learning_rate=0.05,
        random_state=seed,
        class_weight="balanced",
    )


def print_results(label: str, y_test: np.ndarray, y_proba: np.ndarray) -> float:
    y_pred = (y_proba >= DECISION_THRESHOLD).astype(int)
    auc    = roc_auc_score(y_test, y_proba)
    print(f"\n{'─'*60}")
    print(f"{label}")
    print(f"{'─'*60}")
    print(f"  Test AUC : {auc:.4f}")
    print(classification_report(y_test, y_pred,
                                target_names=["No conflict", "Conflict"], digits=3))
    return auc


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-subsets",   type=int, default=8,
                        help="Number of 250K-subset models to train (default: 8)")
    parser.add_argument("--subset-size", type=int, default=250_000,
                        help="Rows per subset (default: 250,000)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n_subsets   = args.n_subsets
    subset_size = args.subset_size

    df = load_all_labelled()
    df = engineer_features(df)

    ml = df[FEATURES + [TARGET]].dropna().reset_index(drop=True)
    print(f"\nTotal labelled rows : {len(ml):,}")
    print(f"Positive rate       : {ml[TARGET].mean():.1%}")

    # Fixed 20% test set — held out from ALL training
    train_pool, test_df = train_test_split(
        ml, test_size=0.2, random_state=RANDOM_SEED, stratify=ml[TARGET]
    )
    X_test = test_df[FEATURES].values
    y_test = test_df[TARGET].values

    # Fit scaler on the full training pool (not on subsets)
    scaler = StandardScaler()
    scaler.fit(train_pool[FEATURES].values)
    X_test_s = scaler.transform(X_test)

    print(f"\nTraining pool : {len(train_pool):,} rows")
    print(f"Test set      : {len(test_df):,} rows")
    print(f"Subset size   : {subset_size:,} per model")
    print(f"N subsets     : {n_subsets}")

    # ── Baseline: single model on one 250K sample ──────────────────────────
    baseline = train_pool.sample(n=subset_size, random_state=RANDOM_SEED)
    X_b = scaler.transform(baseline[FEATURES].values)
    y_b = baseline[TARGET].values
    clf_base = make_classifier(RANDOM_SEED)
    t0 = time.time()
    clf_base.fit(X_b, y_b)
    print(f"\nBaseline model trained in {time.time()-t0:.1f}s")
    base_proba = clf_base.predict_proba(X_test_s)[:, 1]
    base_auc = print_results(f"BASELINE  (1 model × {subset_size:,} rows)", y_test, base_proba)

    # ── Ensemble: N models on different random subsets ─────────────────────
    print(f"\nTraining {n_subsets} models on different {subset_size:,}-row subsets...")
    all_probas: list[np.ndarray] = []
    individual_aucs: list[float] = []

    for i in range(n_subsets):
        seed_i  = RANDOM_SEED + i + 1
        subset  = train_pool.sample(n=subset_size, random_state=seed_i)
        X_sub   = scaler.transform(subset[FEATURES].values)
        y_sub   = subset[TARGET].values
        clf_i   = make_classifier(seed_i)
        t0 = time.time()
        clf_i.fit(X_sub, y_sub)
        proba_i = clf_i.predict_proba(X_test_s)[:, 1]
        auc_i   = roc_auc_score(y_test, proba_i)
        all_probas.append(proba_i)
        individual_aucs.append(auc_i)
        print(f"  Model {i+1:2d}/{n_subsets}: AUC={auc_i:.4f}  seed={seed_i}  "
              f"time={time.time()-t0:.1f}s")

    ensemble_proba = np.mean(all_probas, axis=0)
    ensemble_auc = print_results(
        f"ENSEMBLE  ({n_subsets} models × {subset_size:,} rows, soft vote)",
        y_test, ensemble_proba,
    )

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Individual model AUCs: min={min(individual_aucs):.4f}  "
          f"mean={np.mean(individual_aucs):.4f}  max={max(individual_aucs):.4f}")
    print(f"  Baseline (1 × {subset_size:,})    : AUC = {base_auc:.4f}")
    print(f"  Ensemble ({n_subsets} × {subset_size:,}): AUC = {ensemble_auc:.4f}")
    delta = (ensemble_auc - base_auc) * 100
    print(f"  Improvement             : {delta:+.2f} AUC points")
    print(
        "\n  Tip: increase --n-subsets for diminishing but real improvement."
        "\n  Tip: use --subset-size 500000 to train each model on more data."
    )


if __name__ == "__main__":
    main()
