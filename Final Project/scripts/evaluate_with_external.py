"""Evaluate model labels against external independent ground truth.

Sources supported (use whatever you have):
  GDELT Event Database  — fetch_gdelt_events.py   (no key, covers 2024-2026)
  ACLED                 — fetch_acled_data.py      (needs API key)
  UCDP GED              — fetch_ucdp_data.py       (covers up to 2023 only)

This script answers the core question:
    "Was conflict_news_nearby=1 actually correct according to real conflict
     databases that are INDEPENDENT of the news used to create the label?"

It performs a date-level match: for each (region, date) in the monitoring
summary, it checks whether an external source recorded ANY conflict event in
that region within +-WINDOW_DAYS days.

Run at least:
    python scripts/fetch_gdelt_events.py

Usage:
    python scripts/evaluate_with_external.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
RAW      = PROJECT_ROOT / "data" / "raw"
FIGURES  = PROJECT_ROOT / "figures"

WINDOW_DAYS = 3   # ±3 days tolerance for date matching


# ── loaders ───────────────────────────────────────────────────────────────────

def load_monitoring() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED / "daily_region_monitoring_summary.csv")
    df["acq_date"] = pd.to_datetime(df["acq_date"])
    df["conflict_news_nearby"] = df["conflict_news_nearby"].fillna(0).astype(int)
    return df


def load_acled() -> pd.DataFrame | None:
    path = RAW / "acled_events_raw.csv"
    if not path.exists():
        print("  ACLED file not found — run fetch_acled_data.py first.")
        return None
    df = pd.read_csv(path, low_memory=False)
    df["event_date"] = pd.to_datetime(df.get("event_date"), errors="coerce")
    df = df.rename(columns={"project_region": "region"})
    for col in ("latitude", "longitude"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["event_date", "region"])


def load_ucdp() -> pd.DataFrame | None:
    path = RAW / "ucdp_events_raw.csv"
    if not path.exists():
        print("  UCDP file not found — run fetch_ucdp_data.py first.")
        return None
    df = pd.read_csv(path, low_memory=False)
    for date_col in ("event_date", "date_start"):
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if "event_date" not in df.columns and "date_start" in df.columns:
        df["event_date"] = df["date_start"]
    df = df.rename(columns={"project_region": "region"})
    return df.dropna(subset=["event_date", "region"])


def load_gdelt() -> pd.DataFrame | None:
    path = RAW / "gdelt_conflict_events_raw.csv"
    if not path.exists():
        print("  GDELT file not found — run fetch_gdelt_events.py first.")
        return None
    df = pd.read_csv(path, low_memory=False)
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df = df.rename(columns={"project_region": "region"})
    return df.dropna(subset=["event_date", "region"])


# ── ground-truth builder ───────────────────────────────────────────────────────

def build_ground_truth(events: pd.DataFrame, window: int = WINDOW_DAYS) -> pd.DataFrame:
    """Expand each event into a ±window_days presence window.

    Returns a deduplicated DataFrame with columns [region, acq_date]
    where acq_date is every date within ±window days of an event.
    """
    rows = []
    for _, ev in events.iterrows():
        region = ev["region"]
        date   = ev["event_date"]
        if pd.isna(date):
            continue
        for delta in range(-window, window + 1):
            rows.append({"region": region, "acq_date": date + pd.Timedelta(days=delta)})

    if not rows:
        return pd.DataFrame(columns=["region", "acq_date"])

    return (
        pd.DataFrame(rows)
        .drop_duplicates()
        .assign(external_event=True)
    )


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(monitoring: pd.DataFrame, ground_truth: pd.DataFrame,
             source_name: str, ax_cm=None) -> dict:
    merged = monitoring.merge(ground_truth, on=["region", "acq_date"], how="left")
    merged["external_event"] = merged["external_event"].fillna(False).astype(bool)

    # Evaluate only on conflict regions — non-conflict reference regions are
    # designed to have no conflict events and should all be True Negatives.
    conflict_rows = merged[merged["region_category"] == "conflict"].copy()

    y_true = conflict_rows["external_event"].astype(int).values
    y_pred = conflict_rows["conflict_news_nearby"].fillna(0).astype(int).values

    # AUC requires probabilistic scores; use y_pred directly (it's 0/1 here)
    auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else float("nan")

    print(f"\n{'='*65}")
    print(f"  Evaluation vs {source_name}")
    print(f"{'='*65}")
    print(f"  Conflict-region rows : {len(conflict_rows):,}")
    print(f"  Days with real event : {y_true.sum():,}  ({100*y_true.mean():.1f}%)")
    print(f"  Days model said 1    : {y_pred.sum():,}  ({100*y_pred.mean():.1f}%)")
    if not np.isnan(auc):
        print(f"  AUC                  : {auc:.4f}")
    print()
    print(classification_report(y_true, y_pred, target_names=["No event", "Event"],
                                digits=3))

    cm = confusion_matrix(y_true, y_pred)
    print(f"  Confusion matrix (rows=ground truth, cols=model prediction):")
    print(f"               Pred 0   Pred 1")
    print(f"  True 0 (TN/FP): {cm[0,0]:6,}   {cm[0,1]:6,}")
    print(f"  True 1 (FN/TP): {cm[1,0]:6,}   {cm[1,1]:6,}")

    if ax_cm is not None:
        disp = ConfusionMatrixDisplay(cm, display_labels=["No event", "Event"])
        disp.plot(ax=ax_cm, colorbar=False, cmap="Blues")
        ax_cm.set_title(f"vs {source_name}", fontsize=9)

    return {"source": source_name, "auc": auc,
            "n_rows": len(conflict_rows), "n_events": y_true.sum(),
            "cm": cm}


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    monitoring = load_monitoring()
    print(f"Monitoring rows loaded: {len(monitoring):,}")
    print(f"  Conflict rows: {(monitoring['region_category']=='conflict').sum():,}")

    acled = load_acled()
    ucdp  = load_ucdp()
    gdelt = load_gdelt()

    sources = []
    if gdelt is not None and len(gdelt):
        sources.append(("GDELT (2024-2026)", gdelt))
    if acled is not None and len(acled):
        sources.append(("ACLED", acled))
    if ucdp is not None and len(ucdp):
        # Warn if UCDP dates don't overlap with monitoring
        ucdp_max = ucdp["event_date"].max()
        monitoring = load_monitoring()
        mon_min = monitoring["acq_date"].min()
        if ucdp_max < mon_min:
            print(f"  Warning: UCDP data ends {ucdp_max.date()} but monitoring starts "
                  f"{mon_min.date()} — no date overlap, UCDP results will be meaningless.")
        sources.append(("UCDP (up to 2023)", ucdp))

    if not sources:
        print("\nNo external data available.")
        print("Run: python scripts/fetch_gdelt_events.py")
        return

    n_plots = len(sources)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    all_results = []
    for (name, events), ax in zip(sources, axes):
        gt = build_ground_truth(events, window=WINDOW_DAYS)
        result = evaluate(monitoring, gt, name, ax_cm=ax)
        all_results.append(result)

    plt.suptitle(
        f"Model label vs external ground truth  (±{WINDOW_DAYS}-day window)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    out_path = FIGURES / "external_ground_truth_evaluation.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved -> {out_path}")

    print("\n\nSummary table:")
    print(f"{'Source':<22} {'Rows':>8} {'Events':>8} {'AUC':>8}")
    print("-" * 52)
    for r in all_results:
        auc_str = f"{r['auc']:.4f}" if not (r['auc'] != r['auc']) else "  n/a "
        print(f"  {r['source']:<20} {r['n_rows']:>8,} {r['n_events']:>8,} {auc_str:>8}")


if __name__ == "__main__":
    main()
