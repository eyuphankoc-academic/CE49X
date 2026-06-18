"""
compare_methods.py — Train identical models on Method 1 and Method 2 labels
and produce a side-by-side comparison.

Method 1: conflict_news_nearby  (region + date news-proximity label)
Method 2: conflict_gdelt_nearby (point-level GDELT event spatial match)

Run build_gdelt_labels.py first to generate data/processed/firms_method2_labeled.csv

Usage:
    python scripts/compare_methods.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED    = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR  = PROJECT_ROOT / "figures"

LABELED_CSV  = PROCESSED / "firms_method2_labeled.csv"

RANDOM_SEED  = 49
DECISION_THR = 0.30
TEST_SIZE    = 0.20

FEATURES = [
    "latitude", "longitude",
    "bright_ti4", "frp", "log_frp",
    "scan", "track",
    "is_night", "month", "year", "region_enc",
]

MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED, C=0.1
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=20, min_samples_leaf=50,
        class_weight="balanced", random_state=RANDOM_SEED,
    ),
    "Naive Bayes": GaussianNB(),
    "SVM Linear": CalibratedClassifierCV(
        LinearSVC(class_weight="balanced", max_iter=2000, random_state=RANDOM_SEED)
    ),
}


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month"]    = df["acq_date"].pipe(pd.to_datetime).dt.month
    df["year"]     = df["acq_date"].pipe(pd.to_datetime).dt.year
    df["is_night"] = (df["daynight"].astype(str).str.upper() == "N").astype(int)
    df["log_frp"]  = np.log1p(pd.to_numeric(df["frp"],        errors="coerce").fillna(0))
    df["frp"]      = pd.to_numeric(df["frp"],        errors="coerce").fillna(0)
    df["bright_ti4"] = pd.to_numeric(df["bright_ti4"], errors="coerce")
    df["scan"]     = pd.to_numeric(df["scan"],       errors="coerce")
    df["track"]    = pd.to_numeric(df["track"],      errors="coerce")
    le = LabelEncoder()
    df["region_enc"] = le.fit_transform(df["region"].astype(str))
    return df


def train_evaluate(X_train, X_test, y_train, y_test, label: str) -> dict:
    """Train all 4 models; return results dict."""
    scaler = StandardScaler()
    Xtr_s  = scaler.fit_transform(X_train)
    Xte_s  = scaler.transform(X_test)

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    for name, clf in MODELS.items():
        clf.fit(Xtr_s, y_train)
        proba = clf.predict_proba(Xte_s)[:, 1]
        pred  = (proba >= DECISION_THR).astype(int)
        auc   = roc_auc_score(y_test, proba)
        cv_auc = cross_val_score(clf, Xtr_s, y_train, cv=cv, scoring="roc_auc").mean()
        cm = confusion_matrix(y_test, pred)
        rep = classification_report(y_test, pred, output_dict=True, zero_division=0)

        results[name] = {
            "auc":       auc,
            "cv_auc":    cv_auc,
            "recall":    rep["1"]["recall"],
            "precision": rep["1"]["precision"],
            "f1":        rep["1"]["f1-score"],
            "accuracy":  rep["accuracy"],
            "proba":     proba,
            "pred":      pred,
            "cm":        cm,
        }
        print(f"    {name:22s}  AUC={auc:.4f}  Recall={rep['1']['recall']:.3f}  "
              f"Prec={rep['1']['precision']:.3f}  F1={rep['1']['f1-score']:.3f}  "
              f"[{label}]")
    return results


def plot_comparison(res1: dict, res2: dict, y_test1, y_test2) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    model_names = list(MODELS.keys())
    metrics = ["auc", "recall", "precision", "f1"]
    metric_labels = ["AUC", "Recall", "Precision", "F1"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(
        "Method 1 (News Region+Date)  vs  Method 2 (GDELT Point-Level)\n"
        "Same 4 models · Same 250K sample · Threshold = 0.30",
        fontsize=13, fontweight="bold", y=1.01,
    )

    colors = {"Method 1": "#3b82f6", "Method 2": "#f97316"}

    for ax, metric, mlabel in zip(axes.flat, metrics, metric_labels):
        x = np.arange(len(model_names))
        w = 0.35
        v1 = [res1[m][metric] for m in model_names]
        v2 = [res2[m][metric] for m in model_names]

        bars1 = ax.bar(x - w/2, v1, w, label="Method 1 (news)", color=colors["Method 1"],
                       edgecolor="white", alpha=0.9)
        bars2 = ax.bar(x + w/2, v2, w, label="Method 2 (GDELT)", color=colors["Method 2"],
                       edgecolor="white", alpha=0.9)

        for bar in bars1 + bars2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7.5)

        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=12, ha="right", fontsize=9)
        ax.set_ylabel(mlabel)
        ax.set_title(mlabel, fontweight="bold")
        ax.set_ylim(0, 1.12)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = FIGURES_DIR / "method_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out}")

    # ROC curves — Decision Tree only, both methods
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    for method_label, res, y_test, color in [
        ("Method 1 — News Region+Date", res1, y_test1, "#3b82f6"),
        ("Method 2 — GDELT Point-Level", res2, y_test2, "#f97316"),
    ]:
        for mname in ["Decision Tree", "Logistic Regression"]:
            fpr, tpr, _ = roc_curve(y_test, res[mname]["proba"])
            auc = res[mname]["auc"]
            ls = "-" if mname == "Decision Tree" else "--"
            ax2.plot(fpr, tpr, color=color, lw=2, ls=ls,
                     label=f"{method_label} — {mname} (AUC={auc:.3f})")

    ax2.plot([0, 1], [0, 1], "k:", alpha=0.5, label="Random (0.5)")
    ax2.axvline(DECISION_THR, color="grey", ls="--", alpha=0.5,
                label=f"threshold={DECISION_THR}")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curves — Method 1 vs Method 2\n(Decision Tree and Logistic Regression)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    out2 = FIGURES_DIR / "method_comparison_roc.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out2}")


def print_summary_table(res1: dict, res2: dict) -> None:
    rows = []
    for name in MODELS.keys():
        for method, res in [("Method 1 (news)", res1), ("Method 2 (GDELT)", res2)]:
            rows.append({
                "Model": name,
                "Method": method,
                "AUC": f"{res[name]['auc']:.4f}",
                "Recall": f"{res[name]['recall']:.3f}",
                "Precision": f"{res[name]['precision']:.3f}",
                "F1": f"{res[name]['f1']:.3f}",
                "Accuracy": f"{res[name]['accuracy']:.3f}",
            })
    df = pd.DataFrame(rows)
    print("\n" + "="*80)
    print("FULL COMPARISON TABLE")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)


def main() -> None:
    if not LABELED_CSV.exists():
        raise SystemExit(
            f"Labeled CSV not found: {LABELED_CSV}\n"
            "Run:  python scripts/build_gdelt_labels.py"
        )

    print("Loading labeled dataset...")
    df = pd.read_csv(LABELED_CSV, low_memory=False)
    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
    df = prepare_features(df)

    print(f"  Total rows: {len(df):,}")
    print(f"  Method 1 positive rate: {df['conflict_news_nearby'].mean():.1%}")
    print(f"  Method 2 positive rate: {df['conflict_gdelt_nearby'].mean():.1%}")

    # Drop rows missing features
    ml = df[FEATURES + ["conflict_news_nearby", "conflict_gdelt_nearby"]].dropna()
    print(f"  Rows with full features: {len(ml):,}")

    X = ml[FEATURES].values
    y1 = ml["conflict_news_nearby"].values
    y2 = ml["conflict_gdelt_nearby"].values

    X_tr1, X_te1, y_tr1, y_te1 = train_test_split(
        X, y1, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y1
    )
    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
        X, y2, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y2
    )

    print("\n-- Training on Method 1 (news label) --")
    res1 = train_evaluate(X_tr1, X_te1, y_tr1, y_te1, "M1")

    print("\n-- Training on Method 2 (GDELT point-level) --")
    res2 = train_evaluate(X_tr2, X_te2, y_tr2, y_te2, "M2")

    print_summary_table(res1, res2)
    plot_comparison(res1, res2, y_te1, y_te2)


if __name__ == "__main__":
    main()
