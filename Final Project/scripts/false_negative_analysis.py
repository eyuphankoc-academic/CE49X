"""False-negative dissection for the Decision Tree conflict classifier.

Replicates the notebook's exact training recipe (Cell 18/19), isolates the
test-set false negatives (true label = conflict, model said NON-conflict at
the 0.30 threshold), picks one illustrative case, and traces the exact
decision path that node-by-node drove the tree to the wrong call.

Run:  python scripts/false_negative_analysis.py
Writes: figures/false_negative_case.json   (machine-readable result)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"

FEATURES = [
    "latitude", "longitude", "bright_ti4", "frp", "log_frp",
    "scan", "track", "is_night", "month", "year", "region_enc",
]
TARGET = "conflict_news_nearby"
SEED = 49
THRESHOLD = 0.30

# ── replicate notebook feature engineering ──────────────────────────────────
sample = pd.read_csv(PROCESSED / "firms_model_sample.csv", low_memory=False)
monitoring = pd.read_csv(PROCESSED / "daily_region_monitoring_summary.csv")

firms = sample.copy()
firms["acq_date"] = pd.to_datetime(firms["acq_date"], errors="coerce")
firms["month"] = firms["acq_date"].dt.month
firms["year"] = firms["acq_date"].dt.year
firms["is_night"] = (firms["daynight"].astype(str).str.upper() == "N").astype(int)
firms["log_frp"] = np.log1p(pd.to_numeric(firms["frp"], errors="coerce").fillna(0))
firms["frp"] = pd.to_numeric(firms["frp"], errors="coerce").fillna(0)
firms["region_enc"] = pd.Categorical(firms["region"]).codes
for c in ["bright_ti4", "scan", "track"]:
    firms[c] = pd.to_numeric(firms[c], errors="coerce")

monitoring["acq_date"] = pd.to_datetime(monitoring["acq_date"], errors="coerce")
firms = firms.merge(
    monitoring[["region", "acq_date", TARGET]], on=["region", "acq_date"], how="left"
)
firms[TARGET] = firms[TARGET].fillna(0).astype(int)

# keep human-readable columns alongside the model matrix
KEEP = FEATURES + [TARGET, "region", "region_category", "acq_date",
                   "daynight", "bright_ti5", "confidence", "satellite"]
ml = firms[KEEP].dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)

X = ml[FEATURES].values
y = ml[TARGET].values
idx = np.arange(len(ml))

X_tr, X_te, y_tr, y_te, _, idx_te = train_test_split(
    X, y, idx, test_size=0.2, random_state=SEED, stratify=y
)

clf = DecisionTreeClassifier(
    max_depth=8, min_samples_leaf=50, class_weight="balanced", random_state=SEED
)
clf.fit(X_tr, y_tr)

proba = clf.predict_proba(X_te)[:, 1]
pred = (proba >= THRESHOLD).astype(int)

# ── isolate false negatives (true conflict, predicted non-conflict) ─────────
fn_mask = (y_te == 1) & (pred == 0)
n_fn = int(fn_mask.sum())
n_pos = int((y_te == 1).sum())
print(f"Test set: {len(y_te):,} rows | positives: {n_pos:,} | "
      f"false negatives: {n_fn:,} ({100*n_fn/max(n_pos,1):.1f}% of positives missed)")

fn_positions = np.where(fn_mask)[0]

# pick an illustrative, *confident* mistake: the FN with the LOWEST conflict
# probability — the case the model was most sure was peaceful but wasn't.
fn_proba = proba[fn_positions]
chosen = fn_positions[int(np.argmin(fn_proba))]
row = ml.iloc[idx_te[chosen]]
x_row = X_te[chosen].reshape(1, -1)
p_conflict = float(proba[chosen])

print("\n" + "=" * 64)
print("CHOSEN FALSE NEGATIVE (most confident wrong 'non-conflict' call)")
print("=" * 64)
print(f"  Region        : {row['region']}  ({row['region_category']})")
print(f"  Date          : {row['acq_date'].date()}")
print(f"  Lat, Lon      : {row['latitude']:.3f}, {row['longitude']:.3f}")
print(f"  bright_ti4 (K): {row['bright_ti4']:.1f}")
print(f"  FRP (MW)      : {row['frp']:.2f}   (log_frp={row['log_frp']:.2f})")
print(f"  scan/track    : {row['scan']:.2f} / {row['track']:.2f}")
print(f"  day/night     : {row['daynight']}")
print(f"  TRUE label    : {int(row[TARGET])}  (CONFLICT)")
print(f"  Model P(conf) : {p_conflict:.3f}  < {THRESHOLD}  -> predicted NON-CONFLICT")

# ── trace the decision path through the tree ────────────────────────────────
feat = clf.tree_.feature
thr = clf.tree_.threshold
node_path = clf.decision_path(x_row).indices
leaf = clf.apply(x_row)[0]

print("\nDECISION PATH (node-by-node):")
steps = []
for node in node_path:
    if feat[node] == -2:  # leaf
        val = clf.tree_.value[node][0]
        # class_weight balanced -> value holds weighted counts; normalise
        leaf_p = val[1] / val.sum()
        steps.append({"node": int(node), "leaf": True, "leaf_prob_conflict": float(leaf_p)})
        print(f"  -> LEAF {node}: weighted P(conflict)={leaf_p:.3f}")
        continue
    fname = FEATURES[feat[node]]
    fval = float(x_row[0, feat[node]])
    goes_left = fval <= thr[node]
    direction = "<=" if goes_left else ">"
    print(f"  node {node:>3}: {fname} = {fval:.3f}  {direction} {thr[node]:.3f}"
          f"   -> go {'LEFT' if goes_left else 'RIGHT'}")
    steps.append({
        "node": int(node), "feature": fname, "value": fval,
        "threshold": float(thr[node]), "decision": direction,
        "went": "left" if goes_left else "right",
    })

# how many training samples landed in this leaf, and their real label mix
leaf_train = clf.apply(X_tr)
in_leaf = leaf_train == leaf
print(f"\nLEAF COMPOSITION (training rows that ended in this same leaf):")
print(f"  total train rows in leaf : {int(in_leaf.sum()):,}")
print(f"  actually conflict (y=1)  : {int(y_tr[in_leaf].sum()):,} "
      f"({100*y_tr[in_leaf].mean():.1f}%)")

out = {
    "test_rows": int(len(y_te)),
    "test_positives": n_pos,
    "false_negatives": n_fn,
    "fn_rate_of_positives": round(100 * n_fn / max(n_pos, 1), 2),
    "case": {
        "region": str(row["region"]),
        "region_category": str(row["region_category"]),
        "date": str(row["acq_date"].date()),
        "latitude": float(row["latitude"]),
        "longitude": float(row["longitude"]),
        "bright_ti4": float(row["bright_ti4"]),
        "frp": float(row["frp"]),
        "log_frp": float(row["log_frp"]),
        "scan": float(row["scan"]),
        "track": float(row["track"]),
        "daynight": str(row["daynight"]),
        "true_label": int(row[TARGET]),
        "model_prob_conflict": round(p_conflict, 4),
        "threshold": THRESHOLD,
        "predicted": "non-conflict",
    },
    "decision_path": steps,
    "leaf_train_rows": int(in_leaf.sum()),
    "leaf_train_conflict_share": round(float(y_tr[in_leaf].mean()), 4),
}
FIGURES.mkdir(exist_ok=True)
(FIGURES / "false_negative_case.json").write_text(json.dumps(out, indent=2))
print("\nWrote figures/false_negative_case.json")
