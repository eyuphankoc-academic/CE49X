"""Generate a visual project-progress dashboard PNG."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "figures"


MILESTONES = [
    ("Define project scope & requirements",                       "done", "Day 1"),
    ("Set up project structure (scripts, notebooks, config)",     "done", "Day 1"),
    ("Get NASA FIRMS API key",                                    "done", "Day 1"),
    ("Define 6 geopolitical conflict regions",                    "done", "Day 1"),
    ("Collect FIRMS thermal data — 2024 (conflict regions)",      "done", "Day 2"),
    ("Collect FIRMS thermal data — 2025-2026 (conflict regions)", "done", "Day 2"),
    ("Collect news (GDELT for 2024)",                             "done", "Day 2"),
    ("Collect news (Google News RSS for 2025-2026)",              "done", "Day 2"),
    ("Initial data cleaning & summarisation",                     "done", "Day 3"),
    ("Spatial & temporal analysis + DBSCAN clustering",           "done", "Day 3"),
    ("Build initial ML model (regional daily aggregates)",        "done", "Day 4"),
    ("Generate dashboard figures",                                "done", "Day 4"),
    ("Install Docker Desktop + WSL2",                             "done", "Day 5"),
    ("Set up PostgreSQL container",                               "done", "Day 5"),
    ("Load all data into PostgreSQL database",                    "done", "Day 5"),
    ("Add DB verification section to notebook",                   "done", "Day 5"),
    ("REWRITE ML at individual-detection level",                  "done", "Day 6"),
    ("Use required models: LogReg, Decision Tree, NB, SVM",       "done", "Day 6"),
    ("Expand news keywords (12 -> 38 terms)",                     "done", "Day 7"),
    ("Re-collect news with broader keywords",                     "done", "Day 7"),
    ("Widen news window to +/- 14 days",                          "done", "Day 7"),
    ("Lower FIRMS confidence threshold (more data kept)",         "done", "Day 7"),
    ("Use median-based balanced labeling",                        "done", "Day 7"),
    ("Lower ML decision threshold to 0.30 (higher recall)",       "done", "Day 7"),
    ("Re-execute notebook with all improvements",                 "done", "Day 7"),
    ("Define 5 non-conflict fire reference regions",              "done", "Day 8"),
    ("Build fire-keyword config (wildfire / bushfire / peat)",    "done", "Day 8"),
    ("Collect FIRMS for non-conflict regions (~2M detections)",   "done", "Day 8"),
    ("Re-clean & generate TP/TN/FP/FN-aware labels",              "done", "Day 8"),
    ("Re-train ML with confirmed True Negatives",                 "done", "Day 8"),
    ("Build interactive 3D rotatable globe dashboard",            "done", "Day 8"),
    ("Collect news for non-conflict regions (~31k articles)",     "done", "Day 8"),
    ("False-positive keyword analysis (25.1% FP rate)",           "done", "Day 8"),
    ("Switch to uniform data-driven labelling (region-agnostic)", "done", "Day 8"),
    ("Helper script to open globe in browser",                    "done", "Day 8"),
    ("Write final discussion / reflection section",               "done", "Day 9"),
    ("Prepare presentation / slides",                             "todo", "AHEAD"),
    ("Submit final project",                                      "todo", "AHEAD"),
]


METRICS = [
    ("FIRMS thermal detections (clean)", "5,988,583", "rows"),
    ("Conflict region detections",       "3,927,695", "war zones"),
    ("Non-conflict region detections",   "2,060,888", "wildfires (TN)"),
    ("News articles collected (clean)",  "85,697",    "conflict + non-conflict"),
    ("Regions monitored",                "11",        "6 conflict + 5 wildfire"),
    ("Date range covered",               "2024-01-01 -> 2026-05-24", ""),
    ("ML training samples",              "200,000",   "individual detections"),
    ("ML test samples",                  "50,000",    "individual detections"),
    ("Best model",                       "Decision Tree", "AUC = 0.942"),
    ("Conflict-event recall",            "97%",       "threshold = 0.30"),
    ("Overall accuracy",                 "69%",       "threshold = 0.30"),
    ("News keywords used",               "38",        "conflict terms"),
    ("Fire keywords (non-conflict)",     "38",        "wildfire terms"),
    ("News window",                      "+/- 14 days", "centred on detection"),
    ("Database",                         "PostgreSQL", "via Docker container"),
    ("Interactive dashboards",           "2",         "HTML globe + dashboard PNG"),
    ("Output figures generated",         "10+",       "PNGs at 150-300 DPI"),
]


def draw_dashboard() -> Path:
    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.55, 3.4, 1.4], hspace=0.40, wspace=0.25)

    title_ax = fig.add_subplot(gs[0, :])
    title_ax.axis("off")
    title_ax.text(
        0.5, 0.7,
        "CE49X Final Project — Progress Dashboard",
        ha="center", va="center", fontsize=22, fontweight="bold",
    )
    n_done = sum(1 for _, status, _ in MILESTONES if status == "done")
    n_total = len(MILESTONES)
    pct = 100 * n_done / n_total
    title_ax.text(
        0.5, 0.25,
        f"Conflict Situation Monitoring for Maritime Shipping  |  "
        f"{n_done} of {n_total} milestones complete  ({pct:.0f}% done)",
        ha="center", va="center", fontsize=13, color="#333333",
    )

    milestone_ax = fig.add_subplot(gs[1, :])
    milestone_ax.set_title("Milestone timeline", fontsize=14, fontweight="bold", loc="left")
    milestone_ax.set_xlim(0, 1)
    milestone_ax.set_ylim(-0.5, len(MILESTONES) - 0.5)
    milestone_ax.invert_yaxis()
    milestone_ax.axis("off")

    for i, (label, status, when) in enumerate(MILESTONES):
        is_done = status == "done"
        color = "#2ca02c" if is_done else "#d62728"
        face = "#e8f5e9" if is_done else "#fdecea"
        marker = "OK" if is_done else "TODO"

        milestone_ax.add_patch(
            mpatches.FancyBboxPatch(
                (0.02, i - 0.35), 0.96, 0.7,
                boxstyle="round,pad=0.02",
                linewidth=1.2, edgecolor=color, facecolor=face,
            )
        )
        milestone_ax.text(0.04, i, marker, ha="left", va="center",
                          fontsize=9, fontweight="bold", color=color)
        milestone_ax.text(0.12, i, label, ha="left", va="center",
                          fontsize=10.5, color="#222222")
        milestone_ax.text(0.965, i, when, ha="right", va="center",
                          fontsize=9, color="#555555", style="italic")

    metrics_ax = fig.add_subplot(gs[2, 0])
    metrics_ax.set_title("Key numbers", fontsize=13, fontweight="bold", loc="left")
    metrics_ax.axis("off")
    half = (len(METRICS) + 1) // 2
    for i, (label, value, unit) in enumerate(METRICS[:half]):
        metrics_ax.text(0.0, 0.95 - i * 0.16,
                        f"{label}:", fontsize=10, color="#444444")
        metrics_ax.text(1.0, 0.95 - i * 0.16,
                        f"{value} {unit}".strip(),
                        fontsize=10, color="#1a1a1a", fontweight="bold", ha="right")

    metrics2_ax = fig.add_subplot(gs[2, 1])
    metrics2_ax.set_title(" ", fontsize=13, loc="left")
    metrics2_ax.axis("off")
    for i, (label, value, unit) in enumerate(METRICS[half:]):
        metrics2_ax.text(0.0, 0.95 - i * 0.16,
                         f"{label}:", fontsize=10, color="#444444")
        metrics2_ax.text(1.0, 0.95 - i * 0.16,
                         f"{value} {unit}".strip(),
                         fontsize=10, color="#1a1a1a", fontweight="bold", ha="right")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "project_progress.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    path = draw_dashboard()
    print(f"Progress dashboard saved to: {path}")
