"""Interactive 3D globe dashboard for FIRMS thermal detections.

Output: figures/globe_dashboard.html

The HTML file can be opened in any browser. You can:
  - Click & drag to rotate the globe.
  - Scroll to zoom in/out.
  - Click region names in the legend to toggle them on/off.
  - Use the buttons at the top to switch between Conflict, Non-conflict and All.
  - Hover any point to see region, date, FRP and brightness.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
FIGURES_DIR = PROJECT_ROOT / "figures"

CONFLICT_SAMPLE = PROCESSED_DIR / "firms_model_sample.csv"
NONCONFLICT_FILES = [
    RAW_DIR / "firms_thermal_nonconflict_2024_raw.csv",
    RAW_DIR / "firms_thermal_nonconflict_2025_2026_raw.csv",
]

MAX_POINTS_PER_REGION = 12_000

CONFLICT_PALETTE = {
    "Ukraine_Black_Sea":      "#d62728",   # vivid red
    "Red_Sea_Yemen":          "#e6550d",   # orange-red
    "Gaza_Israel_Lebanon":    "#fc8d59",   # salmon
    "Persian_Gulf_Iraq_Iran": "#b2182b",   # dark crimson
    "Sudan_Red_Sea_Corridor": "#ef6548",   # coral
    "Libya_Mediterranean":    "#f46d43",   # warm orange
}
NONCONFLICT_PALETTE = {
    "Australia_Bushfires":   "#2171b5",   # medium blue
    "California_Wildfires":  "#6baed6",   # sky blue
    "Amazon_Brazil":         "#08519c",   # deep navy
    "Indonesia_Peat":        "#4292c6",   # cornflower
    "Canada_Boreal":         "#9ecae1",   # pale blue
}


def load_conflict_sample() -> pd.DataFrame:
    if not CONFLICT_SAMPLE.exists():
        return pd.DataFrame()
    frame = pd.read_csv(CONFLICT_SAMPLE)
    frame["category"] = "Conflict"
    return frame


def load_nonconflict_sample() -> pd.DataFrame:
    frames = []
    for path in NONCONFLICT_FILES:
        if path.exists():
            frame = pd.read_csv(path)
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined.columns = [column.strip().lower() for column in combined.columns]
    combined["category"] = "Non-conflict"
    return combined


def downsample(frame: pd.DataFrame, max_per_region: int, seed: int = 49) -> pd.DataFrame:
    if frame.empty:
        return frame
    parts = []
    for region, group in frame.groupby("region", sort=False):
        if len(group) > max_per_region:
            parts.append(group.sample(n=max_per_region, random_state=seed))
        else:
            parts.append(group)
    return pd.concat(parts, ignore_index=True)


def make_globe(conflict: pd.DataFrame, nonconflict: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    trace_meta: list[dict] = []

    for region, color in CONFLICT_PALETTE.items():
        sub = conflict[conflict["region"] == region]
        if sub.empty:
            continue
        hover = (
            "<b>" + sub["region"].astype(str).str.replace("_", " ") + "</b><br>"
            + "Date: " + sub["acq_date"].astype(str) + "<br>"
            + "Lat / Lon: " + sub["latitude"].round(2).astype(str)
            + " / " + sub["longitude"].round(2).astype(str) + "<br>"
            + "FRP: " + sub["frp"].round(1).astype(str) + " MW<br>"
            + "Brightness: " + sub["bright_ti4"].round(1).astype(str) + " K<br>"
            + "Day/Night: " + sub["daynight"].astype(str)
        )
        fig.add_trace(
            go.Scattergeo(
                lon=sub["longitude"],
                lat=sub["latitude"],
                name=f"{region.replace('_', ' ')}  (conflict)",
                mode="markers",
                marker=dict(size=3, opacity=0.65, color=color, line=dict(width=0)),
                hovertext=hover,
                hoverinfo="text",
                visible=True,
            )
        )
        trace_meta.append({"category": "Conflict", "region": region})

    for region, color in NONCONFLICT_PALETTE.items():
        if nonconflict.empty or "region" not in nonconflict.columns:
            continue
        sub = nonconflict[nonconflict["region"] == region]
        if sub.empty:
            continue
        if "bright_ti4" not in sub.columns:
            sub = sub.assign(bright_ti4=np.nan)
        if "frp" not in sub.columns:
            sub = sub.assign(frp=np.nan)
        if "daynight" not in sub.columns:
            sub = sub.assign(daynight="?")
        hover = (
            "<b>" + sub["region"].astype(str).str.replace("_", " ") + "</b><br>"
            + "Date: " + sub["acq_date"].astype(str) + "<br>"
            + "Lat / Lon: " + sub["latitude"].round(2).astype(str)
            + " / " + sub["longitude"].round(2).astype(str) + "<br>"
            + "FRP: " + sub["frp"].round(1).astype(str) + " MW<br>"
            + "Brightness: " + sub["bright_ti4"].round(1).astype(str) + " K<br>"
            + "Day/Night: " + sub["daynight"].astype(str)
        )
        fig.add_trace(
            go.Scattergeo(
                lon=sub["longitude"],
                lat=sub["latitude"],
                name=f"{region.replace('_', ' ')}  (non-conflict)",
                mode="markers",
                marker=dict(size=3, opacity=0.65, color=color, symbol="diamond",
                            line=dict(width=0)),
                hovertext=hover,
                hoverinfo="text",
                visible=True,
            )
        )
        trace_meta.append({"category": "Non-conflict", "region": region})

    show_all = [True] * len(trace_meta)
    show_conflict = [meta["category"] == "Conflict" for meta in trace_meta]
    show_nonconflict = [meta["category"] == "Non-conflict" for meta in trace_meta]

    fig.update_geos(
        projection_type="orthographic",
        projection_rotation=dict(lon=30, lat=20, roll=0),
        showcountries=True, countrycolor="rgba(80,80,80,0.6)",
        showcoastlines=True, coastlinecolor="rgba(40,40,40,0.7)",
        showland=True, landcolor="rgb(232, 232, 220)",
        showocean=True, oceancolor="rgb(180, 205, 230)",
        showlakes=True, lakecolor="rgb(180, 205, 230)",
        showframe=True, framecolor="rgba(0,0,0,0.4)",
        bgcolor="rgb(240, 240, 245)",
    )

    total_conflict    = len(conflict)
    total_nonconflict = len(nonconflict)
    total_shown       = total_conflict + total_nonconflict

    fig.update_layout(
        title=dict(
            text=(
                "<b>Conflict Situation Monitoring — Interactive 3D Globe</b><br>"
                "<span style='font-size:13px;color:#444;'>"
                "NASA FIRMS thermal anomalies · "
                f"<b>Showing {total_shown:,} representative points</b> sampled from "
                "<b>5,988,583 total detections</b> (up to 12,000 per region) · "
                "Conflict regions = circles, Non-conflict fire regions = diamonds · "
                "Drag to rotate · Scroll to zoom · Click legend to filter."
                "</span>"
            ),
            x=0.5, xanchor="center",
        ),
        height=820,
        margin=dict(l=10, r=10, t=120, b=10),
        legend=dict(
            title="Region (click to toggle)",
            x=1.02, y=0.5, bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#cccccc", borderwidth=1,
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.5, xanchor="center",
                y=1.10, yanchor="bottom",
                showactive=True,
                buttons=[
                    dict(label="All regions",
                         method="update", args=[{"visible": show_all}]),
                    dict(label="Conflict only",
                         method="update", args=[{"visible": show_conflict}]),
                    dict(label="Non-conflict only",
                         method="update", args=[{"visible": show_nonconflict}]),
                ],
            )
        ],
    )

    return fig


def main() -> None:
    conflict = load_conflict_sample()
    nonconflict = load_nonconflict_sample()

    conflict = downsample(conflict, MAX_POINTS_PER_REGION)
    nonconflict = downsample(nonconflict, MAX_POINTS_PER_REGION)

    print(f"Conflict points plotted:     {len(conflict):,}")
    print(f"Non-conflict points plotted: {len(nonconflict):,}")
    if nonconflict.empty:
        print("(Non-conflict data not yet collected — globe will still render the conflict regions.)")

    fig = make_globe(conflict, nonconflict)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output = FIGURES_DIR / "globe_dashboard.html"
    fig.write_html(
        output,
        include_plotlyjs="cdn",
        full_html=True,
        config={"scrollZoom": True, "displaylogo": False},
    )
    print(f"Saved interactive globe dashboard to: {output}")


if __name__ == "__main__":
    main()
