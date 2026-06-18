"""Fetch Wikipedia article summaries for each conflict region.

Uses the Wikipedia REST API — no key or registration required.
Saves structured summaries to data/raw/wikipedia_context.json.

Wikipedia is NOT a structured event database (no lat/lon per explosion),
so it is used for context and background, not for event-level verification.
Use ACLED or UCDP for event-level ground truth.

Usage:
    python scripts/fetch_wikipedia_context.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
HEADERS = {"User-Agent": "CE49X-conflict-monitoring/1.0 (educational project)"}
SLEEP = 0.4
MAX_EXTRACT_CHARS = 2000  # keep summaries concise

# Wikipedia article titles for each conflict region.
# These are the most relevant overview articles — useful for understanding
# the nature and timeline of each conflict.
REGION_ARTICLES: dict[str, list[str]] = {
    "Ukraine_Black_Sea": [
        "Russian invasion of Ukraine",
        "Battle of the Black Sea (2022–present)",
        "2024 in the Russo-Ukrainian War",
        "2025 in the Russo-Ukrainian War",
    ],
    "Red_Sea_Yemen": [
        "Yemeni civil war (2014–present)",
        "Houthi attacks on Israel, Red Sea area, and other countries",
        "2024–25 Red Sea crisis",
    ],
    "Gaza_Israel_Lebanon": [
        "Gaza–Israel conflict",
        "2023 Hamas-led attack on Israel",
        "Israeli invasion of the Gaza Strip",
        "2024 Lebanon conflict",
    ],
    "Persian_Gulf_Iraq_Iran": [
        "Iran–Israel conflict (2023–present)",
        "Insurgency in Iraq (2017–present)",
        "Iran–United States relations",
    ],
    "Sudan_Red_Sea_Corridor": [
        "Sudanese civil war (2023–present)",
        "Rapid Support Forces",
        "War in Darfur",
    ],
    "Libya_Mediterranean": [
        "Second Libyan Civil War",
        "Libyan crisis (2011–present)",
    ],
}


def fetch_summary(title: str) -> dict:
    url = WIKI_SUMMARY.format(title=requests.utils.quote(title, safe=""))
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code == 404:
            return {"title": title, "status": "not_found", "extract": ""}
        resp.raise_for_status()
        data = resp.json()
        return {
            "title": data.get("title", title),
            "description": data.get("description", ""),
            "extract": data.get("extract", "")[:MAX_EXTRACT_CHARS],
            "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
            "status": "ok",
        }
    except Exception as exc:
        return {"title": title, "status": "error", "error": str(exc), "extract": ""}


def main() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, list[dict]] = {}

    for region, articles in REGION_ARTICLES.items():
        print(f"\nFetching Wikipedia summaries for {region}...")
        results[region] = []
        for title in articles:
            summary = fetch_summary(title)
            status = summary.get("status", "?")
            print(f"  [{status:10s}] {title}")
            results[region].append(summary)
            time.sleep(SLEEP)

    out_path = RAW_DATA_DIR / "wikipedia_context.json"
    out_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSaved Wikipedia context -> {out_path}")

    # Print a brief digest
    print("\nDigest (first 200 chars of each article):")
    for region, articles in results.items():
        print(f"\n  {region}:")
        for art in articles:
            if art.get("status") == "ok":
                snippet = art.get("extract", "")[:200].replace("\n", " ")
                print(f"    · {art['title']}: {snippet}...")


if __name__ == "__main__":
    main()
