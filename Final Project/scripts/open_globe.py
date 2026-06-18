"""Open the interactive globe dashboard in your default web browser.

Usage:
    python scripts/open_globe.py
"""

import webbrowser
from pathlib import Path

GLOBE_PATH = Path(__file__).resolve().parents[1] / "figures" / "globe_dashboard.html"

if not GLOBE_PATH.exists():
    raise SystemExit(
        f"Globe dashboard not found at {GLOBE_PATH}. "
        "Run `python scripts/build_globe_dashboard.py` first."
    )

uri = GLOBE_PATH.resolve().as_uri()
print(f"Opening {uri}")
webbrowser.open(uri)
