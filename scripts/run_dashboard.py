"""
CLI Launcher for the Interactive Web Dashboard.

Starts the local web server and automatically opens your browser to the
Predictive VRAM & Split Inference interactive control center.

Usage:
    uv run python scripts/run_dashboard.py
    python scripts/run_dashboard.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dashboard.server import start_server


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Interactive Web Dashboard.")
    parser.add_argument("--port", type=int, default=8000, help="Local port to serve on (default: 8000).")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open the web browser.")
    args = parser.parse_args()

    start_server(port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
