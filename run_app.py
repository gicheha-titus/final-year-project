"""Desktop application entry point.

Launches the PySide6 pathway guidance workspace.  The application
auto-seeds the database and model artifacts on first run so that
teachers can begin reviewing learner data immediately.

Usage::

    python run_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
for candidate in (ROOT_DIR / "vendor", ROOT_DIR / "src"):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from cbc_xai.app import main  # noqa: E402


if __name__ == "__main__":
    main()
