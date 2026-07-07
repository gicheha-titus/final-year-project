"""Pipeline entry point — generates data, trains models, and seeds the database.

Run this script once before launching the desktop application to
produce the synthetic assessment dataset, train and evaluate the
candidate ML models, select the best performer, and populate the
SQLite database with the generated learner records.

Usage::

    python run_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
for candidate in (ROOT_DIR / "vendor", ROOT_DIR / "src"):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from cbc_xai.modeling import train_and_select_model  # noqa: E402
from cbc_xai.storage import import_assessment_frame, initialize_database  # noqa: E402


def main() -> None:
    """Run the full training pipeline and seed the local database."""
    results = train_and_select_model()
    initialize_database()
    imported = import_assessment_frame(results["assessments"])
    print(f"Generated synthetic dataset and imported {imported} assessment rows.")
    print(f"Assessment CSV exported to: {results['assessment_export_path']}")
    print("Model bundle saved to artifacts/models and SQLite database initialized.")


if __name__ == "__main__":
    main()
