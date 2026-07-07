"""Application-wide paths, credentials, and directory bootstrapping.

Every module that needs to locate data files, model artifacts, or the
SQLite database imports its paths from here so that the project has a
single source of truth for filesystem layout.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Directory layout — all paths are derived from the project root so the
# application works correctly regardless of the current working directory.
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
VENDOR_DIR = ROOT_DIR / "vendor"
DATA_DIR = ROOT_DIR / "data"
DOCS_DIR = ROOT_DIR / "docs"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
CHARTS_DIR = ARTIFACTS_DIR / "charts"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

# ---------------------------------------------------------------------------
# Key file paths used by the pipeline, model loader, and storage layer.
# ---------------------------------------------------------------------------
SYNTHETIC_ASSESSMENTS_CSV = DATA_DIR / "synthetic_cbc_assessments.csv"
MODEL_BUNDLE_PATH = MODELS_DIR / "selected_model_bundle.joblib"
METRICS_JSON_PATH = MODELS_DIR / "model_metrics.json"
DATABASE_PATH = ARTIFACTS_DIR / "cbc_xai.db"
APP_LOG_PATH = ARTIFACTS_DIR / "cbc_xai.log"

# ---------------------------------------------------------------------------
# Default credentials seeded on first database initialisation.  These exist
# so that the application is immediately usable in an offline school
# environment without requiring an external identity provider.
# ---------------------------------------------------------------------------
DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "Admin@123"
DEFAULT_TEACHER_USERNAME = "teacher"
DEFAULT_TEACHER_PASSWORD = "Teacher@123"

# ---------------------------------------------------------------------------
# Application version — bump this when the feature schema or bundle format
# changes so that old model artifacts can be detected and retrained.
# ---------------------------------------------------------------------------
APP_VERSION = "1.1.0"

# ---------------------------------------------------------------------------
# Readiness band thresholds and labels.
#
# Maps a probability (0.0–1.0) to a qualitative band label used in both the
# UI and the PDF reports.  Centralised here so the vocabulary is consistent
# across all surfaces.  Listed in descending threshold order.
# ---------------------------------------------------------------------------
READINESS_BANDS: list[tuple[float, str]] = [
    (0.75, "Very strong"),
    (0.55, "Strong"),
    (0.40, "Developing"),
    (0.0, "Watch closely"),
]


def readiness_band(probability: float) -> str:
    """Return the qualitative readiness band label for *probability*.

    Uses the thresholds defined in ``READINESS_BANDS`` so the vocabulary
    is identical in the UI and the PDF reports.
    """
    for threshold, label in READINESS_BANDS:
        if probability >= threshold:
            return label
    return "Watch closely"


def ensure_directories() -> None:
    """Create all required output directories if they do not already exist.

    Called early in the pipeline and at application startup so that file
    writes never fail due to a missing parent directory.
    """
    for path in (
        DATA_DIR,
        DOCS_DIR,
        ARTIFACTS_DIR,
        CHARTS_DIR,
        MODELS_DIR,
        REPORTS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
