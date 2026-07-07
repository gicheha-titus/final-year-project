from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
for candidate in (ROOT_DIR / "vendor", ROOT_DIR / "src"):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


@pytest.fixture(autouse=True)
def isolate_generated_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from cbc_xai import config, modeling, reporting, storage

    data_dir = tmp_path / "data"
    docs_dir = tmp_path / "docs"
    artifacts_dir = tmp_path / "artifacts"
    charts_dir = artifacts_dir / "charts"
    models_dir = artifacts_dir / "models"
    reports_dir = artifacts_dir / "reports"

    for path in (data_dir, docs_dir, artifacts_dir, charts_dir, models_dir, reports_dir):
        path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "DOCS_DIR", docs_dir)
    monkeypatch.setattr(config, "ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr(config, "CHARTS_DIR", charts_dir)
    monkeypatch.setattr(config, "MODELS_DIR", models_dir)
    monkeypatch.setattr(config, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(
        config, "SYNTHETIC_ASSESSMENTS_CSV", data_dir / "synthetic_cbc_assessments.csv"
    )
    monkeypatch.setattr(config, "MODEL_BUNDLE_PATH", models_dir / "selected_model_bundle.joblib")
    monkeypatch.setattr(config, "METRICS_JSON_PATH", models_dir / "model_metrics.json")
    monkeypatch.setattr(config, "DATABASE_PATH", artifacts_dir / "cbc_xai.db")
    monkeypatch.setattr(config, "APP_LOG_PATH", artifacts_dir / "cbc_xai.log")

    monkeypatch.setattr(modeling, "DATA_DIR", config.DATA_DIR)
    monkeypatch.setattr(modeling, "SYNTHETIC_ASSESSMENTS_CSV", config.SYNTHETIC_ASSESSMENTS_CSV)
    monkeypatch.setattr(modeling, "MODEL_BUNDLE_PATH", config.MODEL_BUNDLE_PATH)
    monkeypatch.setattr(modeling, "METRICS_JSON_PATH", config.METRICS_JSON_PATH)
    monkeypatch.setattr(modeling, "ensure_directories", config.ensure_directories)

    monkeypatch.setattr(reporting, "REPORTS_DIR", config.REPORTS_DIR)
    monkeypatch.setattr(reporting, "ensure_directories", config.ensure_directories)

    monkeypatch.setattr(storage, "DATABASE_PATH", config.DATABASE_PATH)
    monkeypatch.setattr(storage, "ensure_directories", config.ensure_directories)

    modeling.load_model_bundle.cache_clear()
    modeling._EXPLAINER_CACHE.clear()
