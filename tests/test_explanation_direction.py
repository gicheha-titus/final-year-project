from __future__ import annotations

import pandas as pd
import pytest

from cbc_xai.features import build_feature_matrix
from cbc_xai.modeling import predict_for_learner, train_and_select_model


@pytest.fixture(scope="module")
def trained_model_bundle() -> dict:
    """Train a model once for this test module to save time."""
    results = train_and_select_model(learner_count=60)
    return results["bundle"]


def test_stem_learner_shap_directions(trained_model_bundle: dict) -> None:
    # Create a learner who is exceptionally strong in STEM and weak in others
    subjects = [
        "English",
        "Kiswahili or Kenyan Sign Language",
        "Mathematics",
        "Integrated Science",
        "Health Education",
        "Social Studies",
        "Religious Education",
        "Pre-Technical and Pre-Career Education",
        "Agriculture",
        "Business Studies",
        "Life Skills Education",
        "Sports and Physical Education",
    ]
    
    rows = []
    for term in ["Term 1", "Term 2", "Term 3"]:
        for subject in subjects:
            if subject in ["Mathematics", "Integrated Science", "Pre-Technical and Pre-Career Education"]:
                score = 95
            else:
                score = 40
            rows.append({
                "learner_id": "STEM_01",
                "learner_name": "Test STEM",
                "grade": "Grade 7",
                "term": term,
                "subject": subject,
                "score": score,
                "assessment_date": "2023-01-01"
            })
            
    assessments = pd.DataFrame(rows)
    
    prediction = predict_for_learner(assessments, bundle=trained_model_bundle)
    
    # We expect a STEM pathway prediction
    assert prediction.predicted_pathway == "STEM"
    
    # SHAP explanations should highlight Math and Science as strengths
    strength_subjects = [s["subject"] for s in prediction.strengths]
    assert "Mathematics" in strength_subjects or "Integrated Science" in strength_subjects
    
    # Their feature importance should be positive
    assert prediction.feature_importance.get("Mathematics", 0) > 0 or prediction.feature_importance.get("Integrated Science", 0) > 0
    
    # A low-scoring subject should be a limiting factor (negative contribution)
    limiting_subjects = [s["subject"] for s in prediction.limiting_factors]
    assert len(limiting_subjects) > 0
    for subject in limiting_subjects:
        assert prediction.feature_importance.get(subject, 0) < 0
