from __future__ import annotations

import numpy as np
import pandas as pd

from cbc_xai.features import build_feature_matrix


def test_sparse_history_no_nans() -> None:
    # Learner with only 1 term of data
    assessments = pd.DataFrame(
        {
            "learner_id": ["L001", "L001"],
            "learner_name": ["Alice", "Alice"],
            "grade": ["Grade 7", "Grade 7"],
            "term": ["Term 1", "Term 1"],
            "subject": ["Mathematics", "English"],
            "score": [75, 80],
            "assessment_date": ["2023-01-15", "2023-01-15"],
        }
    )
    matrix = build_feature_matrix(assessments)
    
    assert not matrix.isnull().values.any()
    assert matrix.iloc[0]["mathematics_consistency"] == 0.0
    assert matrix.iloc[0]["mathematics_trend"] == 0.0
    assert matrix.iloc[0]["mathematics_recent_mean"] == 75.0


def test_missing_subject_no_nans() -> None:
    # Learner with no Science scores at all
    assessments = pd.DataFrame(
        {
            "learner_id": ["L001"],
            "learner_name": ["Alice"],
            "grade": ["Grade 7"],
            "term": ["Term 1"],
            "subject": ["Mathematics"],
            "score": [75],
            "assessment_date": ["2023-01-15"],
        }
    )
    matrix = build_feature_matrix(assessments)
    
    assert not matrix.isnull().values.any()
    assert matrix.iloc[0]["integrated_science_mean"] == 0.0
    assert matrix.iloc[0]["integrated_science_consistency"] == 100.0


def test_flat_scores_no_nans() -> None:
    # Learner with exactly the same score (std = 0)
    assessments = pd.DataFrame(
        {
            "learner_id": ["L001"] * 3,
            "learner_name": ["Alice"] * 3,
            "grade": ["Grade 7"] * 3,
            "term": ["Term 1", "Term 2", "Term 3"],
            "subject": ["Mathematics"] * 3,
            "score": [80, 80, 80],
            "assessment_date": ["2023-01-15", "2023-05-15", "2023-09-15"],
        }
    )
    matrix = build_feature_matrix(assessments)
    
    assert not matrix.isnull().values.any()
    assert matrix.iloc[0]["mathematics_consistency"] == 0.0
    assert matrix.iloc[0]["mathematics_trend"] == 0.0
