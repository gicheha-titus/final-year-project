from __future__ import annotations

import pandas as pd
import pytest

from cbc_xai.exceptions import IngestionError
from cbc_xai.ingestion import validate_assessment_frame


def test_missing_columns_returns_error_list() -> None:
    frame = pd.DataFrame({"learner_id": ["L001"], "score": [50]})
    with pytest.raises(IngestionError) as exc_info:
        validate_assessment_frame(frame)
    
    assert "Missing required columns" in str(exc_info.value)
    assert len(exc_info.value.row_errors) == 1
    assert "grade" in exc_info.value.row_errors[0]


def test_bad_scores_in_multiple_rows_returns_all_errors() -> None:
    frame = pd.DataFrame(
        {
            "learner_id": ["L001", "L001", "L002"],
            "learner_name": ["Alice", "Alice", "Bob"],
            "grade": ["Grade 7", "Grade 7", "Grade 7"],
            "term": ["Term 1", "Term 2", "Term 1"],
            "subject": ["Mathematics", "Mathematics", "English"],
            "score": [105, "not_a_number", -10],
            "assessment_date": ["2023-01-15", "2023-05-15", "2023-01-20"],
        }
    )
    with pytest.raises(IngestionError) as exc_info:
        validate_assessment_frame(frame)
    
    errors = exc_info.value.row_errors
    assert len(errors) == 3
    assert "score 105.0 is outside" in errors[0]
    assert "score is not a number" in errors[1]
    assert "score -10.0 is outside" in errors[2]


def test_bad_subject_names_reported_per_row() -> None:
    frame = pd.DataFrame(
        {
            "learner_id": ["L001"],
            "learner_name": ["Alice"],
            "grade": ["Grade 7"],
            "term": ["Term 1"],
            "subject": ["Maths (Typo)"],
            "score": [50],
            "assessment_date": ["2023-01-15"],
        }
    )
    with pytest.raises(IngestionError) as exc_info:
        validate_assessment_frame(frame)
    
    assert "unrecognised subject 'Maths (Typo)'" in exc_info.value.row_errors[0]


def test_mixed_valid_invalid_rows_returns_all_errors() -> None:
    frame = pd.DataFrame(
        {
            "learner_id": ["L001", "L002"],
            "learner_name": ["Alice", "Bob"],
            "grade": ["Grade 7", "Grade 8"],
            "term": ["Term 1", "Term 2"],
            "subject": ["Mathematics", "Fake Subject"],
            "score": [75, 80],
            "assessment_date": ["2023-01-15", "invalid_date"],
        }
    )
    with pytest.raises(IngestionError) as exc_info:
        validate_assessment_frame(frame)
    
    errors = exc_info.value.row_errors
    assert len(errors) == 2
    assert "unrecognised subject" in errors[0]
    assert "assessment_date could not be parsed" in errors[1]
