"""CSV ingestion and validation for learner assessment records.

This module enforces a strict schema contract on incoming CSV files so
that downstream feature engineering and model prediction receive clean,
normalised data.

Validation collects **all** row-level errors rather than stopping on the
first failure.  This was an explicit requirement from teacher interviews
in the research: staff want to fix all problems in a file before
re-importing, not cycle through one error at a time.

The public interface has two entry points:
  - ``validate_assessment_frame`` — returns a cleaned DataFrame or raises
    ``IngestionError`` with the full list of problems.
  - ``load_assessment_csv`` — thin wrapper around the above for file paths.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .domain import CSV_COLUMNS, GRADE_TERM_SEQUENCE, SUBJECT_ALIASES, SUBJECTS
from .exceptions import IngestionError

# Pre-compute the allowed grade and term values once at import time so that
# every call to ``validate_assessment_frame`` uses a fast set-membership test.
VALID_GRADES = {grade for grade, _ in GRADE_TERM_SEQUENCE}
VALID_TERMS = {term for _, term in GRADE_TERM_SEQUENCE}


def validate_assessment_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalise a raw assessment DataFrame.

    Raises ``IngestionError`` with a descriptive message **and** a list of
    all row-level problems found, not just the first one.  This means a
    teacher who imports a file with 20 bad rows sees all 20 at once and
    can fix the whole file before trying again.

    Returns a cleaned copy sorted by learner → grade → term → subject so
    that downstream feature builders can assume chronological ordering.
    """
    # --- Structural check first (column presence is file-level, not row-level) ---
    missing_columns = [col for col in CSV_COLUMNS if col not in frame.columns]
    if missing_columns:
        raise IngestionError(
            f"Missing required columns: {', '.join(missing_columns)}",
            row_errors=[f"Column(s) absent from file: {', '.join(missing_columns)}"],
        )

    cleaned = frame[CSV_COLUMNS].copy()
    cleaned["learner_id"] = cleaned["learner_id"].astype(str).str.strip()
    cleaned["learner_name"] = cleaned["learner_name"].astype(str).str.strip()
    cleaned["grade"] = cleaned["grade"].astype(str).str.strip()
    cleaned["term"] = cleaned["term"].astype(str).str.strip()
    cleaned["subject"] = cleaned["subject"].astype(str).str.strip()

    # Normalise known subject aliases (e.g. "Kiswahili" → "Kiswahili or
    # Kenyan Sign Language") so the feature matrix has consistent columns.
    cleaned["subject"] = cleaned["subject"].map(lambda value: SUBJECT_ALIASES.get(value, value))

    # Attempt score and date coercion — failures become NaN / NaT so
    # we can continue collecting errors rather than raising immediately.
    cleaned["score"] = pd.to_numeric(cleaned["score"], errors="coerce")
    cleaned["assessment_date"] = pd.to_datetime(cleaned["assessment_date"], errors="coerce").dt.date

    # --- Row-level validation (collect all problems) ---
    errors: list[str] = []
    for idx, row in cleaned.iterrows():
        row_num = int(idx) + 2  # 1-indexed, +1 for header row
        if not str(row["learner_id"]):
            errors.append(f"Row {row_num}: learner_id is blank.")
        if str(row["grade"]) not in VALID_GRADES:
            errors.append(f"Row {row_num}: unrecognised grade '{row['grade']}'.")
        if str(row["term"]) not in VALID_TERMS:
            errors.append(f"Row {row_num}: unrecognised term '{row['term']}'.")
        if str(row["subject"]) not in SUBJECTS:
            errors.append(f"Row {row_num}: unrecognised subject '{row['subject']}'.")
        if pd.isna(row["score"]):
            errors.append(f"Row {row_num}: score is not a number.")
        elif not (0 <= float(row["score"]) <= 100):
            errors.append(f"Row {row_num}: score {row['score']} is outside [0, 100].")
        if pd.isna(row["assessment_date"]):
            errors.append(f"Row {row_num}: assessment_date could not be parsed.")

    if errors:
        # Surface all problems at once so the user can fix the whole file.
        summary = f"Validation found {len(errors)} problem(s) in the file."
        raise IngestionError(summary, row_errors=errors)

    # Drop rows where coercion failed (shouldn't remain after the error check,
    # but guard against edge cases in the coercion step).
    cleaned = cleaned.dropna(subset=["score", "assessment_date"])

    return cleaned.sort_values(
        ["learner_id", "grade", "term", "subject", "assessment_date"]
    ).reset_index(drop=True)


def load_assessment_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV file from disk and return a validated assessment DataFrame.

    Raises ``IngestionError`` if the file does not exist, cannot be read
    as CSV, or fails schema validation.
    """
    path = Path(path)
    if not path.exists():
        raise IngestionError(f"File not found: {path}", row_errors=[f"No file at path: {path}"])
    try:
        raw = pd.read_csv(path)
    except Exception as exc:
        raise IngestionError(f"Could not read CSV: {exc}", row_errors=[str(exc)]) from exc
    return validate_assessment_frame(raw)
