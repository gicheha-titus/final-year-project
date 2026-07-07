"""Feature engineering for learner assessment data.

Transforms raw per-term, per-subject assessment rows into a fixed-width
numeric feature matrix suitable for scikit-learn classifiers.  For each
of the 12 CBC subjects the builder computes four statistics:

    * **mean** — long-run average across all available terms
    * **recent_mean** — average of the last three terms (recency signal)
    * **trend** — linear slope over time (improving vs declining)
    * **consistency** — standard deviation (score stability)

Plus four global features: ``overall_mean``, ``overall_recent_mean``,
``overall_trend``, and ``assessment_count``.  This yields a
4 + (12 × 4) = 52-column numeric feature vector per learner.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .domain import FEATURE_SUFFIXES, SUBJECT_TO_SLUG, SUBJECTS, TERM_INDEX


def ordered_assessments(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort assessment rows into chronological grade-term order.

    Attaches a ``term_order`` column (0–8) derived from the canonical
    ``GRADE_TERM_SEQUENCE`` so that trend calculations operate on the
    correct temporal ordering rather than alphabetical sorting.
    """
    ordered = frame.copy()
    ordered["term_order"] = ordered.apply(
        lambda row: TERM_INDEX[(row["grade"], row["term"])],
        axis=1,
    )
    return ordered.sort_values(["learner_id", "subject", "term_order"]).reset_index(drop=True)


def _subject_stats(subject_scores: list[float]) -> dict[str, float]:
    """Compute the four statistical features for a single subject.

    Returns zero-safe defaults when no scores are available so the
    feature matrix never contains NaN values.
    """
    values = np.asarray(subject_scores, dtype=float)
    if values.size == 0:
        return {
            "mean": 0.0,
            "recent_mean": 0.0,
            "trend": 0.0,
            "consistency": 100.0,
        }

    # Recent window: last 3 terms gives a recency-weighted signal.
    recent_window = values[-3:] if values.size >= 3 else values

    # Trend: slope of a first-degree polynomial fit over time.
    # A positive trend indicates improving performance.
    trend = float(np.polyfit(range(values.size), values, 1)[0]) if values.size > 1 else 0.0

    return {
        "mean": float(values.mean()),
        "recent_mean": float(recent_window.mean()),
        "trend": trend,
        "consistency": float(values.std(ddof=0)),
    }


def build_subject_summary(learner_frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Build a per-subject statistics dictionary for a single learner.

    Used both for feature matrix construction and for the subject
    readiness table displayed in the application UI.
    """
    summary: dict[str, dict[str, float]] = {}
    ordered = ordered_assessments(learner_frame)
    for subject in SUBJECTS:
        subject_scores = ordered.loc[ordered["subject"] == subject, "score"].astype(float).tolist()
        summary[subject] = _subject_stats(subject_scores)
    return summary


def build_feature_row(learner_frame: pd.DataFrame) -> dict[str, float | str]:
    """Convert a single learner's assessments into a flat feature dictionary.

    The returned dictionary contains identifiers (``learner_id``,
    ``learner_name``), global statistics, and per-subject statistics
    keyed by ``{slug}_{suffix}`` names that match the model's expected
    feature columns.
    """
    ordered = ordered_assessments(learner_frame)
    summary = build_subject_summary(ordered)
    row: dict[str, float | str] = {
        "learner_id": str(ordered.iloc[0]["learner_id"]),
        "learner_name": str(ordered.iloc[0]["learner_name"]),
    }
    all_scores = ordered["score"].astype(float).to_numpy()
    overall_trend = (
        float(np.polyfit(range(all_scores.size), all_scores, 1)[0]) if all_scores.size > 1 else 0.0
    )
    row["overall_mean"] = float(all_scores.mean()) if all_scores.size else 0.0
    row["overall_recent_mean"] = float(all_scores[-8:].mean()) if all_scores.size else 0.0
    row["overall_trend"] = overall_trend
    row["assessment_count"] = int(all_scores.size)

    for subject, stats in summary.items():
        slug = SUBJECT_TO_SLUG[subject]
        for suffix in FEATURE_SUFFIXES:
            row[f"{slug}_{suffix}"] = round(float(stats[suffix]), 4)
    return row


def build_feature_matrix(assessments: pd.DataFrame) -> pd.DataFrame:
    """Transform a multi-learner assessment DataFrame into a feature matrix.

    Groups by ``learner_id``, builds one feature row per learner, and
    returns a DataFrame whose columns match ``feature_columns()``.
    """
    rows = [
        build_feature_row(group)
        for _, group in ordered_assessments(assessments).groupby("learner_id", sort=True)
    ]
    return pd.DataFrame(rows)


def feature_columns() -> list[str]:
    """Return the ordered list of numeric feature column names.

    This defines the exact column order that the trained model expects
    at prediction time.  The order must remain stable between training
    and inference.
    """
    columns = ["overall_mean", "overall_recent_mean", "overall_trend", "assessment_count"]
    for subject in SUBJECTS:
        slug = SUBJECT_TO_SLUG[subject]
        for suffix in FEATURE_SUFFIXES:
            columns.append(f"{slug}_{suffix}")
    return columns


def feature_to_subject(feature_name: str) -> str:
    """Map a feature column name back to its human-readable subject.

    Used by the SHAP explanation layer to aggregate per-feature
    importance values into per-subject contributions that are
    meaningful to teachers and counsellors.
    """
    for suffix in FEATURE_SUFFIXES:
        token = f"_{suffix}"
        if feature_name.endswith(token):
            slug = feature_name[: -len(token)]
            for subject, subject_slug in SUBJECT_TO_SLUG.items():
                if subject_slug == slug:
                    return subject
    return "Overall Performance"
