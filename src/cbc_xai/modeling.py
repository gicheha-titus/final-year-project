"""Model training, selection, and prediction with SHAP explanations.

This module implements the full ML workflow:

1. **Dataset construction** — generates synthetic assessments, builds
   the feature matrix, and assigns ground-truth track labels via the
   rule engine.
2. **Model comparison** — trains Logistic Regression, Decision Tree,
   and Random Forest classifiers; evaluates each with stratified 5-fold
   cross-validation for reliable metric estimates, then re-trains the
   winner on the full training split.
3. **Model selection** — picks the best model by mean macro-F1 across
   folds, with a preference for Random Forest when it is within 2
   percentage points of the absolute best (because tree ensembles pair
   well with SHAP ``TreeExplainer`` for faster explanations).
4. **Prediction** — loads the saved model bundle, runs inference for a
   single learner, and generates SHAP-based subject-level importance
   values that are surfaced in the UI as supporting and limiting factors.

Why Random Forest:
  Interpretability (SHAP TreeExplainer) and robustness on small datasets
  were deliberate, documented choices in the original research.  If a
  different model wins the F1 comparison by more than 2 pp, it is selected,
  but the preference exists because the dual-engine design relies on
  reliable SHAP attributions, not marginal accuracy gains.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from io import BytesIO
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import (
    APP_VERSION,
    DATA_DIR,
    METRICS_JSON_PATH,
    MODEL_BUNDLE_PATH,
    SYNTHETIC_ASSESSMENTS_CSV,
    ensure_directories,
)
from .exceptions import ModelError
from .features import (
    build_feature_matrix,
    build_subject_summary,
    feature_columns,
    feature_to_subject,
)
from .rules import (
    build_guidance_notes,
    derive_labels,
    pathway_probabilities_from_track_probabilities,
)
from .synthetic_data import generate_synthetic_assessments

log = logging.getLogger(__name__)

# Module-level cache for SHAP explainers.  Keyed by (model_name, id(model))
# so that explainers are reused across consecutive predictions and only
# rebuilt when the model bundle changes.
_EXPLAINER_CACHE: dict[tuple[str, int], tuple[Any, Any, list[str]]] = {}

# Number of folds for cross-validation during model selection.
# 5 folds is a reasonable balance between variance and computation time
# on a 240-learner synthetic dataset.
_CV_FOLDS = 5


@dataclass
class PredictionOutput:
    """Complete prediction result returned to the UI and report generator."""

    learner_id: str
    learner_name: str
    predicted_pathway: str
    predicted_track: str
    top_track_score: float
    pathway_probabilities: dict[str, float]
    track_probabilities: dict[str, float]
    guidance_notes: list[str]
    strengths: list[dict[str, float | str]]
    limiting_factors: list[dict[str, float | str]]
    feature_importance: dict[str, float]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _serialized_model_size_bytes(model: Any) -> int:
    """Measure the serialised size of a model without writing to disk.

    Used during model comparison to record footprint as a selection
    consideration (smaller models are preferable for offline deployments).
    """
    buffer = BytesIO()
    joblib.dump(model, buffer)
    return len(buffer.getvalue())


def _persist_assessments_csv(assessments: pd.DataFrame) -> str:
    """Save the synthetic assessment DataFrame to the data directory.

    Tries the canonical path first, then a timestamped fallback if the
    file is locked (e.g. open in Excel).  Raises ``PermissionError`` if
    all candidates fail.
    """
    candidate_paths = [
        SYNTHETIC_ASSESSMENTS_CSV,
        DATA_DIR / "synthetic_cbc_assessments_refreshed.csv",
        DATA_DIR / f"synthetic_cbc_assessments_{int(time.time())}.csv",
    ]
    for candidate in candidate_paths:
        try:
            assessments.to_csv(candidate, index=False)
            return str(candidate)
        except PermissionError:
            continue
    raise PermissionError("Unable to write the synthetic assessment CSV to the data directory.")


def _dataset_fingerprint(feature_frame: pd.DataFrame) -> str:
    """Return a short SHA-256 fingerprint of the feature matrix.

    Stored in the model bundle so that a report generated next term can
    be traced back to the exact dataset the model was trained on.
    """
    data_bytes = feature_frame.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(data_bytes).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Dataset and model training
# ---------------------------------------------------------------------------

def _build_dataset(learner_count: int = 240) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic data, build features, and derive labels.

    Returns the raw assessments DataFrame (for database import) and a
    merged feature-plus-label DataFrame (for model training).
    """
    ensure_directories()
    assessments = generate_synthetic_assessments(learner_count=learner_count)
    export_path = _persist_assessments_csv(assessments)
    feature_frame = build_feature_matrix(assessments)
    labels = []
    for learner_id, learner_assessments in assessments.groupby("learner_id", sort=True):
        derived = derive_labels(build_subject_summary(learner_assessments))
        labels.append(
            {
                "learner_id": learner_id,
                "track_label": derived["predicted_track"],
                "pathway_label": derived["predicted_pathway"],
                "top_track_score": derived["top_track_score"],
            }
        )
    label_frame = pd.DataFrame(labels)
    dataset = feature_frame.merge(label_frame, on="learner_id", how="inner")
    dataset.attrs["assessment_export_path"] = export_path
    return assessments, dataset


def _candidate_models(numeric_columns: list[str]) -> dict[str, Any]:
    """Return the three candidate classifiers to compare.

    Logistic Regression is wrapped in a Pipeline with StandardScaler
    because it is sensitive to feature scale; tree-based models are
    scale-invariant and used directly.
    """
    preprocessing = ColumnTransformer(
        transformers=[("numeric", StandardScaler(), numeric_columns)],
        remainder="drop",
    )
    return {
        "LogisticRegression": Pipeline(
            steps=[
                ("preprocess", preprocessing),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "DecisionTreeClassifier": DecisionTreeClassifier(max_depth=8, random_state=42),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
        ),
    }


def _cross_validate_model(
    name: str,
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict[str, Any]:
    """Run stratified k-fold CV and return averaged metrics.

    Using 5-fold stratified CV instead of a single holdout gives a more
    reliable estimate of generalisation on a small (240-learner) dataset
    where a single split can be misleadingly optimistic or pessimistic
    depending on chance.
    """
    skf = StratifiedKFold(n_splits=_CV_FOLDS, shuffle=True, random_state=42)
    fold_f1s: list[float] = []
    fold_precisions: list[float] = []
    fold_recalls: list[float] = []
    fold_accuracies: list[float] = []

    for train_idx, val_idx in skf.split(X, y):
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]

        # Clone-free: refit the same object each fold (it resets on fit).
        model.fit(X_fold_train, y_fold_train)
        preds = model.predict(X_fold_val)

        acc = accuracy_score(y_fold_val, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_fold_val, preds, average="macro", zero_division=0
        )
        fold_accuracies.append(float(acc))
        fold_precisions.append(float(prec))
        fold_recalls.append(float(rec))
        fold_f1s.append(float(f1))

    return {
        "model_name": name,
        "cv_accuracy_mean": round(float(np.mean(fold_accuracies)), 4),
        "cv_accuracy_std": round(float(np.std(fold_accuracies)), 4),
        "cv_precision_macro_mean": round(float(np.mean(fold_precisions)), 4),
        "cv_recall_macro_mean": round(float(np.mean(fold_recalls)), 4),
        "cv_f1_macro_mean": round(float(np.mean(fold_f1s)), 4),
        "cv_f1_macro_std": round(float(np.std(fold_f1s)), 4),
        "cv_folds": _CV_FOLDS,
    }


def _evaluate_model_holdout(
    name: str,
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Train a model on the holdout split and compute final metrics.

    The holdout split is used for the confusion matrix, per-class metrics,
    and timing benchmarks stored in the bundle.  Model selection itself
    uses the CV metrics.
    """
    started = time.perf_counter()
    model.fit(X_train, y_train)
    training_seconds = time.perf_counter() - started

    prediction_started = time.perf_counter()
    predictions = model.predict(X_test)
    prediction_seconds = time.perf_counter() - prediction_started

    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="macro", zero_division=0
    )

    classes = sorted(y_test.unique().tolist())

    # Per-class breakdown: teachers reviewing the research write-up should
    # be able to see which pathways the model handles well and where it
    # struggles, not just the macro average.
    report_dict = classification_report(
        y_test,
        predictions,
        labels=classes,
        output_dict=True,
        zero_division=0,
    )
    per_class_metrics = {
        cls: {
            "precision": round(report_dict[cls]["precision"], 4),
            "recall": round(report_dict[cls]["recall"], 4),
            "f1": round(report_dict[cls]["f1-score"], 4),
            "support": int(report_dict[cls]["support"]),
        }
        for cls in classes
        if cls in report_dict
    }

    # Benchmark single-sample latency by averaging 30 predictions.
    single_sample = X_test.iloc[[0]]
    single_started = time.perf_counter()
    for _ in range(30):
        model.predict(single_sample)
    single_prediction_seconds = (time.perf_counter() - single_started) / 30

    return {
        "model_name": name,
        "holdout_accuracy": round(float(accuracy), 4),
        "holdout_precision_macro": round(float(precision), 4),
        "holdout_recall_macro": round(float(recall), 4),
        "holdout_f1_macro": round(float(f1), 4),
        "per_class_metrics": per_class_metrics,
        "training_seconds": round(float(training_seconds), 4),
        "prediction_seconds": round(float(prediction_seconds), 4),
        "single_prediction_seconds": round(float(single_prediction_seconds), 6),
        "model_size_bytes": _serialized_model_size_bytes(model),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=classes).tolist(),
        "classes": classes,
        "estimator": model,
    }


def train_and_select_model(learner_count: int = 240) -> dict[str, Any]:
    """Run the full training pipeline and persist the selected model.

    Selection strategy:
    - Use 5-fold stratified CV to estimate each model's macro-F1.
    - Select the highest CV-F1 model, with a built-in preference for
      Random Forest if its CV-F1 is within 0.02 of the absolute best.
    - Re-train the selected model on the full training split (75%) and
      compute holdout metrics on the test split (25%).

    Returns a dictionary containing the model bundle, raw assessments,
    dataset, and the CSV export path.
    """
    ensure_directories()
    assessments, dataset = _build_dataset(learner_count=learner_count)
    numeric_columns = feature_columns()
    X = dataset[numeric_columns]
    y = dataset["track_label"]

    # Holdout split for final evaluation and SHAP background data.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y,
    )

    candidates = _candidate_models(numeric_columns)
    cv_results: list[dict[str, Any]] = []
    for name, candidate in candidates.items():
        log.info("Cross-validating %s …", name)
        cv_result = _cross_validate_model(name, candidate, X_train, y_train)
        cv_results.append(cv_result)
        log.info(
            "%s: CV F1=%.4f ± %.4f",
            name,
            cv_result["cv_f1_macro_mean"],
            cv_result["cv_f1_macro_std"],
        )

    # Selection: best CV-F1, with a preference for RandomForest if close.
    best_cv = max(cv_results, key=lambda r: r["cv_f1_macro_mean"])
    selected_cv = next(
        (
            r for r in cv_results
            if r["model_name"] == "RandomForestClassifier"
            and r["cv_f1_macro_mean"] >= best_cv["cv_f1_macro_mean"] - 0.02
        ),
        best_cv,
    )
    selected_name = selected_cv["model_name"]
    log.info("Selected model: %s (CV F1=%.4f)", selected_name, selected_cv["cv_f1_macro_mean"])

    # Re-train all models on the full training split for holdout metrics.
    holdout_results: list[dict[str, Any]] = []
    for name, candidate in _candidate_models(numeric_columns).items():
        holdout = _evaluate_model_holdout(name, candidate, X_train, X_test, y_train, y_test)
        # Merge CV metrics into the holdout record.
        cv_for_this = next((r for r in cv_results if r["model_name"] == name), {})
        holdout.update({k: v for k, v in cv_for_this.items() if k != "model_name"})
        holdout_results.append(holdout)

    selected_holdout = next(r for r in holdout_results if r["model_name"] == selected_name)

    bundle = {
        "selected_model_name": selected_name,
        "model": selected_holdout["estimator"],
        "feature_columns": numeric_columns,
        "training_background": X_train.head(60),
        "metrics": [
            {key: value for key, value in item.items() if key != "estimator"}
            for item in holdout_results
        ],
        # Metadata for tracing a report back to the model that generated it.
        "trained_at": datetime.now(tz=timezone.utc).isoformat(),
        "cbc_xai_version": APP_VERSION,
        "feature_schema_version": "v1",  # bump when feature_columns() changes
        "dataset_fingerprint": _dataset_fingerprint(X),
        "learner_count": learner_count,
    }
    joblib.dump(bundle, MODEL_BUNDLE_PATH)

    # Clear caches so the next call to load_model_bundle() picks up
    # the freshly trained model.
    _EXPLAINER_CACHE.clear()
    load_model_bundle.cache_clear()

    METRICS_JSON_PATH.write_text(json.dumps(bundle["metrics"], indent=2), encoding="utf-8")
    log.info("Model bundle saved to %s", MODEL_BUNDLE_PATH)
    return {
        "bundle": bundle,
        "assessments": assessments,
        "dataset": dataset,
        "assessment_export_path": dataset.attrs.get("assessment_export_path", str(SYNTHETIC_ASSESSMENTS_CSV)),
    }


@lru_cache(maxsize=1)
def load_model_bundle() -> dict[str, Any]:
    """Load the persisted model bundle, training on demand if absent.

    The result is cached so that repeated calls during a single
    application session avoid redundant disk I/O.
    """
    if not MODEL_BUNDLE_PATH.exists():
        log.info("No model bundle found; training now.")
        return train_and_select_model()["bundle"]
    bundle = joblib.load(MODEL_BUNDLE_PATH)
    log.debug(
        "Loaded model bundle: %s trained at %s",
        bundle.get("selected_model_name", "unknown"),
        bundle.get("trained_at", "unknown"),
    )
    return bundle


# ---------------------------------------------------------------------------
# SHAP explanation helpers
# ---------------------------------------------------------------------------

def _subject_shap_values(shap_row: np.ndarray, columns: list[str]) -> dict[str, float]:
    """Aggregate per-feature SHAP values into per-subject totals.

    Each subject has four features (mean, recent_mean, trend, consistency);
    summing their SHAP values gives a single "how much did this subject
    push toward/away from the predicted track" number.
    """
    subject_totals: dict[str, float] = {}
    for feature_name, value in zip(columns, shap_row, strict=True):
        subject = feature_to_subject(feature_name)
        subject_totals[subject] = subject_totals.get(subject, 0.0) + float(value)
    return subject_totals


def _prediction_shap_row(
    model_bundle: dict[str, Any],
    feature_frame: pd.DataFrame,
    predicted_track: str,
) -> np.ndarray:
    """Compute the SHAP values for one learner's prediction.

    Lazily constructs the SHAP explainer on first use and caches it for
    subsequent predictions with the same model.  Uses ``TreeExplainer``
    for tree-based models and ``LinearExplainer`` for logistic regression.
    """
    model = model_bundle["model"]
    selected_model_name = model_bundle["selected_model_name"]
    explainer_key = (selected_model_name, id(model))

    if explainer_key not in _EXPLAINER_CACHE:
        if selected_model_name == "LogisticRegression":
            preprocessing = model.named_steps["preprocess"]
            linear_model = model.named_steps["model"]
            transformed_background = preprocessing.transform(model_bundle["training_background"])
            explainer = shap.LinearExplainer(linear_model, transformed_background)
            class_labels = list(linear_model.classes_)
            _EXPLAINER_CACHE[explainer_key] = (explainer, preprocessing, class_labels)
        else:
            explainer = shap.TreeExplainer(model)
            class_labels = list(model.classes_)
            _EXPLAINER_CACHE[explainer_key] = (explainer, None, class_labels)

    explainer, preprocessing, class_labels = _EXPLAINER_CACHE[explainer_key]

    if selected_model_name == "LogisticRegression":
        transformed_instance = preprocessing.transform(feature_frame)
        shap_values = explainer.shap_values(transformed_instance)
    else:
        shap_values = explainer.shap_values(feature_frame)

    # Extract the SHAP row for the predicted class.  SHAP returns
    # different array shapes depending on the explainer type.
    if predicted_track not in class_labels:
        raise ModelError(
            f"Predicted track '{predicted_track}' not found in model classes: {class_labels}"
        )
    class_index = class_labels.index(predicted_track)
    if isinstance(shap_values, list):
        return np.asarray(shap_values[class_index][0], dtype=float)
    if getattr(shap_values, "ndim", 0) == 3:
        return np.asarray(shap_values[0, :, class_index], dtype=float)
    return np.asarray(shap_values[0], dtype=float)


# ---------------------------------------------------------------------------
# Public prediction API
# ---------------------------------------------------------------------------

def predict_for_learner(
    learner_assessments: pd.DataFrame,
    bundle: dict[str, Any] | None = None,
) -> PredictionOutput:
    """Generate a full readiness prediction for one learner.

    Combines the ML model's probabilistic output with the rule engine's
    track scores and SHAP's per-subject importance to produce a
    ``PredictionOutput`` that the UI and report generator consume.

    Strengths and limiting factors are the top-3 subjects with the
    highest positive and negative SHAP contributions respectively.
    """
    bundle = bundle or load_model_bundle()
    model = bundle["model"]
    columns = bundle["feature_columns"]

    try:
        feature_frame = build_feature_matrix(learner_assessments)
        learner_row = feature_frame.iloc[0]

        # ML probability distribution across tracks.
        probabilities = model.predict_proba(feature_frame[columns])[0]
        track_probabilities = {
            label: round(float(probability), 6)
            for label, probability in zip(model.classes_, probabilities, strict=True)
        }
        predicted_track = max(track_probabilities, key=track_probabilities.get)
        pathway_probabilities = pathway_probabilities_from_track_probabilities(track_probabilities)

        # Rule-engine scoring for the readiness score display.
        subject_summary = build_subject_summary(learner_assessments)
        rule_outputs = derive_labels(subject_summary)

        # SHAP explanation — which subjects pushed toward / away from the
        # predicted track.
        shap_row = _prediction_shap_row(bundle, feature_frame[columns], predicted_track)
        subject_importance = _subject_shap_values(shap_row, columns)

    except ModelError:
        raise
    except Exception as exc:
        raise ModelError(f"Prediction failed for learner '{learner_assessments.iloc[0].get('learner_id', '?')}': {exc}") from exc

    strengths = [
        {"subject": subject, "contribution": round(value, 4)}
        for subject, value in sorted(subject_importance.items(), key=lambda item: item[1], reverse=True)
        if value > 0
    ][:3]
    limiting_factors = [
        {"subject": subject, "contribution": round(value, 4)}
        for subject, value in sorted(subject_importance.items(), key=lambda item: item[1])
        if value < 0
    ][:3]

    predicted_pathway = max(pathway_probabilities, key=pathway_probabilities.get)
    return PredictionOutput(
        learner_id=str(learner_row["learner_id"]),
        learner_name=str(learner_row["learner_name"]),
        predicted_pathway=predicted_pathway,
        predicted_track=predicted_track,
        top_track_score=float(rule_outputs["track_scores"][predicted_track]),
        pathway_probabilities=pathway_probabilities,
        track_probabilities=track_probabilities,
        guidance_notes=build_guidance_notes(subject_summary, predicted_pathway),
        strengths=strengths,
        limiting_factors=limiting_factors,
        feature_importance={subject: round(value, 4) for subject, value in subject_importance.items()},
    )
