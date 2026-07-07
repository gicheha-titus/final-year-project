from __future__ import annotations

from cbc_xai import modeling
from cbc_xai.features import build_feature_matrix, feature_columns
from cbc_xai.modeling import predict_for_learner, train_and_select_model
from cbc_xai.reporting import generate_pdf_report
from cbc_xai.synthetic_data import generate_synthetic_assessments


def test_synthetic_dataset_shape() -> None:
    assessments = generate_synthetic_assessments(learner_count=10, random_seed=7)
    assert len(assessments) == 10 * 9 * 12


def test_feature_matrix_contains_expected_columns() -> None:
    assessments = generate_synthetic_assessments(learner_count=4, random_seed=11)
    matrix = build_feature_matrix(assessments)
    for column in feature_columns():
        assert column in matrix.columns


def test_training_pipeline_generates_bundle() -> None:
    results = train_and_select_model(learner_count=120)
    assert results["bundle"]["selected_model_name"] in {
        "LogisticRegression",
        "DecisionTreeClassifier",
        "RandomForestClassifier",
    }
    assert "dataset_fingerprint" in results["bundle"]
    assert "cbc_xai_version" in results["bundle"]
    assert modeling.MODEL_BUNDLE_PATH.exists()
    assert modeling.METRICS_JSON_PATH.exists()


def test_prediction_and_report_generation() -> None:
    assessments = generate_synthetic_assessments(learner_count=8, random_seed=19)
    learner_assessments = assessments[assessments["learner_id"] == "L001"]
    prediction = predict_for_learner(learner_assessments)
    assert prediction.predicted_pathway
    assert prediction.predicted_track
    assert abs(sum(prediction.pathway_probabilities.values()) - 1.0) < 0.02
    report_path = generate_pdf_report(prediction)
    assert report_path.exists()
