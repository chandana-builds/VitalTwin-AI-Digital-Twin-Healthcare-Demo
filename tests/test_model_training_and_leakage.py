"""
Unit tests for Machine Learning Model Training, Prediction, and Evaluation
VitalTwin AI - Happiest Health Digital Twin Challenge 2026
"""

import pytest
import numpy as np
import pandas as pd

from config.settings import FEATURE_COLUMNS, DEFAULT_HIGH_GLUCOSE_THRESHOLD
from src.data.patient_generator import get_patient_profile
from src.data.sensor_generator import generate_patient_timeseries
from src.features.feature_engineering import prepare_dataset_for_modeling, extract_features_for_single_point
from src.models.predictor import GlucoseRiskPredictor, train_and_evaluate_model, ModelEvaluationResults


@pytest.fixture(scope="module")
def trained_predictor_and_eval():
    """Trains a test model instance."""
    predictor, eval_results = train_and_evaluate_model(threshold=180.0, random_seed=42)
    return predictor, eval_results


def test_model_training_and_evaluation_structure(trained_predictor_and_eval):
    """Verify that model training completes and yields authentic evaluation results."""
    predictor, eval_results = trained_predictor_and_eval

    assert predictor.is_trained
    assert isinstance(eval_results, ModelEvaluationResults)

    # Genuine metric bounds
    assert 0.0 <= eval_results.accuracy <= 1.0
    assert 0.0 <= eval_results.precision <= 1.0
    assert 0.0 <= eval_results.recall <= 1.0
    assert 0.0 <= eval_results.f1 <= 1.0
    assert eval_results.n_train > 100
    assert eval_results.n_test > 50

    # Verify confusion matrix
    cm = eval_results.confusion_matrix
    assert len(cm) == 2 and len(cm[0]) == 2
    total_cm_samples = cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1]
    assert total_cm_samples == eval_results.n_test


def test_feature_importance_validity(trained_predictor_and_eval):
    """Verify that feature importances are calculated and sum to 1.0."""
    _, eval_results = trained_predictor_and_eval
    fi = eval_results.feature_importances

    assert len(fi) == len(FEATURE_COLUMNS)
    total_importance = sum(fi.values())
    assert np.isclose(total_importance, 1.0, atol=1e-3)
    # Ensure importances sum to 1.0 and both baseline and dynamic features are utilized
    all_features = list(fi.keys())
    assert any("glucose" in f or "carbs" in f for f in all_features[:10])
    assert any("hba1c" in f or "age" in f or "bmi" in f for f in all_features[:10])


def test_prediction_output_validation(trained_predictor_and_eval):
    """Verify output types and ranges for predict_risk."""
    predictor, _ = trained_predictor_and_eval
    profile = get_patient_profile("PT-1001")
    df = generate_patient_timeseries("PT-1001", days=1, random_seed=42)
    feat_row = extract_features_for_single_point(df, profile=profile)

    proba, tier, factors = predictor.predict_risk(feat_row)

    assert isinstance(proba, float)
    assert 0.0 <= proba <= 1.0
    assert tier in ["Low Risk", "Moderate Risk", "High Risk"]
    assert isinstance(factors, list)
    assert len(factors) > 0
    for factor in factors:
        assert "feature" in factor
        assert "value" in factor
        assert "importance" in factor
        assert "description" in factor


def test_uncalibrated_exception_on_untrained_model():
    """Verify that an untrained model raises RuntimeError rather than returning fake numbers."""
    untrained = GlucoseRiskPredictor()
    dummy_df = pd.DataFrame([{col: 0.0 for col in FEATURE_COLUMNS}])
    with pytest.raises(RuntimeError):
        untrained.predict_risk(dummy_df)
