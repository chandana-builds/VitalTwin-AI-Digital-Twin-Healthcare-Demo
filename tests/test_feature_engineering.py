"""
Unit tests for Temporal Feature Engineering and Target Labeling
VitalTwin AI - Happiest Health Digital Twin Challenge 2026
"""

from datetime import datetime, timedelta
import pytest
import numpy as np
import pandas as pd

from config.settings import (
    DEFAULT_HIGH_GLUCOSE_THRESHOLD,
    FEATURE_COLUMNS,
    HORIZON_STEPS,
)
from src.data.patient_generator import get_patient_profile
from src.data.sensor_generator import generate_patient_timeseries
from src.features.feature_engineering import (
    compute_temporal_features,
    construct_prediction_target,
    prepare_dataset_for_modeling,
    extract_features_for_single_point,
)


@pytest.fixture
def sample_timeseries():
    """Generates a 1-day deterministic synthetic dataset."""
    return generate_patient_timeseries(patient_id="PT-1001", days=1, random_seed=42)


def test_feature_columns_presence(sample_timeseries):
    """Verify that all configured feature columns are computed."""
    profile = get_patient_profile("PT-1001")
    feat_df = compute_temporal_features(sample_timeseries, profile=profile)

    for col in FEATURE_COLUMNS:
        assert col in feat_df.columns, f"Missing feature column: {col}"
        # Ensure no NaN in engineered features
        assert not feat_df[col].isna().any(), f"Feature {col} contains unexpected NaN values"


def test_causal_rolling_zero_future_leakage(sample_timeseries):
    """
    CRITICAL TEST: Verify zero data leakage in feature engineering.
    Modifying future row t + k MUST NOT alter feature values computed at row t.
    """
    profile = get_patient_profile("PT-1001")
    df1 = sample_timeseries.copy()
    feat_df1 = compute_temporal_features(df1, profile=profile)

    # Pick test index
    test_idx = 50
    t_feat_original = feat_df1.iloc[test_idx][FEATURE_COLUMNS].to_dict()

    # Modify future rows (e.g. inject extreme glucose spikes in row 55 to 100)
    df2 = df1.copy()
    df2.loc[55:100, "glucose"] = 400.0
    df2.loc[55:100, "meal_carbs"] = 120.0
    df2.loc[55:100, "steps"] = 1000
    feat_df2 = compute_temporal_features(df2, profile=profile)

    t_feat_after_future_mutation = feat_df2.iloc[test_idx][FEATURE_COLUMNS].to_dict()

    for col in FEATURE_COLUMNS:
        assert np.isclose(
            t_feat_original[col],
            t_feat_after_future_mutation[col],
            rtol=1e-5,
            atol=1e-5,
        ), f"Data leakage detected! Feature '{col}' at index {test_idx} was affected by future values."


def test_target_label_construction(sample_timeseries):
    """Verify binary target label construction and edge truncation."""
    threshold = 180.0
    labeled_df = construct_prediction_target(sample_timeseries, threshold=threshold, horizon_steps=HORIZON_STEPS)

    assert "target_high_glucose_120m" in labeled_df.columns
    # Check that the last HORIZON_STEPS rows have NaN targets (unobserved future)
    last_rows_target = labeled_df["target_high_glucose_120m"].iloc[-HORIZON_STEPS:]
    assert last_rows_target.isna().all(), "Trailing rows with incomplete future window must have NaN target"

    # Check that earlier rows are 0 or 1
    valid_targets = labeled_df["target_high_glucose_120m"].iloc[:-HORIZON_STEPS]
    assert valid_targets.isin([0, 1]).all()


def test_extract_features_for_single_point(sample_timeseries):
    """Verify live single-point feature extraction format and validity."""
    profile = get_patient_profile("PT-1001")
    feat_row = extract_features_for_single_point(sample_timeseries, profile=profile)

    assert isinstance(feat_row, pd.DataFrame)
    assert len(feat_row) == 1
    assert list(feat_row.columns) == FEATURE_COLUMNS
    assert not feat_row.isna().any().any()
