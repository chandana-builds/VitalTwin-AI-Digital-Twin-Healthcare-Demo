"""
Temporal Feature Engineering and Causal Target Label Construction
VitalTwin AI - Happiest Health Digital Twin Challenge 2026

Strictly enforces causality:
- Temporal features at index t are computed using ONLY history at timestamps <= t.
- Target label represents future glucose crossing the high-glucose threshold in (t, t + 120min].
- Future observations are masked/excluded at the end of the series where the 120-min window is incomplete.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from config.settings import (
    DEFAULT_HIGH_GLUCOSE_THRESHOLD,
    PREDICTION_HORIZON_MINUTES,
    SAMPLING_INTERVAL_MINUTES,
    HORIZON_STEPS,
    FEATURE_COLUMNS,
)
from src.data.patient_generator import PatientProfile, get_patient_profile


def compute_temporal_features(
    df: pd.DataFrame,
    profile: Optional[PatientProfile] = None,
) -> pd.DataFrame:
    """
    Computes causal rolling and trend features over a 5-minute sampled time series.
    All rolling windows use `closed='both'` or shift-equivalent past index so that
    observation t NEVER sees t+1.
    """
    df = df.copy()
    if not np.issubdtype(df["timestamp"].dtype, np.datetime64):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values("timestamp").reset_index(drop=True)

    # 1. Glucose current and lag rolling averages
    # 15m = 3 steps, 30m = 6 steps, 60m = 12 steps (at 5-min intervals)
    df["glucose_current"] = df["glucose"]
    df["glucose_mean_15m"] = df["glucose"].rolling(window=3, min_periods=1).mean()
    df["glucose_mean_30m"] = df["glucose"].rolling(window=6, min_periods=1).mean()
    df["glucose_mean_60m"] = df["glucose"].rolling(window=12, min_periods=1).mean()

    # 2. Glucose trends (velocity in mg/dL per minute)
    # df.shift(k) is strictly causal (historical)
    df["glucose_trend_15m"] = (df["glucose"] - df["glucose"].shift(3).fillna(df["glucose"])) / 15.0
    df["glucose_trend_30m"] = (df["glucose"] - df["glucose"].shift(6).fillna(df["glucose"])) / 30.0
    df["glucose_trend_60m"] = (df["glucose"] - df["glucose"].shift(12).fillna(df["glucose"])) / 60.0

    # 3. Glycemic variability and range over past 60 minutes
    df["glucose_std_60m"] = df["glucose"].rolling(window=12, min_periods=2).std().fillna(0.0)
    df["glucose_min_60m"] = df["glucose"].rolling(window=12, min_periods=1).min()
    df["glucose_max_60m"] = df["glucose"].rolling(window=12, min_periods=1).max()

    # 4. Meal carbohydrates accumulation
    meal_col = df["meal_carbs"] if "meal_carbs" in df.columns else pd.Series(0.0, index=df.index)
    df["carbs_last_30m"] = meal_col.rolling(window=6, min_periods=1).sum()
    df["carbs_last_60m"] = meal_col.rolling(window=12, min_periods=1).sum()
    df["carbs_last_120m"] = meal_col.rolling(window=24, min_periods=1).sum()

    # Minutes since last meal (causal scan)
    mins_since_meal = []
    last_meal_time = None
    for idx, row in df.iterrows():
        t = row["timestamp"]
        if row.get("meal_carbs", 0.0) >= 5.0:
            last_meal_time = t
        if last_meal_time is None:
            mins_since_meal.append(240.0)  # Default fasting state
        else:
            diff_mins = (t - last_meal_time).total_seconds() / 60.0
            mins_since_meal.append(min(480.0, max(0.0, diff_mins)))
    df["mins_since_last_meal"] = mins_since_meal

    # 5. Activity metrics
    steps_col = df["steps"] if "steps" in df.columns else pd.Series(0, index=df.index)
    df["activity_steps_last_30m"] = steps_col.rolling(window=6, min_periods=1).sum()
    df["activity_steps_last_60m"] = steps_col.rolling(window=12, min_periods=1).sum()

    # 6. Wearable Heart rate & Sleep
    hr_col = df["heart_rate"] if "heart_rate" in df.columns else pd.Series(72.0, index=df.index)
    df["heart_rate_current"] = hr_col
    df["heart_rate_mean_30m"] = hr_col.rolling(window=6, min_periods=1).mean()

    # Sleep hours and diurnal night flag
    if "sleep_hours" in df.columns:
        df["sleep_hours_prior_night"] = df["sleep_hours"]
    elif profile is not None:
        df["sleep_hours_prior_night"] = profile.average_sleep_hours
    else:
        df["sleep_hours_prior_night"] = 7.0

    hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    df["is_nighttime"] = ((hour >= 22.0) | (hour < 6.0)).astype(int)

    # 7. Static patient baseline EHR features
    if profile is None and "patient_id" in df.columns and len(df) > 0:
        pid = df["patient_id"].iloc[0]
        profile = get_patient_profile(pid)

    if profile is not None:
        df["age"] = profile.age
        df["bmi"] = profile.bmi
        df["diabetes_duration_years"] = profile.diabetes_duration_years
        df["hba1c"] = profile.hba1c
        df["on_insulin"] = profile.on_insulin
    else:
        df["age"] = 55
        df["bmi"] = 28.0
        df["diabetes_duration_years"] = 5.0
        df["hba1c"] = 7.5
        df["on_insulin"] = 0

    return df


def construct_prediction_target(
    df: pd.DataFrame,
    threshold: float = DEFAULT_HIGH_GLUCOSE_THRESHOLD,
    horizon_steps: int = HORIZON_STEPS,
) -> pd.DataFrame:
    """
    Constructs the binary prediction target:
    Will glucose exceed `threshold` in the next `horizon_steps` (e.g. 24 steps = 120 minutes)?

    IMPORTANT: To prevent data leakage, this uses future forward window (t+1 to t+horizon_steps).
    The last `horizon_steps` rows DO NOT have a complete future window, so their target is NaN.
    """
    df = df.copy()
    future_max_glucose = pd.Series(np.nan, index=df.index)

    # For each index i, evaluate max glucose in [i + 1, i + horizon_steps]
    glucose_vals = df["glucose"].values
    n = len(glucose_vals)

    targets = np.full(n, np.nan)
    for i in range(n - horizon_steps):
        window_future = glucose_vals[i + 1 : i + 1 + horizon_steps]
        max_val = np.max(window_future)
        targets[i] = 1 if max_val >= threshold else 0

    df["target_high_glucose_120m"] = targets
    return df


def prepare_dataset_for_modeling(
    timeseries_df: pd.DataFrame,
    profile: Optional[PatientProfile] = None,
    threshold: float = DEFAULT_HIGH_GLUCOSE_THRESHOLD,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Computes all features and targets, drops trailing rows with incomplete future windows,
    and returns (X, y) ready for model training or evaluation.
    """
    feat_df = compute_temporal_features(timeseries_df, profile=profile)
    labeled_df = construct_prediction_target(feat_df, threshold=threshold)

    # Drop rows where target is NaN (incomplete future horizon)
    valid_df = labeled_df.dropna(subset=["target_high_glucose_120m"]).copy()
    valid_df["target_high_glucose_120m"] = valid_df["target_high_glucose_120m"].astype(int)

    X = valid_df[FEATURE_COLUMNS]
    y = valid_df["target_high_glucose_120m"]
    return X, y


def extract_features_for_single_point(
    history_df: pd.DataFrame,
    profile: PatientProfile,
) -> pd.DataFrame:
    """
    Extracts the feature vector at the latest point (row -1) in history_df.
    Guarantees strict causality for live inference in the Streamlit app.
    """
    feat_df = compute_temporal_features(history_df, profile=profile)
    latest_row = feat_df.iloc[[-1]][FEATURE_COLUMNS]
    return latest_row
