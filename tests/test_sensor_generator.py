"""
Unit tests for Synthetic Sensor Stream Generator
VitalTwin AI - Happiest Health Digital Twin Challenge 2026
"""

from datetime import datetime, timedelta
import pytest
import numpy as np
import pandas as pd

from src.data.patient_generator import get_patient_profile
from src.data.sensor_generator import SyntheticSensorStream, generate_patient_timeseries


def test_sensor_stream_step_attributes():
    """Verify that each simulation step returns all expected physiological sensor channels."""
    profile = get_patient_profile("PT-1001")
    stream = SyntheticSensorStream(profile=profile, random_seed=42)

    obs = stream.step(minutes=5, meal_carbs=0.0)
    assert isinstance(obs, dict)
    expected_keys = [
        "timestamp",
        "patient_id",
        "glucose",
        "heart_rate",
        "steps",
        "meal_carbs",
        "is_sleeping",
        "sleep_hours",
    ]
    for key in expected_keys:
        assert key in obs, f"Missing sensor channel: {key}"

    assert 40.0 <= obs["glucose"] <= 450.0
    assert 40.0 <= obs["heart_rate"] <= 200.0
    assert obs["steps"] >= 0


def test_sensor_meal_excursion():
    """Verify that registering a meal triggers an upward postprandial glucose excursion."""
    profile = get_patient_profile("PT-1001")
    stream = SyntheticSensorStream(profile=profile, random_seed=42)

    # Initial baseline
    baseline_g = stream.glucose
    # Ingest 75g carb meal
    stream.add_meal(carbs_grams=75.0, peak_mins=45.0)

    # Advance 12 steps (60 minutes)
    glucose_trace = []
    for _ in range(12):
        obs = stream.step(minutes=5)
        glucose_trace.append(obs["glucose"])

    # Peak excursion within 60 minutes should exceed baseline
    assert max(glucose_trace) > baseline_g, "Meal failed to generate postprandial glycemic rise"


def test_activity_clearance():
    """Verify that high physical activity clears glucose compared to resting state."""
    profile = get_patient_profile("PT-1001")
    
    # Run resting stream
    stream_rest = SyntheticSensorStream(profile=profile, random_seed=123)
    stream_rest.glucose = 200.0
    for _ in range(12):
        stream_rest.step(minutes=5, manual_activity_steps=0)

    # Run active stream (brisk walking 400 steps per 5-min)
    stream_active = SyntheticSensorStream(profile=profile, random_seed=123)
    stream_active.glucose = 200.0
    for _ in range(12):
        stream_active.step(minutes=5, manual_activity_steps=400)

    assert stream_active.glucose < stream_rest.glucose, "Physical activity should exert glucose clearing effect"


def test_generate_patient_timeseries_length_and_structure():
    """Verify multi-day time series generation structure."""
    days = 1
    df = generate_patient_timeseries(patient_id="PT-1001", days=days, random_seed=42)

    expected_rows = days * 24 * 12  # 288 rows for 1 day at 5-min intervals
    assert len(df) == expected_rows
    assert "timestamp" in df.columns
    assert "glucose" in df.columns
    assert "heart_rate" in df.columns
    assert "steps" in df.columns
    # Check timestamps are sequential
    diffs = df["timestamp"].diff().dropna()
    assert (diffs == timedelta(minutes=5)).all(), "Time series intervals must strictly be 5 minutes"
