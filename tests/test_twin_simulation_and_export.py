"""
Unit tests for Digital Twin Simulation Engine, What-If Simulator, and CSV Export
VitalTwin AI - Happiest Health Digital Twin Challenge 2026
"""

from datetime import datetime, timedelta
import pytest
import pandas as pd

from src.models.predictor import train_and_evaluate_model
from src.simulation.twin_engine import DigitalTwinEngine, WhatIfScenarioSimulator
from src.utils.export_utils import export_observations_csv, export_predictions_csv


@pytest.fixture(scope="module")
def shared_predictor():
    """Trained predictor for simulation tests."""
    pred, _ = train_and_evaluate_model(threshold=180.0, random_seed=42)
    return pred


def test_twin_initialization_and_metrics(shared_predictor):
    """Verify digital twin initialization and initial metric summaries."""
    twin = DigitalTwinEngine(patient_id="PT-1001", predictor=shared_predictor, initial_history_hours=12)

    assert twin.patient_id == "PT-1001"
    assert twin.profile.name.startswith("Eleanor Vance")
    assert len(twin.history_df) == 12 * 12  # 144 steps
    assert twin.latest_prediction is not None
    assert "probability" in twin.latest_prediction

    summary = twin.get_summary_metrics()
    assert "current_glucose" in summary
    assert "tir_pct" in summary
    assert "cv_pct" in summary
    assert 0.0 <= summary["tir_pct"] <= 100.0


def test_twin_step_forward(shared_predictor):
    """Verify manual stepping advances time and triggers state update."""
    twin = DigitalTwinEngine(patient_id="PT-1001", predictor=shared_predictor, initial_history_hours=6)
    initial_time = twin.current_time
    initial_len = len(twin.history_df)

    step_res = twin.step_forward(minutes=5, meal_carbs=0.0)

    assert twin.current_time == initial_time + timedelta(minutes=5)
    assert len(twin.history_df) == initial_len + 1
    assert step_res["glucose"] == twin.current_glucose
    assert twin.latest_prediction["timestamp"] == twin.current_time


def test_patient_switching(shared_predictor):
    """Verify that switching patient profiles completely updates the twin state."""
    twin = DigitalTwinEngine(patient_id="PT-1001", predictor=shared_predictor, initial_history_hours=6)
    assert twin.profile.patient_id == "PT-1001"

    # Switch to PT-1002 (brittle, higher HbA1c)
    twin.reset(patient_id="PT-1002", initial_history_hours=6)
    assert twin.profile.patient_id == "PT-1002"
    assert twin.profile.hba1c == 8.9
    assert twin.history_df["patient_id"].iloc[-1] == "PT-1002"


def test_what_if_scenario_simulation(shared_predictor):
    """Verify that the What-If sandbox computes differences without altering baseline state."""
    twin = DigitalTwinEngine(patient_id="PT-1001", predictor=shared_predictor, initial_history_hours=6)
    simulator = WhatIfScenarioSimulator(shared_predictor)

    baseline_features = dict(twin.latest_prediction["features"])

    # Simulate scenario: high carb intake (80g)
    scen_result = simulator.simulate_scenario(
        baseline_features=baseline_features,
        hypothetical_carbs=80.0,
        hypothetical_activity_steps=0,
    )

    assert "baseline_probability" in scen_result
    assert "scenario_probability" in scen_result
    assert "delta_probability" in scen_result
    assert "scenario_explanation" in scen_result

    # 80g carb meal should elevate risk relative to baseline
    assert scen_result["scenario_probability"] >= scen_result["baseline_probability"]

    # Verify baseline patient state was NOT mutated
    assert twin.latest_prediction["features"] == baseline_features


def test_csv_export_validity(shared_predictor):
    """Verify CSV export utility produces valid, non-empty text outputs."""
    twin = DigitalTwinEngine(patient_id="PT-1001", predictor=shared_predictor, initial_history_hours=6)
    csv_str = export_observations_csv(twin.history_df)

    assert isinstance(csv_str, str)
    assert "timestamp,patient_id,glucose" in csv_str
    assert len(csv_str.splitlines()) > 50

    pred_csv = export_predictions_csv(twin.history_df, [twin.latest_prediction])
    assert isinstance(pred_csv, str)
    assert len(pred_csv.splitlines()) >= 2
