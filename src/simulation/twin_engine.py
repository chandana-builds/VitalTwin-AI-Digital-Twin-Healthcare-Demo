"""
Digital Twin State Engine and What-If Scenario Sandbox
VitalTwin AI - Happiest Health Digital Twin Challenge 2026

Coordinates real-time patient twin state evolution, streaming sensor observations,
prediction triggering, and isolated hypothetical intervention simulations.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from config.settings import (
    DEFAULT_HIGH_GLUCOSE_THRESHOLD,
    FEATURE_COLUMNS,
)
from src.data.patient_generator import PatientProfile, get_patient_profile
from src.data.sensor_generator import SyntheticSensorStream, generate_patient_timeseries
from src.features.feature_engineering import (
    compute_temporal_features,
    extract_features_for_single_point,
)
from src.models.predictor import GlucoseRiskPredictor


class DigitalTwinEngine:
    """
    Maintains the dynamic internal state of the virtual patient digital twin.
    Simulates real-time sensor streams, tracks rolling history, and coordinates
    machine-learning predictions.
    """

    def __init__(
        self,
        patient_id: str = "PT-1001",
        predictor: Optional[GlucoseRiskPredictor] = None,
        initial_history_hours: int = 24,
        random_seed: int = 42,
    ):
        self.random_seed = random_seed
        self.predictor = predictor
        self.patient_id = patient_id
        self.profile = get_patient_profile(patient_id)

        # Simulation clock and state
        self.is_running = False
        self.reset(patient_id, initial_history_hours=initial_history_hours)

    def reset(self, patient_id: Optional[str] = None, initial_history_hours: int = 24) -> None:
        """Resets the twin to initial baseline history."""
        if patient_id:
            self.patient_id = patient_id
            self.profile = get_patient_profile(patient_id)

        # Generate initial burn-in history (e.g. 24 hours of 5-min readings = 288 steps)
        days = max(1, int(np.ceil(initial_history_hours / 24.0)))
        raw_df = generate_patient_timeseries(
            patient_id=self.patient_id,
            days=days,
            random_seed=self.random_seed,
        )
        # Keep the exact number of hours requested
        n_points = initial_history_hours * 12
        self.history_df = raw_df.iloc[-n_points:].copy().reset_index(drop=True)

        # Initialize live sensor stream continuing from last history timestamp
        last_dt = self.history_df["timestamp"].iloc[-1]
        self.sensor_stream = SyntheticSensorStream(
            profile=self.profile,
            start_time=last_dt,
            random_seed=self.random_seed + 100,
        )
        self.sensor_stream.glucose = float(self.history_df["glucose"].iloc[-1])

        # Current state snapshot
        self.current_time = last_dt
        self.current_glucose = float(self.history_df["glucose"].iloc[-1])
        self.current_heart_rate = float(self.history_df["heart_rate"].iloc[-1])
        self.current_steps = int(self.history_df["steps"].iloc[-1])

        # Run initial prediction
        self.latest_prediction: Optional[Dict[str, Any]] = None
        self._refresh_prediction()

    def step_forward(
        self,
        minutes: int = 5,
        meal_carbs: float = 0.0,
        manual_activity_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Advances the virtual patient by `minutes` (e.g. 5 or 15 mins),
        records the observation, and triggers updated model inference.
        """
        steps_count = minutes // 5
        last_obs = None

        for i in range(max(1, steps_count)):
            carbs_to_add = meal_carbs if i == 0 else 0.0
            last_obs = self.sensor_stream.step(
                minutes=5,
                meal_carbs=carbs_to_add,
                manual_activity_steps=manual_activity_steps,
            )
            # Append new observation row to history
            new_row_df = pd.DataFrame([last_obs])
            self.history_df = pd.concat([self.history_df, new_row_df], ignore_index=True)

        # Update twin state variables
        self.current_time = self.sensor_stream.current_time
        self.current_glucose = round(float(self.sensor_stream.glucose), 1)
        self.current_heart_rate = round(float(last_obs["heart_rate"]), 1)
        self.current_steps = int(last_obs["steps"])

        # Trigger fresh prediction
        self._refresh_prediction()

        return {
            "timestamp": self.current_time,
            "glucose": self.current_glucose,
            "heart_rate": self.current_heart_rate,
            "steps": self.current_steps,
            "prediction": self.latest_prediction,
        }

    def _refresh_prediction(self) -> None:
        """Computes causal features from history_df and performs model inference."""
        if self.predictor is None or not self.predictor.is_trained:
            self.latest_prediction = {
                "probability": 0.0,
                "risk_tier": "Uncalibrated",
                "top_factors": [],
                "timestamp": self.current_time,
            }
            return

        feat_row = extract_features_for_single_point(self.history_df, self.profile)
        proba, tier, factors = self.predictor.predict_risk(feat_row)

        self.latest_prediction = {
            "probability": proba,
            "risk_tier": tier,
            "top_factors": factors,
            "timestamp": self.current_time,
            "features": feat_row.to_dict(orient="records")[0],
        }

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Calculates key ambulatory glucose metrics (Time in Range, Mean, SD, Trend)."""
        recent_df = self.history_df.iloc[-288:]  # Past 24 hours max
        glucose_vals = recent_df["glucose"].values

        in_range = np.sum((glucose_vals >= 70.0) & (glucose_vals <= 180.0))
        tir_pct = (in_range / len(glucose_vals)) * 100.0 if len(glucose_vals) > 0 else 0.0

        above_range = np.sum(glucose_vals > 180.0)
        tar_pct = (above_range / len(glucose_vals)) * 100.0 if len(glucose_vals) > 0 else 0.0

        below_range = np.sum(glucose_vals < 70.0)
        tbr_pct = (below_range / len(glucose_vals)) * 100.0 if len(glucose_vals) > 0 else 0.0

        mean_g = float(np.mean(glucose_vals))
        std_g = float(np.std(glucose_vals))
        cv_pct = (std_g / mean_g * 100.0) if mean_g > 0 else 0.0

        # Recent trend (last 15 mins)
        if len(glucose_vals) >= 4:
            trend_val = (glucose_vals[-1] - glucose_vals[-4]) / 15.0
        else:
            trend_val = 0.0

        return {
            "current_glucose": self.current_glucose,
            "trend_rate": round(trend_val, 2),
            "tir_pct": round(tir_pct, 1),
            "tar_pct": round(tar_pct, 1),
            "tbr_pct": round(tbr_pct, 1),
            "mean_glucose": round(mean_g, 1),
            "std_glucose": round(std_g, 1),
            "cv_pct": round(cv_pct, 1),
            "total_observations": len(self.history_df),
        }


class WhatIfScenarioSimulator:
    """
    Evaluates hypothetical changes in meal carbohydrates, physical activity,
    sleep, or diurnal timing without altering baseline patient twin records.
    """

    def __init__(self, predictor: GlucoseRiskPredictor):
        self.predictor = predictor

    def simulate_scenario(
        self,
        baseline_features: Dict[str, Any],
        hypothetical_carbs: float = 0.0,
        hypothetical_activity_steps: int = 0,
        hypothetical_sleep_hours: float = 7.0,
        is_nighttime_override: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Modifies feature vector with hypothetical interventions and re-evaluates risk.
        Strictly preserves baseline state.
        """
        # Copy feature dictionary
        scenario_features = dict(baseline_features)

        # Baseline prediction
        df_base = pd.DataFrame([baseline_features])
        base_proba, base_tier, _ = self.predictor.predict_risk(df_base)

        # Apply hypothetical changes
        # 1. Carbs addition
        if hypothetical_carbs > 0:
            scenario_features["carbs_last_30m"] = scenario_features.get("carbs_last_30m", 0.0) + hypothetical_carbs
            scenario_features["carbs_last_60m"] = scenario_features.get("carbs_last_60m", 0.0) + hypothetical_carbs
            scenario_features["mins_since_last_meal"] = 5.0  # Fresh meal
            # Expected immediate upward glucose velocity push
            scenario_features["glucose_trend_15m"] = scenario_features.get("glucose_trend_15m", 0.0) + (hypothetical_carbs * 0.03)

        # 2. Activity addition
        if hypothetical_activity_steps > 0:
            scenario_features["activity_steps_last_30m"] = scenario_features.get("activity_steps_last_30m", 0) + hypothetical_activity_steps
            scenario_features["activity_steps_last_60m"] = scenario_features.get("activity_steps_last_60m", 0) + hypothetical_activity_steps
            # Physical activity exerts downward dampening on glucose trend
            trend_reduction = (hypothetical_activity_steps / 500.0) * 0.4
            scenario_features["glucose_trend_15m"] = scenario_features.get("glucose_trend_15m", 0.0) - trend_reduction

        # 3. Sleep modification
        scenario_features["sleep_hours_prior_night"] = float(hypothetical_sleep_hours)

        # 4. Nighttime override
        if is_nighttime_override is not None:
            scenario_features["is_nighttime"] = int(is_nighttime_override)

        # Scenario prediction
        df_scen = pd.DataFrame([scenario_features])
        scen_proba, scen_tier, top_factors = self.predictor.predict_risk(df_scen)

        delta_proba = scen_proba - base_proba
        delta_pct = delta_proba * 100.0

        # Scenario textual explanation
        explanation = self._build_explanation(
            delta_pct,
            hypothetical_carbs,
            hypothetical_activity_steps,
            hypothetical_sleep_hours,
        )

        return {
            "baseline_probability": round(base_proba, 4),
            "baseline_risk_tier": base_tier,
            "scenario_probability": round(scen_proba, 4),
            "scenario_risk_tier": scen_tier,
            "delta_probability": round(delta_proba, 4),
            "delta_percentage_points": round(delta_pct, 2),
            "changed_features": {
                "hypothetical_carbs": hypothetical_carbs,
                "hypothetical_activity_steps": hypothetical_activity_steps,
                "hypothetical_sleep_hours": hypothetical_sleep_hours,
            },
            "top_scenario_factors": top_factors,
            "scenario_explanation": explanation,
        }

    @staticmethod
    def _build_explanation(
        delta_pct: float,
        carbs: float,
        steps: int,
        sleep: float,
    ) -> str:
        """Generates clear, explainable narrative of the hypothetical simulation outcome."""
        parts = []
        if carbs > 0:
            parts.append(f"hypothetical intake of {carbs:.0f}g carbohydrates")
        if steps > 0:
            parts.append(f"post-meal physical activity of {steps} steps")
        if sleep < 6.0:
            parts.append(f"short sleep duration ({sleep:.1f}h)")

        context = "With " + ", and ".join(parts) if parts else "With the specified baseline parameters"
        if delta_pct > 5.0:
            direction = f"increases the predicted 2-hour excursion risk by +{delta_pct:.1f}% points."
        elif delta_pct < -5.0:
            direction = f"attenuates the predicted 2-hour excursion risk by {delta_pct:.1f}% points."
        else:
            direction = f"results in minimal change ({delta_pct:+.1f}% points) to the 2-hour risk."

        disclaimer = (
            " (Note: Digital twin simulation estimate for research review; "
            "not a personalized prescription or therapeutic promise.)"
        )
        return f"{context}, the model {direction}{disclaimer}"
