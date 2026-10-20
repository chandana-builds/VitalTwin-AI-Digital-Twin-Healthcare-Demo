"""
Synthetic Continuous Glucose Monitoring (CGM) & Wearable Sensor Stream Generator
VitalTwin AI - Happiest Health Digital Twin Challenge 2026

Simulates realistic, physiologically grounded time series of:
- Interstitial Glucose (CGM, 5-minute sampling)
- Heart Rate (Photoplethysmography / Wearable)
- Physical Activity (Step counts and intensity)
- Meals (Carbohydrate grams and timing)
- Sleep state & quality
"""

import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from .patient_generator import PatientProfile, get_patient_profile


class SyntheticSensorStream:
    """
    Simulates a continuous physiological stream for a synthetic digital twin patient.
    Implements gut absorption dynamics, diurnal circadian rhythms, physical activity
    glucose clearance, and sensor measurement noise.
    """

    def __init__(
        self,
        profile: PatientProfile,
        start_time: Optional[datetime] = None,
        random_seed: int = 42,
    ):
        self.profile = profile
        self.rng = np.random.RandomState(random_seed)
        self.current_time = start_time or datetime(2026, 10, 20, 8, 0, 0)
        
        # State variables
        self.glucose = profile.fasting_glucose_baseline
        self.resting_hr = 68.0 + (self.profile.bmi - 25.0) * 0.8
        self.noise_state = 0.0
        
        # Active meals list: each meal is dict(time, carbs, peak_time_min, duration_min)
        self.active_meals: List[Dict[str, Any]] = []

    def _get_circadian_offset(self, dt: datetime) -> float:
        """
        Computes diurnal glucose variation (Dawn phenomenon & nocturnal nadir).
        Dawn phenomenon peak typically occurs around 05:30 - 07:30 AM.
        """
        hour = dt.hour + dt.minute / 60.0
        # Dawn phenomenon peak around 6.5 AM
        dawn_intensity = 18.0 if self.profile.hba1c > 8.0 else 8.0
        dawn_effect = dawn_intensity * math.exp(-0.5 * ((hour - 6.5) / 1.5) ** 2)
        
        # Circadian base wave (lowest at 3 AM, highest in afternoon)
        base_cycle = 6.0 * math.sin(2 * math.pi * (hour - 3.0) / 24.0)
        return dawn_effect + base_cycle

    def _is_sleep_period(self, dt: datetime) -> bool:
        """Determines if the current timestamp falls in nominal sleep hours (22:30 - 06:30)."""
        hour = dt.hour + dt.minute / 60.0
        return hour >= 22.5 or hour < 6.5

    def add_meal(self, carbs_grams: float, peak_mins: float = 45.0, duration_mins: float = 180.0):
        """Register a meal event starting at the current time."""
        if carbs_grams > 0:
            self.active_meals.append({
                "start_time": self.current_time,
                "carbs": float(carbs_grams),
                "peak_mins": float(peak_mins),
                "duration_mins": float(duration_mins),
            })

    def _compute_active_meal_impact(self, current_dt: datetime) -> float:
        """
        Calculates cumulative glycemic rate of appearance from active meals using
        a standard gamma-like absorption curve:
        dG/dt ~ (t / tau) * exp(1 - t / tau)
        """
        total_meal_rate = 0.0
        remaining_meals = []

        for meal in self.active_meals:
            elapsed_mins = (current_dt - meal["start_time"]).total_seconds() / 60.0
            if 0 <= elapsed_mins <= meal["duration_mins"]:
                tau = meal["peak_mins"]
                # Normalized absorption shape peaking at tau with value 1.0
                shape = (elapsed_mins / tau) * math.exp(1.0 - (elapsed_mins / tau))
                # Total peak excursion scales with patient carb sensitivity
                max_excursion = meal["carbs"] * self.profile.carb_sensitivity
                # Incremental glucose contribution in mg/dL per 5-min step
                # Total area under curve approximately matches max_excursion
                step_contribution = (max_excursion / (meal["duration_mins"] / 5.0)) * shape * 2.2
                total_meal_rate += step_contribution
                remaining_meals.append(meal)
            elif elapsed_mins < 0:
                remaining_meals.append(meal)

        self.active_meals = remaining_meals
        return total_meal_rate

    def step(
        self,
        minutes: int = 5,
        meal_carbs: float = 0.0,
        manual_activity_steps: Optional[int] = None,
        manual_hr: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Advances the simulation by the specified number of minutes (typically 5 mins).
        Returns a dictionary representing the observation point.
        """
        dt_step = minutes
        self.current_time += timedelta(minutes=dt_step)
        hour = self.current_time.hour + self.current_time.minute / 60.0

        if meal_carbs > 0:
            self.add_meal(meal_carbs)

        is_sleeping = self._is_sleep_period(self.current_time)

        # Activity simulation
        if manual_activity_steps is not None:
            steps = manual_activity_steps
        elif is_sleeping:
            steps = int(self.rng.choice([0, 2, 5]))
        else:
            # Daytime base steps + occasional bursts (commute, walks)
            is_walk_time = (8.2 <= hour <= 8.7) or (13.0 <= hour <= 13.5) or (18.5 <= hour <= 19.2)
            if is_walk_time:
                steps = int(self.rng.normal(240, 40))
            else:
                steps = int(self.rng.exponential(25))
            steps = max(0, steps)

        # Activity-based glucose clearance
        activity_clearance = steps * self.profile.activity_clearance_factor * (dt_step / 5.0)

        # Meal absorption
        meal_glucose_influx = self._compute_active_meal_impact(self.current_time)

        # Baseline homeostasis pull
        baseline_target = self.profile.fasting_glucose_baseline + self._get_circadian_offset(self.current_time)
        # Pull toward baseline (clearance rate)
        pull_rate = 0.04 * (dt_step / 5.0)
        homeostatic_adjustment = (baseline_target - self.glucose) * pull_rate

        # Autoregressive sensor and physiological noise (AR-1)
        self.noise_state = 0.85 * self.noise_state + self.rng.normal(0.0, 1.2)

        # Update glucose state
        delta_g = homeostatic_adjustment + meal_glucose_influx - activity_clearance + self.noise_state
        self.glucose = max(45.0, min(420.0, self.glucose + delta_g))

        # Heart rate simulation
        if manual_hr is not None:
            heart_rate = manual_hr
        elif is_sleeping:
            heart_rate = self.resting_hr - 12.0 + self.rng.normal(0, 1.5)
        else:
            # Elevated with steps
            activity_elevation = min(55.0, steps * 0.12)
            # Slight postprandial elevation
            pp_elevation = 4.0 if meal_glucose_influx > 1.0 else 0.0
            heart_rate = self.resting_hr + activity_elevation + pp_elevation + self.rng.normal(0, 2.5)

        heart_rate = float(np.clip(heart_rate, 45.0, 185.0))

        # Sleep quality / duration metrics
        sleep_hours = self.profile.average_sleep_hours + (0.5 if not is_sleeping else 0.0)

        observation = {
            "timestamp": self.current_time,
            "patient_id": self.profile.patient_id,
            "glucose": round(float(self.glucose), 1),
            "heart_rate": round(heart_rate, 1),
            "steps": int(steps),
            "meal_carbs": float(meal_carbs),
            "is_sleeping": int(is_sleeping),
            "sleep_hours": round(float(sleep_hours), 1),
        }
        return observation


def generate_patient_timeseries(
    patient_id: str = "PT-1001",
    days: int = 3,
    start_time: Optional[datetime] = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generates a continuous, high-fidelity synthetic time series for a specified patient.
    Default: 3 days of 5-minute sampling = 864 observations.
    Includes realistic daily breakfast, lunch, snack, and dinner meal events.
    """
    profile = get_patient_profile(patient_id)
    stream = SyntheticSensorStream(profile=profile, start_time=start_time, random_seed=random_seed)

    total_steps = days * 24 * 12  # 12 observations per hour
    records = []

    # Nominal daily meal schedule: (hour, minute, avg_carbs, std_carbs)
    meal_schedule = [
        (8, 0, 45, 10),    # Breakfast
        (12, 30, 65, 15),  # Lunch
        (16, 30, 20, 5),   # Afternoon Snack
        (19, 30, 70, 15),  # Dinner
    ]

    for step_i in range(total_steps):
        next_dt = stream.current_time + timedelta(minutes=5)
        
        # Check if a meal happens at this 5-min step
        meal_carbs = 0.0
        for m_hour, m_min, m_avg, m_std in meal_schedule:
            if next_dt.hour == m_hour and abs(next_dt.minute - m_min) < 3:
                # Randomize meal carbs slightly
                meal_carbs = max(10.0, float(stream.rng.normal(m_avg, m_std)))
                break

        obs = stream.step(minutes=5, meal_carbs=meal_carbs)
        records.append(obs)

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df
