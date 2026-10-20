"""
VitalTwin AI - Configuration and Clinical Simulation Parameters
Challenge: Happiest Health Digital Twin Challenge 2026
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List

PROJECT_NAME = "VitalTwin AI"
TAGLINE = "An Explainable Digital Twin for Predictive Healthcare"
CHALLENGE_NAME = "Happiest Health Digital Twin Challenge 2026"
HEALTHCARE_FOCUS = "Type 2 Diabetes and Two-Hour-Ahead Glucose Risk Prediction"
TARGET_USERS = "Healthcare researchers and clinicians evaluating proactive-care concepts"

# Non-Clinical Disclaimer
DISCLAIMER_TEXT = (
    "RESEARCH PROOF OF CONCEPT ONLY. VitalTwin AI utilizes synthetic physiological data "
    "and mathematical approximations for algorithm demonstration. It is NOT a certified "
    "medical device, diagnostic tool, or clinical decision support system. Do not use for "
    "actual patient therapy, insulin dosing, or clinical treatment."
)

# Clinical Thresholds (mg/dL) - ADA Guidelines
DEFAULT_HIGH_GLUCOSE_THRESHOLD = 180.0  # Standard postprandial target ceiling
SEVERE_HYPERGLYCEMIA_THRESHOLD = 250.0
HYPOGLYCEMIA_THRESHOLD = 70.0
SEVERE_HYPOGLYCEMIA_THRESHOLD = 54.0
TARGET_GLUCOSE_LOWER = 70.0
TARGET_GLUCOSE_UPPER = 180.0

# Prediction Engine Parameters
PREDICTION_HORIZON_MINUTES = 120  # 2 hours ahead
SAMPLING_INTERVAL_MINUTES = 5     # 5-minute CGM standard interval
HORIZON_STEPS = PREDICTION_HORIZON_MINUTES // SAMPLING_INTERVAL_MINUTES # 24 steps

# Risk Probability Thresholds
RISK_THRESHOLD_LOW = 0.30
RISK_THRESHOLD_HIGH = 0.60

# Palette Configuration for UI Aesthetics (Navy, Teal, White, Accents)
THEME_COLORS = {
    "primary_navy": "#0F172A",      # Slate 900
    "secondary_navy": "#1E293B",    # Slate 800
    "card_bg": "#1E293B",
    "accent_teal": "#0D9488",       # Teal 600
    "bright_teal": "#14B8A6",       # Teal 500
    "light_teal": "#2DD4BF",        # Teal 400
    "alert_red": "#EF4444",         # Red 500
    "alert_amber": "#F59E0B",       # Amber 500
    "success_green": "#10B981",     # Emerald 500
    "neutral_white": "#FFFFFF",
    "neutral_light": "#F8FAFC",     # Slate 50
    "neutral_gray": "#94A3B8",      # Slate 400
    "border_color": "#334155"       # Slate 700
}

# Features List for Model Training and Inference
FEATURE_COLUMNS = [
    "glucose_current",
    "glucose_mean_15m",
    "glucose_mean_30m",
    "glucose_mean_60m",
    "glucose_trend_15m",
    "glucose_trend_30m",
    "glucose_trend_60m",
    "glucose_std_60m",
    "glucose_min_60m",
    "glucose_max_60m",
    "carbs_last_30m",
    "carbs_last_60m",
    "carbs_last_120m",
    "mins_since_last_meal",
    "activity_steps_last_30m",
    "activity_steps_last_60m",
    "heart_rate_current",
    "heart_rate_mean_30m",
    "sleep_hours_prior_night",
    "is_nighttime",
    "age",
    "bmi",
    "diabetes_duration_years",
    "hba1c",
    "on_insulin"
]
