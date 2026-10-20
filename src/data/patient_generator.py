"""
Synthetic Patient Profile Generator
VitalTwin AI - Happiest Health Digital Twin Challenge 2026

Generates realistic, medically coherent synthetic Electronic Health Record (EHR) profiles
for individuals with Type 2 Diabetes Mellitus (T2DM).
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Any


@dataclass
class PatientProfile:
    patient_id: str
    name: str
    age: int
    sex: str
    bmi: float
    blood_pressure: str
    diabetes_duration_years: float
    hba1c: float
    medication_status: str
    on_insulin: int  # 0: No, 1: Yes
    fasting_glucose_baseline: float
    carb_sensitivity: float  # Excursion amplitude per gram of carbohydrate
    activity_clearance_factor: float  # Glucose clearance per MET-minute
    baseline_activity_level: str
    average_sleep_hours: float
    target_range_lower: float = 70.0
    target_range_upper: float = 180.0
    clinical_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_synthetic_patient_profiles() -> Dict[str, PatientProfile]:
    """
    Returns 3 distinct synthetic patient profiles representing standard clinical phenotypes
    in Type 2 Diabetes proactive care.
    All records are fictional and synthetic.
    """
    profiles = {
        "PT-1001": PatientProfile(
            patient_id="PT-1001",
            name="Eleanor Vance (Synthetic)",
            age=58,
            sex="Female",
            bmi=27.4,
            blood_pressure="128/82 mmHg",
            diabetes_duration_years=6.0,
            hba1c=7.4,
            medication_status="Metformin 1000mg BID",
            on_insulin=0,
            fasting_glucose_baseline=132.0,
            carb_sensitivity=0.85,
            activity_clearance_factor=0.035,
            baseline_activity_level="Moderate (Daily 30-min walking)",
            average_sleep_hours=7.2,
            target_range_lower=70.0,
            target_range_upper=180.0,
            clinical_notes=(
                "Patient with 6-year history of well-managed T2D. Shows moderate postprandial "
                "glycemic response to high-carb lunches. Compliant with oral hypoglycemics. "
                "Responds favorably to post-meal walks."
            ),
        ),
        "PT-1002": PatientProfile(
            patient_id="PT-1002",
            name="Marcus Reed (Synthetic)",
            age=64,
            sex="Male",
            bmi=32.8,
            blood_pressure="142/88 mmHg",
            diabetes_duration_years=14.0,
            hba1c=8.9,
            medication_status="Basal Insulin Glargine 28u + Empagliflozin 10mg",
            on_insulin=1,
            fasting_glucose_baseline=174.0,
            carb_sensitivity=1.25,
            activity_clearance_factor=0.020,
            baseline_activity_level="Sedentary (<3000 steps/day)",
            average_sleep_hours=5.8,
            target_range_lower=70.0,
            target_range_upper=180.0,
            clinical_notes=(
                "Long-standing T2D with high glycemic variability (GV). Pronounced dawn phenomenon "
                "and sustained postprandial excursions exceeding 200 mg/dL. Sleep fragmentation "
                "compounds morning hepatic glucose release."
            ),
        ),
        "PT-1003": PatientProfile(
            patient_id="PT-1003",
            name="Priya Sharma (Synthetic)",
            age=42,
            sex="Female",
            bmi=24.6,
            blood_pressure="118/76 mmHg",
            diabetes_duration_years=1.5,
            hba1c=6.7,
            medication_status="Metformin 500mg Once Daily + Lifestyle",
            on_insulin=0,
            fasting_glucose_baseline=114.0,
            carb_sensitivity=0.65,
            activity_clearance_factor=0.050,
            baseline_activity_level="Active (6000-8000 steps/day, cycling)",
            average_sleep_hours=7.8,
            target_range_lower=70.0,
            target_range_upper=180.0,
            clinical_notes=(
                "Recently diagnosed early T2D. Excellent endogenous beta-cell reserve. "
                "Highly responsive to moderate cardiovascular activity. Fasting glucose within "
                "target; excursions occur primarily with irregular meal schedules or high GI foods."
            ),
        ),
    }
    return profiles


def get_patient_profile(patient_id: str) -> PatientProfile:
    """Retrieve profile by ID, defaulting to PT-1001 if not found."""
    profiles = get_synthetic_patient_profiles()
    if patient_id not in profiles:
        return profiles["PT-1001"]
    return profiles[patient_id]
