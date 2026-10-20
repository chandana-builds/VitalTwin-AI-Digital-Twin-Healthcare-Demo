"""
Unit tests for Synthetic Patient Profile Generator
VitalTwin AI - Happiest Health Digital Twin Challenge 2026
"""

import pytest
from src.data.patient_generator import (
    get_synthetic_patient_profiles,
    get_patient_profile,
    PatientProfile,
)


def test_synthetic_patient_profiles_count():
    """Verify at least 3 distinct synthetic patient profiles exist."""
    profiles = get_synthetic_patient_profiles()
    assert len(profiles) >= 3
    assert "PT-1001" in profiles
    assert "PT-1002" in profiles
    assert "PT-1003" in profiles


def test_patient_profiles_clinical_attributes():
    """Verify that all profiles contain coherent clinical attributes."""
    profiles = get_synthetic_patient_profiles()
    for pid, p in profiles.items():
        assert isinstance(p, PatientProfile)
        assert p.patient_id == pid
        assert 18 <= p.age <= 100
        assert 15.0 <= p.bmi <= 55.0
        assert 4.0 <= p.hba1c <= 15.0
        assert p.on_insulin in (0, 1)
        assert 60.0 <= p.fasting_glucose_baseline <= 300.0
        assert len(p.medication_status) > 0
        assert len(p.clinical_notes) > 0


def test_patient_profile_fallback():
    """Verify fallback behavior for invalid or non-existent patient IDs."""
    fallback_profile = get_patient_profile("INVALID-9999")
    assert fallback_profile is not None
    assert fallback_profile.patient_id == "PT-1001"


def test_patient_to_dict():
    """Verify dataclass dictionary export."""
    p = get_patient_profile("PT-1001")
    d = p.to_dict()
    assert isinstance(d, dict)
    assert d["patient_id"] == "PT-1001"
    assert d["age"] == 58
    assert "hba1c" in d
