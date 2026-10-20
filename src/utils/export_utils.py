"""
Data Export Utilities for VitalTwin AI
Prepares CSV exports of synthetic patient observations, engineered features,
and prediction records.
"""

from typing import Dict, Any, Optional
import io
import pandas as pd


def export_observations_csv(df: pd.DataFrame) -> str:
    """Exports time series observations to CSV string."""
    export_df = df.copy()
    if "timestamp" in export_df.columns:
        export_df["timestamp"] = pd.to_datetime(export_df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    return export_df.to_csv(index=False)


def export_predictions_csv(history_df: pd.DataFrame, prediction_records: list) -> str:
    """Combines latest predictions and features into an exportable CSV string."""
    if not prediction_records:
        return export_observations_csv(history_df)
    
    pred_df = pd.DataFrame(prediction_records)
    return pred_df.to_csv(index=False)
