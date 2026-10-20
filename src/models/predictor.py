"""
Machine Learning Prediction Engine
VitalTwin AI - Happiest Health Digital Twin Challenge 2026

Implements a scikit-learn Random Forest Classifier to forecast the 120-minute
probability of crossing a configurable high-glucose threshold (default 180 mg/dL).
Evaluates strictly on held-out test data with genuine performance metrics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from config.settings import (
    DEFAULT_HIGH_GLUCOSE_THRESHOLD,
    RISK_THRESHOLD_LOW,
    RISK_THRESHOLD_HIGH,
    FEATURE_COLUMNS,
)
from src.data.patient_generator import get_synthetic_patient_profiles
from src.data.sensor_generator import generate_patient_timeseries
from src.features.feature_engineering import prepare_dataset_for_modeling


@dataclass
class ModelEvaluationResults:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float]
    confusion_matrix: List[List[int]]
    n_train: int
    n_test: int
    test_prevalence: float
    feature_importances: Dict[str, float]
    threshold_mg_dl: float
    evaluation_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "roc_auc": round(self.roc_auc, 4) if self.roc_auc is not None else None,
            "confusion_matrix": self.confusion_matrix,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "test_prevalence": round(self.test_prevalence, 4),
            "threshold_mg_dl": self.threshold_mg_dl,
            "evaluation_notes": self.evaluation_notes,
        }


class GlucoseRiskPredictor:
    """
    Random Forest-based predictive engine for proactive glycemic excursion forecasting.
    Provides calibrated risk probabilities, risk tier classification, and explainable
    feature contributions.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 8,
        random_state: int = 42,
    ):
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=6,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        self.is_trained = False
        self.evaluation_results: Optional[ModelEvaluationResults] = None
        self.feature_columns = list(FEATURE_COLUMNS)

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> None:
        """Trains the Random Forest model on feature matrix X and target y."""
        X_clean = X_train[self.feature_columns].fillna(0.0)
        self.model.fit(X_clean, y_train)
        self.is_trained = True

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        n_train: int,
        threshold: float = DEFAULT_HIGH_GLUCOSE_THRESHOLD,
    ) -> ModelEvaluationResults:
        """
        Computes genuine evaluation metrics on held-out test data.
        Never fabricates or hard-codes metrics.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation.")

        X_clean = X_test[self.feature_columns].fillna(0.0)
        y_pred = self.model.predict(X_clean)
        y_proba = self.model.predict_proba(X_clean)[:, 1]

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))

        # ROC-AUC requires at least one positive and one negative sample in y_test
        roc = None
        if len(np.unique(y_test)) > 1:
            try:
                roc = float(roc_auc_score(y_test, y_proba))
            except Exception:
                roc = None

        cm = confusion_matrix(y_test, y_pred).tolist()
        prevalence = float(np.mean(y_test))

        # Ranked feature importances
        raw_importances = self.model.feature_importances_
        fi_dict = {
            col: float(imp)
            for col, imp in sorted(
                zip(self.feature_columns, raw_importances),
                key=lambda item: item[1],
                reverse=True,
            )
        }

        eval_res = ModelEvaluationResults(
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1=f1,
            roc_auc=roc,
            confusion_matrix=cm,
            n_train=n_train,
            n_test=len(y_test),
            test_prevalence=prevalence,
            feature_importances=fi_dict,
            threshold_mg_dl=threshold,
            evaluation_notes=(
                "Evaluation performed on chronologically separated, held-out synthetic test records "
                "with an enforced 2-hour purge gap to eliminate temporal window leakage."
            ),
        )
        self.evaluation_results = eval_res
        return eval_res

    def predict_risk(
        self,
        feature_row: pd.DataFrame,
    ) -> Tuple[float, str, List[Dict[str, Any]]]:
        """
        Infers 2-hour high-glucose probability for a single feature row.
        Returns:
            - probability: float in [0.0, 1.0]
            - risk_tier: 'Low Risk', 'Moderate Risk', or 'High Risk'
            - top_factors: List of top 5 influential features with their values and weights
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet.")

        X_clean = feature_row[self.feature_columns].fillna(0.0)
        proba = float(self.model.predict_proba(X_clean)[0, 1])

        # Categorize into risk tiers
        if proba < RISK_THRESHOLD_LOW:
            tier = "Low Risk"
        elif proba < RISK_THRESHOLD_HIGH:
            tier = "Moderate Risk"
        else:
            tier = "High Risk"

        # Explainability: feature values weighted by model importance
        top_factors = []
        if self.evaluation_results and self.evaluation_results.feature_importances:
            for feat, importance in list(self.evaluation_results.feature_importances.items())[:6]:
                val = float(X_clean[feat].iloc[0])
                top_factors.append({
                    "feature": feat,
                    "value": round(val, 2),
                    "importance": round(importance, 4),
                    "description": self._describe_feature(feat, val),
                })

        return proba, tier, top_factors

    @staticmethod
    def _describe_feature(feature_name: str, value: float) -> str:
        """Translates technical feature names and values into clinical contextual descriptions."""
        descriptions = {
            "glucose_current": f"Current CGM Glucose: {value:.1f} mg/dL",
            "glucose_trend_15m": f"15-min Glucose Velocity: {value:+.2f} mg/dL/min",
            "glucose_trend_30m": f"30-min Glucose Velocity: {value:+.2f} mg/dL/min",
            "glucose_mean_60m": f"Past 1-hr Mean Glucose: {value:.1f} mg/dL",
            "glucose_std_60m": f"Glycemic Variability (SD 1h): {value:.1f} mg/dL",
            "carbs_last_60m": f"Carbohydrate Influx (1h): {value:.0f} g",
            "carbs_last_120m": f"Carbohydrate Influx (2h): {value:.0f} g",
            "mins_since_last_meal": f"Time Since Prior Meal: {value:.0f} min",
            "activity_steps_last_60m": f"Physical Activity (1h): {value:.0f} steps",
            "hba1c": f"Baseline Glycated Hemoglobin: {value:.1f}%",
            "diabetes_duration_years": f"Duration of T2D: {value:.1f} years",
            "on_insulin": f"Basal Insulin Regimen: {'Yes' if value > 0.5 else 'No'}",
        }
        return descriptions.get(feature_name, f"{feature_name}: {value:.2f}")


def train_and_evaluate_model(
    threshold: float = DEFAULT_HIGH_GLUCOSE_THRESHOLD,
    random_seed: int = 42,
) -> Tuple[GlucoseRiskPredictor, ModelEvaluationResults]:
    """
    Generates multi-patient synthetic historical datasets, constructs features & labels,
    splits with a purge gap to prevent leakage, trains the Random Forest, and returns
    the trained predictor and its verified evaluation metrics.
    """
    profiles = get_synthetic_patient_profiles()

    all_dfs = []
    # Generate 4 days of simulation for each patient profile (4 * 288 = 1152 steps per patient)
    for idx, (pid, prof) in enumerate(profiles.items()):
        df = generate_patient_timeseries(
            patient_id=pid,
            days=4,
            random_seed=random_seed + idx * 10,
        )
        df_feat = prepare_dataset_for_modeling(df, profile=prof, threshold=threshold)
        all_dfs.append(df_feat)

    # Combine into unified dataset while respecting chronological train-test split
    # For robust evaluation without cross-time leakage, we train on first 75% of time series
    # and test on the final 25% across profiles.
    train_X_list, train_y_list = [], []
    test_X_list, test_y_list = [], []

    for X_sub, y_sub in all_dfs:
        n = len(X_sub)
        split_idx = int(n * 0.75)
        # Enforce a 24-step (2-hour) purge gap between train and test to guarantee zero overlap
        gap = 24
        
        X_train_part = X_sub.iloc[: split_idx - gap]
        y_train_part = y_sub.iloc[: split_idx - gap]
        
        X_test_part = X_sub.iloc[split_idx :]
        y_test_part = y_sub.iloc[split_idx :]
        
        train_X_list.append(X_train_part)
        train_y_list.append(y_train_part)
        test_X_list.append(X_test_part)
        test_y_list.append(y_test_part)

    X_train = pd.concat(train_X_list, ignore_index=True)
    y_train = pd.concat(train_y_list, ignore_index=True)
    X_test = pd.concat(test_X_list, ignore_index=True)
    y_test = pd.concat(test_y_list, ignore_index=True)

    predictor = GlucoseRiskPredictor(n_estimators=100, max_depth=7, random_state=random_seed)
    predictor.train(X_train, y_train)
    eval_results = predictor.evaluate(X_test, y_test, n_train=len(X_train), threshold=threshold)

    return predictor, eval_results
