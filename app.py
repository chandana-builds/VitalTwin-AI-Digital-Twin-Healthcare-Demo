"""
VitalTwin AI - An Explainable Digital Twin for Predictive Healthcare
Challenge: Happiest Health Digital Twin Challenge 2026
Research Proof of Concept — Synthetic Data Only — Not a Medical Device
"""

import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Configure page layout and professional favicon/title
st.set_page_config(
    page_title="VitalTwin AI — Predictive Healthcare Digital Twin",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config.settings import (
    PROJECT_NAME,
    TAGLINE,
    CHALLENGE_NAME,
    HEALTHCARE_FOCUS,
    DEFAULT_HIGH_GLUCOSE_THRESHOLD,
    RISK_THRESHOLD_LOW,
    RISK_THRESHOLD_HIGH,
    THEME_COLORS,
    DISCLAIMER_TEXT,
    FEATURE_COLUMNS,
)
from src.data.patient_generator import get_synthetic_patient_profiles, get_patient_profile
from src.simulation.twin_engine import DigitalTwinEngine, WhatIfScenarioSimulator
from src.models.predictor import train_and_evaluate_model, GlucoseRiskPredictor
from src.utils.export_utils import export_observations_csv, export_predictions_csv
from src.utils.doc_generator import generate_all_documents


# ---------------------------------------------------------
# Custom Professional CSS (Navy, Teal, Slate palette)
# ---------------------------------------------------------
def inject_custom_css():
    st.markdown(
        """
        <style>
        /* Main page background */
        .stApp {
            background-color: #0A0F1D;
            color: #E2E8F0;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #0F172A;
            border-right: 1px solid #1E293B;
        }

        /* Header typography */
        h1, h2, h3, h4 {
            color: #F8FAFC !important;
            font-weight: 700;
            letter-spacing: -0.02em;
        }

        /* Metric Cards */
        .metric-card {
            background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 16px 20px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
            margin-bottom: 12px;
        }
        .metric-label {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #94A3B8;
            margin-bottom: 4px;
            font-weight: 600;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #F8FAFC;
            line-height: 1.2;
        }
        .metric-sub {
            font-size: 0.82rem;
            color: #14B8A6;
            margin-top: 4px;
            display: flex;
            align-items: center;
            gap: 4px;
        }

        /* Alert Disclaimers */
        .disclaimer-banner {
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid #0D9488;
            border-left: 4px solid #14B8A6;
            border-radius: 6px;
            padding: 10px 16px;
            margin-bottom: 18px;
            font-size: 0.82rem;
            color: #CBD5E1;
        }

        /* Risk Badges */
        .badge-low {
            background-color: rgba(16, 185, 129, 0.15);
            color: #34D399;
            border: 1px solid #10B981;
            padding: 4px 10px;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.85rem;
            display: inline-block;
        }
        .badge-mod {
            background-color: rgba(245, 158, 11, 0.15);
            color: #FBBF24;
            border: 1px solid #F59E0B;
            padding: 4px 10px;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.85rem;
            display: inline-block;
        }
        .badge-high {
            background-color: rgba(239, 68, 68, 0.15);
            color: #F87171;
            border: 1px solid #EF4444;
            padding: 4px 10px;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.85rem;
            display: inline-block;
        }

        /* Button styling */
        .stButton>button {
            border-radius: 6px;
            font-weight: 600;
            transition: all 0.2s ease;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# Session State Management & Model Caching
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading pre-trained VitalTwin AI model...")
def get_cached_model_and_eval(threshold: float = DEFAULT_HIGH_GLUCOSE_THRESHOLD):
    cache_path = os.path.join(os.path.dirname(__file__), "data", "model_cache.pkl")
    if threshold == DEFAULT_HIGH_GLUCOSE_THRESHOLD and os.path.exists(cache_path):
        import pickle
        with open(cache_path, "rb") as f:
            predictor, eval_results = pickle.load(f)
    else:
        predictor, eval_results = train_and_evaluate_model(threshold=threshold, random_seed=42)
    
    # Generate presentation and architecture PDFs if missing
    try:
        generate_all_documents(eval_results)
    except Exception as e:
        pass
    return predictor, eval_results


def initialize_session_state(predictor: GlucoseRiskPredictor):
    if "selected_patient_id" not in st.session_state:
        st.session_state["selected_patient_id"] = "PT-1001"

    if "twin" not in st.session_state:
        st.session_state["twin"] = DigitalTwinEngine(
            patient_id=st.session_state["selected_patient_id"],
            predictor=predictor,
            initial_history_hours=24,
            random_seed=42,
        )

    if "prediction_log" not in st.session_state:
        st.session_state["prediction_log"] = []
        if st.session_state["twin"].latest_prediction:
            st.session_state["prediction_log"].append(st.session_state["twin"].latest_prediction)

    if "high_glucose_threshold" not in st.session_state:
        st.session_state["high_glucose_threshold"] = DEFAULT_HIGH_GLUCOSE_THRESHOLD


# ---------------------------------------------------------
# Main App Execution
# ---------------------------------------------------------
def main():
    inject_custom_css()

    # Load / Cache Model
    predictor, eval_results = get_cached_model_and_eval(threshold=DEFAULT_HIGH_GLUCOSE_THRESHOLD)
    initialize_session_state(predictor)

    twin: DigitalTwinEngine = st.session_state["twin"]

    # ---------------------------------------------------------
    # Sidebar: Patient Selection & Simulation Controls
    # ---------------------------------------------------------
    with st.sidebar:
        st.markdown(f"### 🧬 {PROJECT_NAME}")
        st.caption(f"{TAGLINE} • **{CHALLENGE_NAME}**")
        st.markdown("---")

        # Patient Profile Selector
        profiles = get_synthetic_patient_profiles()
        patient_options = {
            f"{p.patient_id}: {p.name.split(' (')[0]} ({p.baseline_activity_level.split(' ')[0]}, HbA1c {p.hba1c}%)": pid
            for pid, p in profiles.items()
        }

        current_label = [k for k, v in patient_options.items() if v == st.session_state["selected_patient_id"]][0]
        selected_label = st.selectbox(
            "Select Virtual Patient Profile",
            options=list(patient_options.keys()),
            index=list(patient_options.keys()).index(current_label),
            help="Switch between synthetic patient phenotypes with distinct insulin resistance and glycemic variability.",
        )
        new_pid = patient_options[selected_label]

        if new_pid != st.session_state["selected_patient_id"]:
            st.session_state["selected_patient_id"] = new_pid
            twin.reset(patient_id=new_pid, initial_history_hours=24)
            st.session_state["prediction_log"] = [twin.latest_prediction]
            st.rerun()

        # Display Selected Profile Snapshot
        prof = twin.profile
        st.markdown(
            f"""
            <div style="background:#1E293B; border-radius:6px; padding:10px; font-size:0.8rem; border-left:3px solid #14B8A6; margin-bottom:12px;">
                <b>ID:</b> {prof.patient_id} | <b>Age:</b> {prof.age} | <b>BMI:</b> {prof.bmi}<br/>
                <b>HbA1c:</b> {prof.hba1c}% | <b>T2D Duration:</b> {prof.diabetes_duration_years} yrs<br/>
                <b>Rx:</b> {prof.medication_status}<br/>
                <b>Insulin Regimen:</b> {'Yes (Basal)' if prof.on_insulin else 'None (Oral/Diet)'}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### ⏱️ Simulation Engine")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            if st.button("▶ Step +5 Min", use_container_width=True, help="Advance simulation by one 5-minute CGM step"):
                obs = twin.step_forward(minutes=5)
                st.session_state["prediction_log"].append(twin.latest_prediction)
                st.rerun()
        with col_s2:
            if st.button("⏩ Step +15 Min", use_container_width=True, help="Advance simulation by 15 minutes (3 steps)"):
                obs = twin.step_forward(minutes=15)
                st.session_state["prediction_log"].append(twin.latest_prediction)
                st.rerun()

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            if st.button("🔄 Reset Twin", use_container_width=True, help="Reset patient history to 24h burn-in"):
                twin.reset(patient_id=st.session_state["selected_patient_id"], initial_history_hours=24)
                st.session_state["prediction_log"] = [twin.latest_prediction]
                st.rerun()
        with col_r2:
            sim_status_badge = "🟢 ACTIVE" if twin.is_running else "⏸️ PAUSED"
            st.markdown(f"<div style='text-align:center; padding-top:6px; font-size:0.8rem;'>{sim_status_badge}</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 🍽️ Log Meal Event (Live)")
        meal_carbs_input = st.slider("Carbohydrate Intake (g)", min_value=0, max_value=120, value=50, step=5)
        if st.button("🍴 Ingest Meal Now", use_container_width=True):
            obs = twin.step_forward(minutes=5, meal_carbs=float(meal_carbs_input))
            st.session_state["prediction_log"].append(twin.latest_prediction)
            st.success(f"Logged {meal_carbs_input}g carbohydrate meal event!")
            st.rerun()

        st.markdown("---")
        st.markdown("#### ⚙️ Threshold Settings")
        thresh = st.slider(
            "High-Glucose Threshold (mg/dL)",
            min_value=140,
            max_value=240,
            value=int(st.session_state["high_glucose_threshold"]),
            step=5,
            help="Clinical threshold defining hyperglycemia (ADA default: 180 mg/dL)",
        )
        if thresh != st.session_state["high_glucose_threshold"]:
            st.session_state["high_glucose_threshold"] = float(thresh)
            st.rerun()

    # ---------------------------------------------------------
    # Main Header & Disclaimer Banner
    # ---------------------------------------------------------
    st.markdown(
        f"""
        <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:10px;">
            <div>
                <h1 style="margin:0; font-size:2.2rem; display:flex; align-items:center; gap:10px;">
                    <span>VitalTwin AI</span>
                    <span style="font-size:0.85rem; font-weight:500; background:#0D9488; color:#FFFFFF; padding:3px 10px; border-radius:12px;">RESEARCH PROTOTYPE</span>
                </h1>
                <p style="margin:4px 0 0 0; color:#94A3B8; font-size:1.0rem;">
                    {TAGLINE} • <b>{CHALLENGE_NAME}</b>
                </p>
            </div>
            <div style="text-align:right;">
                <div style="font-size:0.8rem; color:#94A3B8;">SIMULATION CLOCK</div>
                <div style="font-size:1.15rem; font-weight:700; color:#14B8A6;">
                    {twin.current_time.strftime('%b %d, %Y • %H:%M')}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="disclaimer-banner">
            🛡️ <b>NON-CLINICAL RESEARCH DISCLAIMER:</b> {DISCLAIMER_TEXT}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------
    # Row 1: Key Metric Ribbon Cards
    # ---------------------------------------------------------
    summary = twin.get_summary_metrics()
    pred = twin.latest_prediction or {"probability": 0.0, "risk_tier": "Low Risk", "top_factors": []}
    proba = pred.get("probability", 0.0)
    tier = pred.get("risk_tier", "Low Risk")

    badge_class = "badge-low" if proba < RISK_THRESHOLD_LOW else ("badge-mod" if proba < RISK_THRESHOLD_HIGH else "badge-high")
    trend_val = summary["trend_rate"]
    trend_symbol = "↗" if trend_val > 0.3 else ("↘" if trend_val < -0.3 else "→")
    trend_color = "#F87171" if trend_val > 0.3 else ("#34D399" if trend_val < -0.3 else "#94A3B8")

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Current Interstitial Glucose</div>
                <div class="metric-value">{summary['current_glucose']:.1f} <span style="font-size:1.0rem; font-weight:400; color:#94A3B8;">mg/dL</span></div>
                <div class="metric-sub" style="color:{trend_color};">
                    <span>{trend_symbol}</span> <span>{trend_val:+.2f} mg/dL/min</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Predicted 2-Hr Excursion Risk</div>
                <div class="metric-value">{proba*100:.1f}%</div>
                <div class="metric-sub">
                    <span class="{badge_class}">{tier}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Time in Target (70-180 mg/dL)</div>
                <div class="metric-value">{summary['tir_pct']:.1f}%</div>
                <div class="metric-sub">
                    <span>Target: &gt;70% (ADA)</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Glycemic Variability (CV %)</div>
                <div class="metric-value">{summary['cv_pct']:.1f}%</div>
                <div class="metric-sub">
                    <span>1-Hr SD: {summary['std_glucose']:.1f} mg/dL</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Wearable Physiological State</div>
                <div class="metric-value">{twin.current_heart_rate:.0f} <span style="font-size:1.0rem; font-weight:400; color:#94A3B8;">BPM</span></div>
                <div class="metric-sub">
                    <span>Steps (5m): {twin.current_steps}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---------------------------------------------------------
    # Row 2: Dynamic Multi-Modal Plotly Charts
    # ---------------------------------------------------------
    st.markdown("### 📈 Continuous Glucose & Physiological Dynamics")

    history_df = twin.history_df.iloc[-288:].copy()  # Display past 24 hours
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Continuous Glucose Monitoring (CGM) & 120-Min Forecast Window", "Physical Activity (Steps) & Heart Rate (BPM)"),
        row_heights=[0.70, 0.30],
    )

    # Target in-range band (70 - 180 mg/dL)
    threshold_val = st.session_state["high_glucose_threshold"]
    fig.add_hrect(
        y0=70.0,
        y1=threshold_val,
        fillcolor="rgba(20, 184, 166, 0.08)",
        line_width=0,
        row=1,
        col=1,
    )

    # Threshold horizontal reference line
    fig.add_hline(
        y=threshold_val,
        line_dash="dash",
        line_color="#EF4444",
        line_width=1.5,
        annotation_text=f"High Threshold: {threshold_val:.0f} mg/dL",
        annotation_position="top left",
        annotation_font_color="#EF4444",
        row=1,
        col=1,
    )

    # Glucose trace
    fig.add_trace(
        go.Scatter(
            x=history_df["timestamp"],
            y=history_df["glucose"],
            name="CGM Interstitial Glucose",
            mode="lines",
            line=dict(color="#2DD4BF", width=2.5),
            hovertemplate="%{x|%H:%M}<br>Glucose: <b>%{y:.1f} mg/dL</b><extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Meal markers
    meal_rows = history_df[history_df["meal_carbs"] > 0]
    if len(meal_rows) > 0:
        fig.add_trace(
            go.Scatter(
                x=meal_rows["timestamp"],
                y=meal_rows["glucose"],
                name="Meal Intake",
                mode="markers+text",
                marker=dict(symbol="triangle-up", size=11, color="#F59E0B"),
                text=[f"{int(c)}g" for c in meal_rows["meal_carbs"]],
                textposition="top center",
                textfont=dict(color="#F59E0B", size=10),
                hovertemplate="%{x|%H:%M}<br>Meal: <b>%{text}</b> carbs<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # 120-minute Forward Forecast Projection Cone
    last_t = history_df["timestamp"].iloc[-1]
    last_g = history_df["glucose"].iloc[-1]
    forecast_dt = last_t + timedelta(minutes=120)

    # Estimated projection endpoint based on velocity and probability
    projected_delta = (pred.get("probability", 0.0) - 0.5) * 80.0 + (summary["trend_rate"] * 60.0)
    projected_g = max(50.0, min(350.0, last_g + projected_delta))

    fig.add_trace(
        go.Scatter(
            x=[last_t, forecast_dt],
            y=[last_g, projected_g],
            name="2-Hr Risk Trajectory Projection",
            mode="lines+markers",
            line=dict(color="#F43F5E", width=2, dash="dot"),
            marker=dict(size=6, color="#F43F5E"),
            hovertemplate="Forecast Horizon: %{x|%H:%M}<br>Projected Est: <b>%{y:.1f} mg/dL</b><extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Subplot 2: Heart rate and Steps
    fig.add_trace(
        go.Bar(
            x=history_df["timestamp"],
            y=history_df["steps"],
            name="Activity (Steps)",
            marker_color="rgba(148, 163, 184, 0.45)",
            hovertemplate="%{x|%H:%M}<br>Steps: <b>%{y}</b><extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=history_df["timestamp"],
            y=history_df["heart_rate"],
            name="Heart Rate (BPM)",
            mode="lines",
            line=dict(color="#FB7185", width=1.8),
            yaxis="y3",
            hovertemplate="%{x|%H:%M}<br>Heart Rate: <b>%{y:.1f} BPM</b><extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        margin=dict(l=40, r=40, t=30, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        height=480,
    )
    fig.update_yaxes(title_text="Glucose (mg/dL)", row=1, col=1, gridcolor="#1E293B")
    fig.update_yaxes(title_text="Steps", row=2, col=1, gridcolor="#1E293B")
    fig.update_xaxes(gridcolor="#1E293B")

    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------
    # Row 3: Explainable AI & What-If Scenario Sandbox
    # ---------------------------------------------------------
    c_xai, c_whatif = st.columns([1, 1])

    with c_xai:
        st.markdown("### 🔍 Model Explainability & Risk Drivers")
        st.caption("Interpretable feature contributions influencing the 2-hour high-glucose prediction.")

        top_factors = pred.get("top_factors", [])
        if top_factors:
            fi_df = pd.DataFrame(top_factors)
            fig_fi = go.Figure(
                go.Bar(
                    x=fi_df["importance"],
                    y=fi_df["feature"],
                    orientation="h",
                    marker=dict(color="#0D9488"),
                    hovertemplate="Feature: %{y}<br>Importance: <b>%{x:.3f}</b><extra></extra>",
                )
            )
            fig_fi.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0F172A",
                plot_bgcolor="#0F172A",
                margin=dict(l=10, r=10, t=10, b=10),
                height=220,
                xaxis=dict(title="Ensemble Gini Feature Importance", gridcolor="#1E293B"),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_fi, use_container_width=True)

            # Clinical Context Descriptions
            st.markdown("##### Clinical Factor Breakdown:")
            for f in top_factors[:4]:
                st.markdown(
                    f"• **{f['description']}** *(Global Weight: {f['importance']:.3f})*"
                )
        else:
            st.info("Feature importance data is synchronizing...")

    with c_whatif:
        st.markdown("### 🧪 Interactive What-If Simulator")
        st.caption("Hypothetical lifestyle adjustments evaluated in an isolated sandbox state.")

        whatif_sim = WhatIfScenarioSimulator(predictor)
        current_features = pred.get("features", {})

        col_wi1, col_wi2 = st.columns(2)
        with col_wi1:
            wi_carbs = st.slider("Hypothetical Meal Carbs (g)", min_value=0, max_value=120, value=30, step=5)
            wi_sleep = st.slider("Hypothetical Prior Sleep (hrs)", min_value=4.0, max_value=10.0, value=7.0, step=0.5)
        with col_wi2:
            wi_steps = st.slider("Hypothetical Post-Meal Walking (steps)", min_value=0, max_value=600, value=300, step=50)
            wi_night = st.selectbox("Time of Day Context", options=["Current Daytime", "Nighttime (Circadian)"])
            is_night_flag = 1 if wi_night == "Nighttime (Circadian)" else 0

        # Execute scenario
        if current_features:
            scen_res = whatif_sim.simulate_scenario(
                baseline_features=current_features,
                hypothetical_carbs=float(wi_carbs),
                hypothetical_activity_steps=int(wi_steps),
                hypothetical_sleep_hours=float(wi_sleep),
                is_nighttime_override=is_night_flag,
            )

            b_prob = scen_res["baseline_probability"]
            s_prob = scen_res["scenario_probability"]
            delta = scen_res["delta_percentage_points"]

            scen_color = "#34D399" if delta < -3.0 else ("#F87171" if delta > 3.0 else "#94A3B8")
            delta_sign = "+" if delta > 0 else ""

            st.markdown(
                f"""
                <div style="background:#1E293B; border:1px solid #334155; border-radius:8px; padding:12px; margin-top:8px;">
                    <div style="display:flex; justify-content:space-around; text-align:center;">
                        <div>
                            <div style="font-size:0.75rem; color:#94A3B8;">BASELINE RISK</div>
                            <div style="font-size:1.3rem; font-weight:700;">{b_prob*100:.1f}%</div>
                            <span class="{badge_class}">{scen_res['baseline_risk_tier']}</span>
                        </div>
                        <div style="display:flex; align-items:center; font-size:1.4rem; color:#64748B;">➔</div>
                        <div>
                            <div style="font-size:0.75rem; color:#94A3B8;">SCENARIO RISK</div>
                            <div style="font-size:1.3rem; font-weight:700;">{s_prob*100:.1f}%</div>
                            <span style="font-size:0.85rem; font-weight:600; color:{scen_color};">{delta_sign}{delta:.1f}% pts</span>
                        </div>
                    </div>
                    <div style="margin-top:10px; font-size:0.8rem; color:#CBD5E1; line-height:1.4; border-top:1px solid #334155; padding-top:8px;">
                        💡 <b>Physiological Insight:</b> {scen_res['scenario_explanation']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ---------------------------------------------------------
    # Row 4: Transparency, Patient EHR & CSV Export Tabs
    # ---------------------------------------------------------
    st.markdown("### 📑 Data Transparency, Clinical Profile & Model Verification")

    tab_ehr, tab_eval, tab_feat, tab_export, tab_docs = st.tabs([
        "👤 Patient EHR Profile",
        "📊 Model Evaluation & Metrics",
        "⚙️ Engineered Features Inspector",
        "💾 Data Export (CSV)",
        "📄 Challenge Deliverables & PDFs",
    ])

    with tab_ehr:
        st.markdown(f"#### Electronic Health Record — {prof.name}")
        c_ehr1, c_ehr2 = st.columns(2)
        with c_ehr1:
            st.markdown(
                f"""
                | Clinical Parameter | Baseline Value | Reference Target |
                | :--- | :--- | :--- |
                | **Patient ID** | `{prof.patient_id}` | Synthetic Ident. |
                | **Age / Sex** | {prof.age} years / {prof.sex} | Adult T2D Cohort |
                | **Body Mass Index (BMI)** | {prof.bmi:.1f} kg/m² | 18.5 - 24.9 kg/m² |
                | **Blood Pressure** | {prof.blood_pressure} | < 130/80 mmHg |
                | **Diabetes Duration** | {prof.diabetes_duration_years} years | Clinical Timeline |
                | **Glycated Hemoglobin (HbA1c)** | **{prof.hba1c}%** | < 7.0% (ADA) |
                """
            )
        with c_ehr2:
            st.markdown(
                f"""
                | Attribute | Configuration | Clinical Implication |
                | :--- | :--- | :--- |
                | **Medication Regimen** | {prof.medication_status} | Primary Pharmacotherapy |
                | **On Basal Insulin** | {'Yes (Class 1)' if prof.on_insulin else 'No'} | Beta-cell Exhaustion Marker |
                | **Fasting Baseline Glucose** | {prof.fasting_glucose_baseline} mg/dL | Fasting Setpoint |
                | **Baseline Physical Activity** | {prof.baseline_activity_level} | Daily Exercise Clearance |
                | **Mean Sleep Duration** | {prof.average_sleep_hours} hours | Circadian Homeostasis |
                | **Target Glucose Range** | {prof.target_range_lower:.0f} - {prof.target_range_upper:.0f} mg/dL | Ambulatory TIR Ceiling |
                """
            )
        st.markdown(f"**Clinical Observational Notes:** {prof.clinical_notes}")

    with tab_eval:
        st.markdown("#### Authentic Held-Out Evaluation Performance")
        st.caption(
            "Evaluation metrics genuinely calculated on out-of-sample chronological test partitions with an enforced "
            "2-hour purge gap to eliminate time series window leakage."
        )

        ev_c1, ev_c2, ev_c3, ev_c4, ev_c5 = st.columns(5)
        ev_c1.metric("Classification Accuracy", f"{eval_results.accuracy*100:.1f}%")
        ev_c2.metric("Precision (Hyperglycemia)", f"{eval_results.precision*100:.1f}%")
        ev_c3.metric("Recall (Sensitivity)", f"{eval_results.recall*100:.1f}%")
        ev_c4.metric("F1-Score", f"{eval_results.f1*100:.1f}%")
        if eval_results.roc_auc is not None:
            ev_c5.metric("ROC-AUC Score", f"{eval_results.roc_auc:.3f}")
        else:
            ev_c5.metric("ROC-AUC Score", "N/A (Single class)")

        st.markdown("##### Confusion Matrix (Held-Out Test Sequence):")
        cm = eval_results.confusion_matrix
        cm_df = pd.DataFrame(
            cm,
            index=["Actual Normal/In-Range (<180)", "Actual High-Glucose (≥180)"],
            columns=["Predicted Normal", "Predicted High"],
        )
        st.dataframe(cm_df, use_container_width=True)

        st.markdown(
            f"""
            - **Training Sequence Windows:** {eval_results.n_train} observations
            - **Held-Out Test Windows:** {eval_results.n_test} observations
            - **Test Event Prevalence:** {eval_results.test_prevalence*100:.1f}%
            - **Audit Protocol:** {eval_results.evaluation_notes}
            """
        )

    with tab_feat:
        st.markdown("#### Live Engineered Causal Features at Observation Index $t$")
        st.caption("All features are evaluated strictly on history $\\le t$. Zero future leakage.")
        if current_features:
            feat_table = pd.DataFrame([current_features]).T.reset_index()
            feat_table.columns = ["Feature Name", "Engineered Value at Timestamp t"]
            st.dataframe(feat_table, use_container_width=True, height=350)

    with tab_export:
        st.markdown("#### Export Synthetic Observations and Model Logs")
        st.caption("Download standardized CSV datasets for independent external auditing and benchmark verification.")

        c_exp1, c_exp2 = st.columns(2)
        with c_exp1:
            csv_obs = export_observations_csv(twin.history_df)
            st.download_button(
                label="📥 Download Continuous Sensor Observations (CSV)",
                data=csv_obs,
                file_name=f"vitaltwin_{prof.patient_id}_sensor_stream.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with c_exp2:
            csv_preds = export_predictions_csv(twin.history_df, st.session_state.get("prediction_log", []))
            st.download_button(
                label="📥 Download Twin Prediction Log (CSV)",
                data=csv_preds,
                file_name=f"vitaltwin_{prof.patient_id}_predictions_log.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with tab_docs:
        st.markdown("#### Generated Challenge Documentation & Presentations")
        st.caption("Access the official architecture and jury presentation deliverables.")

        d_col1, d_col2, d_col3 = st.columns(3)
        with d_col1:
            arch_path = "docs/architecture.pdf"
            if os.path.exists(arch_path):
                with open(arch_path, "rb") as f:
                    st.download_button(
                        "📄 Download Architecture PDF",
                        data=f.read(),
                        file_name="VitalTwin_AI_Architecture.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
            else:
                st.warning("docs/architecture.pdf not found")

        with d_col2:
            pres_pdf = "docs/presentation.pdf"
            if os.path.exists(pres_pdf):
                with open(pres_pdf, "rb") as f:
                    st.download_button(
                        "📊 Download Pitch Deck (PDF)",
                        data=f.read(),
                        file_name="VitalTwin_AI_Presentation.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
            else:
                st.warning("docs/presentation.pdf not found")

        with d_col3:
            pres_pptx = "docs/presentation.pptx"
            if os.path.exists(pres_pptx):
                with open(pres_pptx, "rb") as f:
                    st.download_button(
                        "📽️ Download Pitch Deck (PPTX)",
                        data=f.read(),
                        file_name="VitalTwin_AI_Presentation.pptx",
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                        use_container_width=True,
                    )
            else:
                st.warning("docs/presentation.pptx not found")


if __name__ == "__main__":
    main()
