"""
Streamlit app for the Sepsis Detection project.
Two modes: upload a patient's hourly CSV for a trend view, or enter
a single hour's vitals manually for a quick point prediction.
"""

import os
import sys

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_loader import load_config
from predict import load_model_bundle, predict


st.set_page_config(page_title="Sepsis Risk Predictor", layout="wide")

st.title("Sepsis Risk Predictor")
st.caption(
    "Trained on the PhysioNet Sepsis dataset (XGBoost, val AUPRC ≈ 0.20, "
    "test AUPRC ≈ 0.24). This is a course project demo, not a clinical tool — "
    "at the default threshold, roughly 4 in 5 positive alerts are false positives. "
    "Treat probability as a rough risk indicator, not a diagnosis."
)

config = load_config()
bundle = load_model_bundle(config)
vital_cols = config["columns"]["vital_cols"]
lab_cols = config["columns"]["lab_cols"]
static_cols = config["columns"]["static_cols"]
id_col = config["columns"]["id_col"]
time_col = config["columns"]["time_col"]

tab_upload, tab_manual = st.tabs(["Upload Patient CSV", "Manual Single-Hour Entry"])


# ---------- Tab 1: CSV upload ----------
with tab_upload:
    st.subheader("Upload hourly readings for one patient")
    st.write(
        f"CSV must include columns: {id_col}, {time_col}, "
        f"{', '.join(vital_cols + lab_cols + static_cols)}"
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        raw_df = pd.read_csv(uploaded)

        if id_col not in raw_df.columns:
            st.error(f"CSV is missing required column: {id_col}")
        else:
            patient_ids = raw_df[id_col].unique()
            if len(patient_ids) > 1:
                selected_id = st.selectbox("Select patient", patient_ids)
                raw_df = raw_df[raw_df[id_col] == selected_id]

            threshold = st.slider("Alert threshold", 0.0, 1.0, 0.475, 0.01)

            preds = predict(raw_df, config, bundle, threshold=threshold)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=preds[time_col], y=preds["sepsis_probability"],
                mode="lines+markers", name="Sepsis probability",
                line=dict(color="crimson"),
            ))
            fig.add_hline(
                y=threshold, line_dash="dash", line_color="gray",
                annotation_text=f"Threshold ({threshold})",
            )
            fig.update_layout(
                xaxis_title="ICU Length of Stay (hours)",
                yaxis_title="Sepsis Probability",
                yaxis_range=[0, 1],
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

            n_alerts = preds["sepsis_prediction"].sum()
            st.metric("Hours flagged above threshold", f"{n_alerts} / {len(preds)}")

            with st.expander("Full prediction table"):
                st.dataframe(preds, use_container_width=True)


# ---------- Tab 2: Manual single-hour entry ----------
with tab_manual:
    st.subheader("Enter a single hour's vitals")
    st.info(
        "This predicts on one timepoint with no prior history, so "
        "trend-based features (rolling stats, rate-of-change) will be flat. "
        "Predictions here are less reliable than the CSV/trend view above."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Vitals**")
        hr = st.number_input("Heart Rate (HR)", 0.0, 300.0, 80.0)
        o2sat = st.number_input("O2 Saturation (%)", 0.0, 100.0, 97.0)
        temp = st.number_input("Temperature (°C)", 30.0, 45.0, 37.0)
        sbp = st.number_input("Systolic BP (SBP)", 0.0, 300.0, 120.0)
        map_val = st.number_input("Mean Arterial Pressure (MAP)", 0.0, 200.0, 80.0)
        dbp = st.number_input("Diastolic BP (DBP)", 0.0, 200.0, 70.0)
        resp = st.number_input("Respiratory Rate", 0.0, 60.0, 16.0)

    with col2:
        st.markdown("**Labs** (leave at 0 if not drawn)")
        fio2 = st.number_input("FiO2", 0.0, 1.0, 0.0)
        paco2 = st.number_input("PaCO2", 0.0, 100.0, 0.0)
        wbc = st.number_input("WBC", 0.0, 50.0, 0.0)
        creatinine = st.number_input("Creatinine", 0.0, 15.0, 0.0)
        bun = st.number_input("BUN", 0.0, 200.0, 0.0)
        platelets = st.number_input("Platelets", 0.0, 1000.0, 0.0)
        lactate = st.number_input("Lactate", 0.0, 20.0, 0.0)

    with col3:
        st.markdown("**Patient info**")
        age = st.number_input("Age", 0.0, 120.0, 60.0)
        gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        hosp_adm_time = st.number_input("Hours since hospital admission", -500.0, 500.0, 0.0)
        iculos = st.number_input("ICU Length of Stay (hour)", 1, 500, 1)

    if st.button("Predict"):
        lab_inputs = {"FiO2": fio2, "PaCO2": paco2, "WBC": wbc, "Creatinine": creatinine,
                      "BUN": bun, "Platelets": platelets, "Lactate": lactate}
        row = {
            id_col: 999999,
            time_col: iculos,
            "Hour": iculos,
            "HR": hr, "O2Sat": o2sat, "Temp": temp, "SBP": sbp,
            "MAP": map_val, "DBP": dbp, "Resp": resp,
            **{k: (v if v > 0 else np.nan) for k, v in lab_inputs.items()},
            "Age": age, "Gender": gender, "HospAdmTime": hosp_adm_time,
        }
        raw_df = pd.DataFrame([row])
        for col in lab_inputs:
            raw_df[col] = raw_df[col].astype(float)

        preds = predict(raw_df, config, bundle)
        prob = preds["sepsis_probability"].iloc[0]
        pred = preds["sepsis_prediction"].iloc[0]

        st.metric("Sepsis Probability", f"{prob:.1%}")
        if pred == 1:
            st.error("Flagged: above alert threshold")
        else:
            st.success("Not flagged: below alert threshold")