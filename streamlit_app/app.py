from __future__ import annotations

import os
import json
from pathlib import Path

import requests
import streamlit as st
import pandas as pd

from fraud_detection.ui_helpers import (
    default_transaction_payload,
    load_local_status,
    parse_batch_csv,
)

API_URL = os.getenv("API_URL", "http://localhost:8000")


def _api_get(path: str) -> dict:
    response = requests.get(f"{API_URL}{path}", timeout=30)
    response.raise_for_status()
    return response.json()


def _api_post(path: str, payload):
    response = requests.post(f"{API_URL}{path}", json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


# ═══════════════════════════════════════════════════════════════════════════
# Page config & custom CSS
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* ── Global ─────────────────────────────────────────────── */
    .main .block-container { padding-top: 1.5rem; }

    /* ── Metric cards ──────────────────────────────────────── */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        color: #f8fafc;
        box-shadow: 0 4px 24px rgba(0,0,0,0.15);
        margin-bottom: 0.5rem;
    }
    .metric-card h4 {
        margin: 0 0 0.3rem 0;
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
        line-height: 1.2;
    }

    /* ── Status badges ─────────────────────────────────────── */
    .badge-ok {
        display: inline-block;
        background: #10b981;
        color: white;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-warn {
        display: inline-block;
        background: #f59e0b;
        color: white;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-danger {
        display: inline-block;
        background: #ef4444;
        color: white;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* ── Result box ────────────────────────────────────────── */
    .result-fraud {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        border-radius: 12px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .result-legit {
        background: linear-gradient(135deg, #064e3b, #065f46);
        border-radius: 12px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .result-fraud h2, .result-legit h2 {
        margin: 0; font-size: 1.6rem;
    }
    .result-fraud p, .result-legit p {
        margin: 0.3rem 0 0 0; font-size: 1rem; opacity: 0.9;
    }

    /* ── Section headers ───────────────────────────────────── */
    .section-header {
        border-left: 4px solid #6366f1;
        padding-left: 0.8rem;
        margin: 1.5rem 0 1rem 0;
        font-size: 1.1rem;
        font-weight: 600;
    }

    /* ── Sidebar ───────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    [data-testid="stSidebar"] .stMarkdown { color: #e2e8f0; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# Sidebar — System Status
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## Fraud Detection")
    st.markdown("##### MLOps Operations Center")
    st.divider()

    # API health check
    try:
        health = _api_get("/health")
        model_loaded = health.get("model_loaded", False)
        api_status = "Online" if model_loaded else "Degraded"
    except Exception:
        health = {"model_loaded": False, "model_name": "N/A", "version": "N/A"}
        model_loaded = False
        api_status = "Offline"

    st.markdown(f"**API Status:** {api_status}")
    st.markdown(f"**Model:** `{health.get('model_name', 'N/A')}`")
    st.markdown(f"**Version:** `{health.get('version', 'N/A')}`")
    st.divider()

    # Local pipeline status
    local_status = load_local_status()
    training = local_status.get("training", {})
    best_model = training.get("best_model", "N/A")
    best_auprc = training.get("best_val_auprc", 0)

    st.markdown("##### Pipeline Status")
    st.markdown(f"**Best Model:** `{best_model}`")
    st.markdown(f"**Val AUPRC:** `{best_auprc:.4f}`" if isinstance(best_auprc, float) else "N/A")

    promo = local_status.get("promotion", {})
    if promo.get("promoted", False):
        st.markdown('<span class="badge-ok">Promoted</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-warn">Pending</span>', unsafe_allow_html=True)

    st.divider()
    st.caption("Credit Card Fraud Detection — MLOps Project 2026")


# ═══════════════════════════════════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("# Fraud Detection Dashboard")
st.markdown(
    "Real-time transaction scoring, model monitoring, and pipeline insights "
    "for the Credit Card Fraud Detection system."
)

# Quick stats
status = load_local_status()
training_data = status.get("training", {})
results = training_data.get("results", {})
best_name = training_data.get("best_model", "N/A")
best_results = results.get(best_name, {})

col_m1, col_m2, col_m3, col_m4 = st.columns(4)

with col_m1:
    val_auprc = best_results.get("val", {}).get("auprc", 0)
    st.markdown(f"""
    <div class="metric-card">
        <h4>Val AUPRC</h4>
        <p class="value" style="color: #22d3ee;">{val_auprc:.4f}</p>
    </div>
    """, unsafe_allow_html=True)

with col_m2:
    val_recall = best_results.get("val", {}).get("recall", 0)
    st.markdown(f"""
    <div class="metric-card">
        <h4>Val Recall</h4>
        <p class="value" style="color: #34d399;">{val_recall:.4f}</p>
    </div>
    """, unsafe_allow_html=True)

with col_m3:
    val_precision = best_results.get("val", {}).get("precision", 0)
    st.markdown(f"""
    <div class="metric-card">
        <h4>Val Precision</h4>
        <p class="value" style="color: #a78bfa;">{val_precision:.4f}</p>
    </div>
    """, unsafe_allow_html=True)

with col_m4:
    drift_data = status.get("drift", {})
    drift_summary = drift_data.get("summary", {})
    drift_count = drift_summary.get("features_with_drift", 0)
    total_features = drift_summary.get("total_features_checked", 0)
    drift_color = "#10b981" if drift_count == 0 else "#ef4444"
    st.markdown(f"""
    <div class="metric-card">
        <h4>Feature Drift</h4>
        <p class="value" style="color: {drift_color};">{drift_count}/{total_features}</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════
# Tabs
# ═══════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "Single Prediction",
    "Batch Scoring",
    "Model Card",
    "Drift Monitor",
    "Pipeline Status",
])

# ── Tab 1: Single Prediction ────────────────────────────────────────────
with tabs[0]:
    st.markdown('<div class="section-header">Score a Transaction</div>', unsafe_allow_html=True)
    st.markdown("Enter transaction details below to check for fraud.")

    defaults = default_transaction_payload()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Transaction Info**")
        cc_num = st.text_input("Credit Card Number", value=str(defaults["cc_num"]), key="cc")
        merchant = st.text_input("Merchant", value=str(defaults["merchant"]))
        category = st.selectbox(
            "Category",
            [
                "personal_care", "health_fitness", "misc_pos", "misc_net",
                "grocery_pos", "grocery_net", "entertainment", "food_dining",
                "shopping_pos", "shopping_net", "gas_transport", "home",
                "kids_pets", "travel",
            ],
        )
        amt = st.number_input("Amount ($)", min_value=0.0, value=float(defaults["amt"]), step=1.0)
        unix_time = st.number_input("Unix Timestamp", min_value=0, value=int(defaults["unix_time"]))
        trans_num = st.text_input("Transaction ID", value=str(defaults["trans_num"]))

    with col2:
        st.markdown("**Cardholder Info**")
        first = st.text_input("First Name", value=str(defaults["first"]))
        last = st.text_input("Last Name", value=str(defaults["last"]))
        gender = st.selectbox("Gender", ["M", "F"])
        dob = st.text_input("Date of Birth (YYYY-MM-DD)", value=str(defaults["dob"]))
        job = st.text_input("Job Title", value=str(defaults["job"]))
        street = st.text_input("Street Address", value=str(defaults["street"]))

    with col3:
        st.markdown("**Location Info**")
        city = st.text_input("City", value=str(defaults["city"]))
        state = st.text_input("State (2-letter)", value=str(defaults["state"]))
        zip_code = st.text_input("ZIP Code", value=str(defaults["zip"]))
        lat = st.number_input("Latitude", value=float(defaults["lat"]), format="%.4f")
        long_val = st.number_input("Longitude", value=float(defaults["long"]), format="%.4f")
        city_pop = st.number_input("City Population", min_value=0, value=int(defaults["city_pop"]))

    mcol1, mcol2 = st.columns(2)
    with mcol1:
        merch_lat = st.number_input("Merchant Lat", value=float(defaults["merch_lat"]), format="%.6f")
    with mcol2:
        merch_long = st.number_input("Merchant Long", value=float(defaults["merch_long"]), format="%.6f")

    st.markdown("")
    if st.button("Analyze Transaction", use_container_width=True, type="primary"):
        payload = {
            "cc_num": cc_num, "merchant": merchant, "category": category,
            "amt": float(amt), "first": first, "last": last, "gender": gender,
            "street": street, "city": city, "state": state, "zip": zip_code,
            "lat": float(lat), "long": float(long_val), "city_pop": int(city_pop),
            "job": job, "dob": dob, "trans_num": trans_num,
            "unix_time": int(unix_time), "merch_lat": float(merch_lat),
            "merch_long": float(merch_long),
        }
        try:
            result = _api_post("/api/v1/predict", payload)
            prob = result.get("fraud_probability", 0)
            is_fraud = result.get("is_fraud", False)

            if is_fraud:
                st.markdown(f"""
                <div class="result-fraud">
                    <h2>FRAUD DETECTED</h2>
                    <p>Probability: {prob:.4%} — Threshold: {result.get('threshold', 0.5)}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-legit">
                    <h2>Transaction Appears Legitimate</h2>
                    <p>Fraud probability: {prob:.4%} — Threshold: {result.get('threshold', 0.5)}</p>
                </div>
                """, unsafe_allow_html=True)

            with st.expander("Full API Response"):
                st.json(result)

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


# ── Tab 2: Batch Scoring ────────────────────────────────────────────────
with tabs[1]:
    st.markdown('<div class="section-header">Batch Transaction Scoring</div>', unsafe_allow_html=True)
    st.markdown("Upload a CSV file with transaction records to score in bulk.")

    uploaded = st.file_uploader("Upload CSV file", type="csv")
    if uploaded is not None:
        try:
            frame = parse_batch_csv(uploaded.getvalue())
            st.info(f"Loaded **{len(frame)} rows** x **{len(frame.columns)} columns**")
            st.dataframe(frame.head(10), use_container_width=True, height=300)

            if st.button("Score All Transactions", use_container_width=True, type="primary"):
                with st.spinner("Scoring transactions..."):
                    result = _api_post(
                        "/api/v1/predict/batch", frame.to_dict(orient="records")
                    )
                preds = result.get("predictions", result)
                total = len(preds)
                fraud_count = sum(1 for p in preds if p.get("is_fraud", False))

                c1, c2, c3 = st.columns(3)
                c1.metric("Total Scored", total)
                c2.metric("Fraud Detected", fraud_count)
                c3.metric("Fraud Rate", f"{fraud_count/max(total,1):.2%}")

                st.dataframe(pd.DataFrame(preds), use_container_width=True, height=400)

        except Exception as exc:
            st.error(f"Batch scoring failed: {exc}")


# ── Tab 3: Model Card ──────────────────────────────────────────────────
with tabs[2]:
    st.markdown('<div class="section-header">Active Model Information</div>', unsafe_allow_html=True)

    try:
        model_info = _api_get("/api/v1/model")
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Model Name", model_info.get("model_name", "N/A"))
        mcol2.metric("Version", model_info.get("version", "N/A"))
        mcol3.metric("Features", model_info.get("feature_count", 0))

        with st.expander("Full Model Metadata"):
            st.json(model_info)
    except Exception:
        st.warning("API offline. Showing local training metrics.")
        if "training" in status:
            st.json(status["training"])
        else:
            st.info("No local metrics available.")

    # Model comparison table
    if results:
        st.markdown('<div class="section-header">Model Comparison</div>', unsafe_allow_html=True)
        comparison = []
        for model_name, splits in results.items():
            for split_name, metrics in splits.items():
                row = {"Model": model_name, "Split": split_name}
                row.update(metrics)
                comparison.append(row)
        if comparison:
            df_comp = pd.DataFrame(comparison)
            st.dataframe(
                df_comp.style.format({
                    "auc_roc": "{:.4f}", "auprc": "{:.4f}",
                    "recall": "{:.4f}", "precision": "{:.4f}", "f1": "{:.4f}",
                }),
                use_container_width=True,
                height=300,
            )


# ── Tab 4: Drift Monitor ───────────────────────────────────────────────
with tabs[3]:
    st.markdown('<div class="section-header">Data Drift Detection</div>', unsafe_allow_html=True)

    drift_report = status.get("drift", {})
    if not drift_report:
        try:
            drift_report = _api_get("/api/v1/drift")
        except Exception:
            drift_report = {}

    if drift_report:
        summary = drift_report.get("summary", {})
        target_info = drift_report.get("target", {})

        dc1, dc2, dc3 = st.columns(3)
        total_checked = summary.get("total_features_checked", 0)
        drifted = summary.get("features_with_drift", 0)
        overall = summary.get("overall_drift_detected", False)

        dc1.metric("Features Checked", total_checked)
        dc2.metric("Features Drifted", drifted)
        if overall:
            dc3.markdown('<span class="badge-danger">⚠️ Drift Detected</span>', unsafe_allow_html=True)
        else:
            dc3.markdown('<span class="badge-ok">✓ No Drift</span>', unsafe_allow_html=True)

        # Target drift
        if target_info:
            st.markdown("##### Target Distribution")
            tc1, tc2, tc3 = st.columns(3)
            tc1.metric("Ref Fraud Rate", f"{target_info.get('reference_fraud_rate', 0):.4%}")
            tc2.metric("Current Fraud Rate", f"{target_info.get('current_fraud_rate', 0):.4%}")
            tc3.metric("p-value", f"{target_info.get('p_value', 0):.4f}")

        # Feature details
        features = drift_report.get("features", {})
        if features:
            with st.expander(f"Feature-level Drift Details ({len(features)} features)"):
                feature_rows = []
                for fname, fdata in features.items():
                    feature_rows.append({
                        "Feature": fname,
                        "Statistic": round(fdata.get("statistic", 0), 6),
                        "p-value": round(fdata.get("p_value", 0), 4),
                        "Drifted": "⚠️ Yes" if fdata.get("drifted") else "✅ No",
                    })
                st.dataframe(pd.DataFrame(feature_rows), use_container_width=True, height=500)
    else:
        st.info("No drift report available.")


# ── Tab 5: Pipeline Status ─────────────────────────────────────────────
with tabs[4]:
    st.markdown('<div class="section-header">Pipeline &amp; Promotion Status</div>', unsafe_allow_html=True)

    pcol1, pcol2 = st.columns(2)

    with pcol1:
        st.markdown("##### Champion Model")
        promo = status.get("promotion", {})
        if promo:
            st.metric("Promoted", "Yes" if promo.get("promoted") else "No")
            st.markdown(f"**Reason:** {promo.get('reason', 'N/A')}")
            with st.expander("Promotion Details"):
                st.json(promo)
        else:
            st.info("No promotion data available.")

    with pcol2:
        st.markdown("##### Latest Training Run")
        if training_data:
            st.metric("Best Model", training_data.get("best_model", "N/A"))
            st.metric("Best Val AUPRC", f"{training_data.get('best_val_auprc', 0):.4f}")
            st.metric("Run ID", training_data.get("best_run_id", "N/A")[:12] + "...")
        else:
            st.info("No training data available.")

    # Candidate metrics
    candidate = status.get("candidate", {})
    if candidate:
        st.markdown("##### Evaluation Results")
        bias_var = candidate.get("bias_variance", {})
        if bias_var:
            bv1, bv2, bv3, bv4 = st.columns(4)
            bv1.metric("Train AUPRC", f"{bias_var.get('train_auprc', 0):.4f}")
            bv2.metric("Val AUPRC", f"{bias_var.get('val_auprc', 0):.4f}")
            bv3.metric("Test AUPRC", f"{bias_var.get('test_auprc', 0):.4f}")
            bv4.metric("Diagnosis", bias_var.get("diagnosis", "N/A"))
