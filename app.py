import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple

# =========================================
# Page config
# =========================================
st.set_page_config(
    page_title="E-commerce Sales Insights • Profitability",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================
# THEME — Green → subtle Gold
# =========================================
PRIMARY_GREEN = "#0f9d58"
SUBTLE_GOLD  = "#e3c76a"   # ذهبي خفيف
ACCENT_RED   = "#d9534f"
SOFT_BG      = "#f8f9fa"

st.markdown(
    f"""
    <style>
      html, body, [class*="css"] {{
        font-family: 'Segoe UI', sans-serif;
        background-color: {SOFT_BG};
        font-size: 16px;
      }}

      /* Thin elegant banner */
      .banner {{
        padding: 10px 14px;
        background: linear-gradient(90deg, {PRIMARY_GREEN} 0%, {SUBTLE_GOLD} 100%);
        border-radius: 10px;
        color: white;
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 14px;
      }}

      /* Cards */
      .card {{
        background: #fff;
        padding: 1.1rem;
        border-radius: 12px;
        border: 1px solid #e6e6e6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        margin-bottom: 0.9rem;
      }}
      .card-title {{
        display:flex; align-items:center; gap:.6rem;
        font-weight:700; font-size:1.15rem; margin-bottom:.6rem;
      }}
      .muted {{ color:#6c757d; }}

      .profit    {{ background: rgba(15,157,88,0.12); }}
      .notprofit {{ background: rgba(217,83,79,0.12); }}

      /* Centered big button */
      .center-btn button {{
        width: 280px !important;
        height: 44px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================
# Load Model
# =========================================
@st.cache_resource(show_spinner=False)
def load_model():
    path = Path("seller_profitability_model_simple.pkl")
    if not path.exists():
        st.error("Missing model file: seller_profitability_model_simple.pkl")
        st.stop()
    return pickle.load(open(path, "rb"))

model = load_model()

# =========================================
# Prediction helper
# =========================================
def predict_proba(model, X: pd.DataFrame) -> Tuple[int, float]:
    if hasattr(model, "predict_proba"):
        y = model.predict(X)
        proba = model.predict_proba(X)
        conf = np.take_along_axis(proba, y.reshape(-1,1), axis=1).ravel()
        return int(y[0]), float(conf[0])
    y = model.predict(X)
    return int(y[0]), 0.50  # fallback

# =========================================
# Sidebar Navigation
# =========================================
with st.sidebar:
    st.header("🔧 Controls")
    nav = st.radio("Navigation", ["Predict", "Insights", "About"], index=0)

# =========================================
# Header
# =========================================
st.title("💹 E-commerce Sales Insights — Seller Profitability")
st.markdown('<div class="banner">Profit intelligence for e-commerce sellers ⚡</div>', unsafe_allow_html=True)

# =========================================
# Predict
# =========================================
if nav == "Predict":
    st.subheader("Predict Seller Profitability")
    st.caption("Fill the KPIs below — the model predicts whether the seller is profitable (1) or not (0).")

    # Inputs Card (3×2 grid)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📈 Seller Performance Metrics</div>', unsafe_allow_html=True)

    # row 1
    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        total_orders = st.number_input(
            "Total Orders",
            min_value=0, step=1, value=75,
            help="How many orders the seller fulfilled in total."
        )
    with r1c2:
        total_products_sold = st.number_input(
            "Total Products Sold",
            min_value=0, step=1, value=180,
            help="How many products the seller has sold overall."
        )
    with r1c3:
        avg_review = st.slider(
            "Average Review Score",
            1.0, 5.0, 4.3, 0.1,
            help="Customer satisfaction rating (1.0 = lowest, 5.0 = highest)."
        )

    # row 2
    r2c1, r2c2, _ = st.columns([1,1,1])
    with r2c1:
        avg_price = st.number_input(
            "Average Product Price (BRL)",
            min_value=0.0, step=0.1, value=90.0,
            help="Average price per product in BRL."
        )
    with r2c2:
        avg_freight = st.number_input(
            "Average Freight (BRL)",
            min_value=0.0, step=0.1, value=18.0,
            help="Average shipping cost per order in BRL."
        )

    # centered Predict button
    b1, b2, b3 = st.columns([1,1,1])
    with b2:
        with st.container():
            st.markdown('<div class="center-btn">', unsafe_allow_html=True)
            go = st.button("🔮 Predict")
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # end card

    if go:
        X = pd.DataFrame([{
            "total_orders": total_orders,
            "total_products_sold": total_products_sold,
            "avg_price": avg_price,
            "avg_freight": avg_freight,
            "avg_review_score": avg_review,
        }])

        label, conf = predict_proba(model, X)
        percent = conf * 100
        status_text = "✅ Profitable Seller" if label == 1 else "❌ Not Profitable"
        css_class = "profit" if label == 1 else "notprofit"

        st.markdown(f'<div class="card {css_class}">', unsafe_allow_html=True)
        st.subheader(status_text)

        m1, m2, m3 = st.columns(3)
        with m1: st.metric("Confidence", f"{percent:.1f}%")
        with m2: st.metric("Total Orders", f"{total_orders:,}"); st.metric("Products Sold", f"{total_products_sold:,}")
        with m3: st.metric("Avg Price (BRL)", f"{avg_price:.2f}"); st.metric("Avg Freight (BRL)", f"{avg_freight:.2f}")
        st.metric("Avg Review", f"{avg_review:.1f}")

        st.markdown('</div>', unsafe_allow_html=True)

# =========================================
# Insights
# =========================================
if nav == "Insights":
    st.subheader("Feature Importance")
    st.caption("Which KPIs contribute most to profitability?")

    features = ["total_orders","total_products_sold","avg_price","avg_freight","avg_review_score"]
    est = getattr(model, "named_steps", {}).get("clf", model)

    if hasattr(est, "feature_importances_"):
        importances = np.array(est.feature_importances_, dtype=float)
        order = np.argsort(importances)[::-1]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(np.array(features)[order][::-1], importances[order][::-1])
        ax.set_xlabel("Importance")
        st.pyplot(fig)
    else:
        st.info("Feature importance not available for this model.")

# =========================================
# About
# =========================================
if nav == "About":
    st.subheader("About this App")
    st.markdown(
        """
        **E-commerce Sales Insights**
        Predicting seller profitability using machine learning.
        - Streamlit-based web interface
        - Clean, business-oriented UI
        - Original dataset currency: **BRL (Brazilian Real)**
        """
    )
