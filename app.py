import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import NearestNeighbors

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="E-commerce Sales Insights • Profitability",
    page_icon="💹",
    layout="centered",  # Ensures the content is centered
    initial_sidebar_state="expanded",
)

# =========================================
# THEME — Light Mode
# =========================================
PRIMARY_GREEN = "#0f9d58"
SUBTLE_GOLD  = "#e3c76a"
SOFT_BG      = "#f8f9fa"  # Light background

# Add custom light mode CSS and center alignment
st.markdown(
    f"""
    <style>
        html, body, [class*="css"] {{
            font-family: 'Segoe UI', sans-serif;
            background-color: {SOFT_BG};
            color: #333333;
            font-size: 14px;
            display: flex;
            justify-content: center;
            align-items: flex-start;
            margin: 0;
            height: 100vh;
            overflow: hidden;
        }}
        .stApp {{
            width: 100%;
            max-width: 1200px;  /* Adjust the max-width to control content width */
            margin: 0 auto;
        }}
        .banner {{
            padding: 10px 14px;
            background: linear-gradient(90deg, {PRIMARY_GREEN} 0%, {SUBTLE_GOLD} 100%);
            border-radius: 10px;
            color: white;
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 14px;
            text-align: center;
        }}
        .card {{
            background: #ffffff;
            color: #333333;
            padding: 0.9rem;
            border-radius: 10px;
            border: 1px solid #e6e6e6;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            margin-bottom: 0.8rem;
            width: 100%;
            max-width: 900px;  /* To control the width */
            margin-left: auto;
            margin-right: auto;
        }}
        .profit {{
            background: rgba(15,157,88,0.12);
        }}
        .notprofit {{
            background: rgba(217,83,79,0.12);
        }}
        .block-container {{
            padding-top: 1rem;
            padding-bottom: 1rem;
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
        }}
        .stSlider .st-bu {{
            color: #333333;
        }}
        .stMetric, .stTable {{
            text-align: center;  /* Center-align the metrics and tables */
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================
# LOAD MODEL
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
# LOAD DATA
# =========================================
@st.cache_data
def load_seller_data():
    path = Path("cleaned_seller_data.csv")
    if not path.exists():
        st.error("Missing data file: cleaned_seller_data.csv")
        st.stop()
    return pd.read_csv(path)

@st.cache_data
def load_product_data():
    path = Path("olist_products_dataset.csv")
    if not path.exists():
        st.error("Missing product data file: olist_products_dataset.csv")
        st.stop()
    return pd.read_csv(path)

def predict_proba(model, X: pd.DataFrame) -> Tuple[int, float]:
    if hasattr(model, "predict_proba"):
        y = model.predict(X)
        proba = model.predict_proba(X)
        conf = np.take_along_axis(proba, y.reshape(-1,1), axis=1).ravel()
        return int(y[0]), float(conf[0])
    y = model.predict(X)
    return int(y[0]), 0.50

# =========================================
# SIDEBAR
# =========================================
with st.sidebar:
    st.header("🔧 Controls")
    nav = st.radio("Navigation", ["Predict", "Insights", "Results", "Product Recommendation", "About"], index=0)

# =========================================
# HEADER
# =========================================
st.title("💹 E-commerce Sales Insights — Seller Profitability")
st.markdown('<div class="banner">Profit intelligence for e-commerce sellers ⚡</div>', unsafe_allow_html=True)

# =========================================
# PREDICT PAGE
# =========================================
if nav == "Predict":
    st.subheader("Predict Seller Profitability")
    st.caption("Fill the KPIs below — the model predicts whether the seller is profitable (1) or not (0).")

    col1, col2, col3 = st.columns(3)
    with col1:
        total_orders = st.number_input("Total Orders", min_value=0, step=1, value=75)
    with col2:
        total_products_sold = st.number_input("Total Products Sold", min_value=0, step=1, value=180)
    with col3:
        avg_review = st.slider("Average Review Score", 1.0, 5.0, 4.3, 0.1)

    col4, col5 = st.columns(2)
    with col4:
        avg_price = st.number_input("Average Price (BRL)", min_value=0.0, step=0.1, value=90.0)
    with col5:
        avg_freight = st.number_input("Average Freight (BRL)", min_value=0.0, step=0.1, value=18.0)

    if st.button("🔮 Predict"):
        X = pd.DataFrame([{
            "total_orders": total_orders,
            "total_products_sold": total_products_sold,
            "avg_price": avg_price,
            "avg_freight": avg_freight,
            "avg_review_score": avg_review,
        }])
        label, conf = predict_proba(model, X)
        percent = conf * 100
        status = "✅ Profitable Seller" if label == 1 else "❌ Not Profitable"
        css = "profit" if label == 1 else "notprofit"

        st.markdown(f'<div class="card {css}">', unsafe_allow_html=True)
        st.subheader(status)
        st.metric("Confidence", f"{percent:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)

# =========================================
# INSIGHTS PAGE — with SHAP explainability
# =========================================
if nav == "Insights":
    st.subheader("Feature Importance & Explainability")
    st.caption("Which KPIs most influence profitability?")

    features = ["total_orders", "total_products_sold", "avg_price", "avg_freight", "avg_review_score"]
    seller_data = load_seller_data()
    est = getattr(model, "named_steps", {}).get("clf", model)

    # --- Feature Importance ---
    if hasattr(est, "feature_importances_"):
        importances = np.array(est.feature_importances_, dtype=float)
        order = np.argsort(importances)[::-1]
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.barh(np.array(features)[order][::-1], importances[order][::-1], color="seagreen")
        ax.set_xlabel("Importance", fontsize=9)
        ax.tick_params(labelsize=8)
        plt.tight_layout(pad=0.5)
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("Feature importance not available.")

    # --- SHAP Explainability ---
    st.markdown("### 🧠 SHAP Explainability")
    st.caption("Understanding how each feature influences model predictions.")
    try:
        X = seller_data[features].sample(200, random_state=42)
        explainer = shap.Explainer(est, X)
        shap_values = explainer(X)
        shap.summary_plot(shap_values, X, feature_names=features, show=False)
        fig_shap = plt.gcf()
        fig_shap.set_size_inches(6, 3.5)
        plt.tight_layout(pad=0.4)
        st.pyplot(fig_shap, use_container_width=True)
        plt.clf()
    except Exception as e:
        st.warning(f"SHAP could not be displayed: {e}")

# =========================================
# PRODUCT RECOMMENDATION PAGE
# =========================================
if nav == "Product Recommendation":
    st.subheader("Get Product Recommendations")
    st.caption("Enter your Seller ID to get personalized product recommendations.")

    seller_id = st.text_input("Enter your Seller ID", "")

    if seller_id:
        # Assuming 'recommend_products' is a function you define based on customer behavior or purchases
        try:
            recommended_products = recommend_products(seller_id)
            st.write("### Recommended Products:")
            st.dataframe(recommended_products[['product_id', 'product_category', 'avg_price', 'total_sales']])
        except Exception as e:
            st.error(f"Error: {e}")

# =========================================
# RESULTS PAGE — with Model Evaluation + Recommendations
# =========================================
if nav == "Results":
    st.subheader("📊 Analysis Results")
    st.caption("Exploratory analysis of seller data and model performance.")
    seller_data = load_seller_data()

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Sellers", f"{len(seller_data):,}")
    with col2:
        st.metric("Profitable Sellers", f"{seller_data['profitable'].sum():,}")
    with col3:
        pct = seller_data['profitable'].mean() * 100
        st.metric("Profitability Rate", f"{pct:.1f}%")

    # --- Correlation Heatmap ---
    st.markdown("### 🔍 Correlation Heatmap")
    corr = seller_data[['total_orders','total_products_sold','avg_price','avg_freight','avg_review_score','total_revenue']].corr()
    fig2, ax2 = plt.subplots(figsize=(6, 3.5))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax2, annot_kws={"size":8})
    ax2.tick_params(axis='x', labelsize=8)
    ax2.tick_params(axis='y', labelsize=8)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig2, use_container_width=True)

    # --- Top States ---
    st.markdown("### 🌎 Top 10 States by Profitability")
    top_states = seller_data.groupby('seller_state')['profitable'].mean().sort_values(ascending=False).head(10)
    fig3, ax3 = plt.subplots(figsize=(6, 3.5))
    top_states.plot(kind='bar', color='seagreen', ax=ax3)
    ax3.set_ylabel('% Profitable Sellers', fontsize=9)
    ax3.tick_params(axis='x', labelrotation=30, labelsize=8)
    ax3.tick_params(axis='y', labelsize=8)
    plt.tight_layout(pad=0.4)
    st.pyplot(fig3, use_container_width=True)

    # --- Scatter Plot ---
    st.markdown("### ⭐ Review Score vs Total Revenue")
    fig4, ax4 = plt.subplots(figsize=(6, 3.5))
    sns.scatterplot(
        data=seller_data,
        x='avg_review_score',
        y='total_revenue',
        hue='profitable',
        palette='coolwarm',
        alpha=0.6,
        s=25,
        ax=ax4
    )
    ax4.set_xlabel("Average Review Score", fontsize=9)
    ax4.set_ylabel("Total Revenue (BRL)", fontsize=9)
    ax4.legend(fontsize=8, title_fontsize=8, loc="upper right")
    plt.tight_layout(pad=0.5)
    st.pyplot(fig4, use_container_width=True)

    # --- Tables ---
    st.markdown("### 🏆 Top 5 Best Sellers (by Total Revenue)")
    top_5_sellers = seller_data.sort_values(by='total_revenue', ascending=False).head(5)
    st.dataframe(top_5_sellers[['seller_id','seller_city','seller_state','total_revenue','total_orders','avg_review_score']], height=130)
    top5_share = 100 * top_5_sellers['total_revenue'].sum() / seller_data['total_revenue'].sum()
    st.caption(f"These top 5 sellers generate **{top5_share:.1f}%** of all platform revenue.")

    st.markdown("### 🌟 Top 5 Best Rated Sellers")
    top_5_rated = seller_data.sort_values(by='avg_review_score', ascending=False).head(5)
    st.dataframe(top_5_rated[['seller_id','seller_city','seller_state','avg_review_score','total_revenue','total_orders']], height=130)

    st.markdown("### ⚠️ Bottom 5 Sellers (Lowest Revenue)")
    bottom_5 = seller_data.sort_values(by='total_revenue', ascending=True).head(5)
    st.dataframe(bottom_5[['seller_id','seller_city','seller_state','total_revenue','avg_review_score','total_orders']], height=130)

    # --- Model Evaluation ---
    st.markdown("### 🧩 Model Evaluation & Insights")
    features = ["total_orders", "total_products_sold", "avg_price", "avg_freight", "avg_review_score"]
    X = seller_data[features]
    y_true = seller_data["profitable"]
    y_pred = model.predict(X)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    st.metric("Model Accuracy", f"{acc*100:.2f}%")
    fig_cm, ax_cm = plt.subplots(figsize=(5.5, 3.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False, ax=ax_cm)
    ax_cm.set_xlabel("Predicted Label", fontsize=9)
    ax_cm.set_ylabel("True Label", fontsize=9)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig_cm, use_container_width=True)

    if acc >= 0.85:
        verdict = "✅ The model is excellent — it predicts seller profitability accurately and consistently."
    elif acc >= 0.75:
        verdict = "👍 The model is good — reliable for most practical uses, with room for improvement."
    else:
        verdict = "⚠️ The model is weak — accuracy is below expectations, further tuning needed."

    st.info(verdict)

    st.markdown(
        """
        **Interpretation Summary**
        - The model mainly relies on `avg_review_score`, `total_orders`, and `total_products_sold`.
        - Weak SHAP interaction values indicate independent and simple relationships — acceptable for business data.
        - Overall: high performance, stability, and interpretability.
        """
    )

    # =========================================
    # 💡 Business Recommendations — Visual Cards
    # =========================================
    st.markdown("## 💡 Recommendations Summary")

    cols = st.columns(4)

    with cols[0]:
        st.markdown("### 🤝")
        st.markdown("**Improve Reviews**")
        st.caption("Enhance customer support and feedback quality.")

    with cols[1]:
        st.markdown("### 🚚")
        st.markdown("**Optimize Logistics**")
        st.caption("Reduce freight costs in low-profit regions.")

    with cols[2]:
        st.markdown("### 📈")
        st.markdown("**Focus on High-Performing States**")
        st.caption("Target marketing and expansion where sellers perform best.")

    with cols[3]:
        st.markdown("### 🧠")
        st.markdown("**Leverage Model Insights**")
        st.caption("Detect at-risk or high-potential sellers early.")

# =========================================
# ABOUT PAGE
# =========================================
if nav == "About":
    st.subheader("About this App")
    st.markdown(
        """
        **E-commerce Sales Insights**
        - Predicts seller profitability using Random Forest
        - Adds SHAP explainability for feature influence
        - Includes automatic model performance evaluation
        - Shows Top & Bottom performing sellers
        - Balanced visuals and optimized layout
        - Dataset currency: **BRL (Brazilian Real)**
        - Built with ❤️ using Streamlit
        """
    )
