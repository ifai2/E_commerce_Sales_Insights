import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------
# Load Model and Data
# ------------------------------
@st.cache_resource
def load_model():
    with open("seller_profitability_model_simple.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_seller_data():
    return pd.read_csv("cleaned_seller_data.csv")

model = load_model()
seller_data = load_seller_data()

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Seller Profitability Predictor", page_icon="💼", layout="centered")
st.title("Seller Profitability Prediction App")

st.write("Predict if a seller is profitable (1) or not profitable (0) based on key performance indicators.")

# ------------------------------
# User Input
# ------------------------------
st.subheader("Enter Seller Data")

col1, col2 = st.columns(2)
with col1:
    total_orders = st.number_input("Total Orders", min_value=0, step=1, value=50)
    total_products_sold = st.number_input("Total Products Sold", min_value=0, step=1, value=75)
    avg_price = st.number_input("Average Price (BRL)", min_value=0.0, step=1.0, value=120.0)

with col2:
    avg_freight = st.number_input("Average Freight (BRL)", min_value=0.0, step=1.0, value=18.0)
    avg_review_score = st.slider("Average Review Score", 1.0, 5.0, 4.2, 0.1)

input_df = pd.DataFrame([{
    "total_orders": total_orders,
    "total_products_sold": total_products_sold,
    "avg_price": avg_price,
    "avg_freight": avg_freight,
    "avg_review_score": avg_review_score
}])

# ------------------------------
# Prediction
# ------------------------------
if st.button("Predict Profitability"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"Profitable Seller (Confidence: {probability[1]:.1%})")
    else:
        st.error(f"Not Profitable Seller (Confidence: {probability[0]:.1%})")

# ------------------------------
# Feature Importance
# ------------------------------
if st.button("Show Feature Importance"):
    rf = model.named_steps['clf']
    importances = rf.feature_importances_
    features = ['total_orders', 'total_products_sold', 'avg_price', 'avg_freight', 'avg_review_score']

    fig, ax = plt.subplots(figsize=(8, 4))
    y_pos = np.arange(len(features))
    ax.barh(y_pos, importances)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance in Profitability Prediction')
    st.pyplot(fig)

# ------------------------------
# Seller Recommendation
# ------------------------------
if st.button("Find Similar Profitable Sellers"):
    prediction = model.predict(input_df)[0]

    if prediction == 0:
        st.info("Finding profitable sellers with similar characteristics...")
        profitable_sellers = seller_data[seller_data['profitable'] == 1].copy()

        if not profitable_sellers.empty:
            features = ['total_orders', 'total_products_sold', 'avg_price', 'avg_freight', 'avg_review_score']
            for feature in features:
                profitable_sellers[f'{feature}_diff'] = abs(
                    profitable_sellers[feature] - input_df[feature].iloc[0]
                )

            profitable_sellers['total_diff'] = profitable_sellers[[f'{f}_diff' for f in features]].sum(axis=1)
            similar_sellers = profitable_sellers.nsmallest(3, 'total_diff')

            st.subheader("Recommended Profitable Sellers")
            for idx, seller in similar_sellers.iterrows():
                st.write(f"Seller ID: {seller['seller_id']}")
                st.write(f"Location: {seller.get('seller_city', 'Unknown')}, {seller.get('seller_state', 'Unknown')}")
                st.write(f"Performance: {int(seller['total_orders'])} orders, {seller['avg_review_score']:.1f} rating")
                st.write(f"Revenue: BRL {seller['total_revenue']:,.2f}")
                st.write("---")
        else:
            st.warning("No profitable sellers found in the dataset.")
    else:
        st.success("This seller is already profitable.")

st.sidebar.info("""
About this app:
- Predicts seller profitability using a trained Random Forest model
- Built with Olist e-commerce dataset
- Run locally with:  streamlit run app.py
""")
