import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
import streamlit as st
from pathlib import Path

# =========================================
# LOAD PRODUCT DATA
# =========================================
@st.cache_data
def load_product_data():
    # تأكد من مسار الملف
    path = '/home/faial/code/ifai2/E_commerce_Sales_Insights/olist_order_items_dataset.csv'
    if not Path(path).exists():  # التحقق من وجود الملف
        st.error("Missing product data file: olist_products_dataset.csv")
        st.stop()
    return pd.read_csv(path)

# =========================================
# DATA PREPROCESSING
# =========================================
def preprocess_product_data(products_df):
    # تحقق من الأعمدة في البيانات
    print(products_df.columns)

    # إزالة القيم المفقودة من الأعمدة الضرورية
    products_df = products_df.dropna(subset=['price', 'freight_value'])  # تأكد من الأعمدة الصحيحة
    products_df = products_df[['product_id', 'price', 'freight_value']]  # اختر الأعمدة المهمة

    return products_df

# =========================================
# TRAINING THE RECOMMENDATION MODEL (KNN)
# =========================================
def train_recommendation_model(products_df):
    # معالجة البيانات
    products_data = preprocess_product_data(products_df)

    # تحويل البيانات إلى مصفوفة
    X = products_data[['price', 'freight_value']].values  # اختيار الميزات (يمكنك إضافة المزيد من الميزات)

    # تدريب نموذج NearestNeighbors
    model = NearestNeighbors(n_neighbors=5, algorithm='auto', metric='euclidean')
    model.fit(X)

    # حفظ النموذج
    with open('recommendation_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    st.success("Recommendation model trained successfully!")

# =========================================
# FUNCTION FOR PRODUCT RECOMMENDATIONS
# =========================================
def recommend_products(product_id, products_df, model):
    # البحث عن المنتج في البيانات
    product_data = products_df[products_df['product_id'] == product_id]

    if product_data.empty:
        st.error("Product not found!")
        return None

    # استخراج بيانات المنتج المطلوب
    product_features = product_data[['price', 'freight_value']].values

    # العثور على أقرب المنتجات
    distances, indices = model.kneighbors(product_features)

    # استخراج المنتجات الموصى بها
    recommended_products = products_df.iloc[indices[0]]

    return recommended_products[['product_id', 'price', 'freight_value']]

# =========================================
# STREAMLIT APP FOR USER INTERFACE
# =========================================
def main():
    # تحميل البيانات
    products_df = load_product_data()

    # تدريب النموذج إذا لم يكن موجودًا
    try:
        with open('recommendation_model.pkl', 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        st.write("Training the recommendation model...")
        train_recommendation_model(products_df)
        with open('recommendation_model.pkl', 'rb') as f:
            model = pickle.load(f)

    # واجهة المستخدم في Streamlit
    st.title("Product Recommendation System")
    st.caption("Enter a product ID to get recommendations based on similar products.")

    product_id = st.text_input("Enter Product ID:")

    if product_id:
        # الحصول على التوصيات
        recommendations = recommend_products(product_id, products_df, model)
        if recommendations is not None:
            st.write("### Recommended Products:")
            st.dataframe(recommendations)
        else:
            st.write("No recommendations found.")

if __name__ == "__main__":
    main()
