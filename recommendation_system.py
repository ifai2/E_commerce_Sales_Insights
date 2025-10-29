# ==========================================
# recommendation_system.py
# ==========================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# --------------------------------------------------
# LOAD PRODUCT DATA
# --------------------------------------------------
def load_product_data(
    path="olist_products_dataset.csv",
    translation_path="product_category_name_translation.csv"
):
    products = pd.read_csv(path)
    translations = pd.read_csv(translation_path)
    products = products.merge(translations, how="left", on="product_category_name")
    products.rename(columns={"product_category_name_english": "category"}, inplace=True)
    products.dropna(subset=["category"], inplace=True)
    return products


# --------------------------------------------------
# PREPROCESS
# --------------------------------------------------
def preprocess_products(df: pd.DataFrame):
    num_features = ["product_weight_g", "product_length_cm",
                    "product_height_cm", "product_width_cm"]

    for col in num_features:
        df[col] = df[col].fillna(df[col].median()) if col in df.columns else 0

    scaler = MinMaxScaler()
    df[num_features] = scaler.fit_transform(df[num_features])

    df["text_features"] = df["category"].fillna("") + " " + df["product_id"].astype(str)

    vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
    tfidf_matrix = vectorizer.fit_transform(df["text_features"])

    return df.reset_index(drop=True), tfidf_matrix, num_features


# --------------------------------------------------
# RECOMMEND PRODUCTS (TOP-K ONLY)
# --------------------------------------------------
def recommend_similar_products(product_id: str, top_n: int = 5):
    df = load_product_data()
    df, tfidf_matrix, num_features = preprocess_products(df)

    if product_id not in df["product_id"].values:
        raise ValueError(f"Product ID '{product_id}' not found.")

    # Build numeric feature matrix
    num_matrix = df[num_features].values

    # Locate the target product row
    idx = df.index[df["product_id"] == product_id][0]
    target_tfidf = tfidf_matrix[idx]
    target_num = num_matrix[idx].reshape(1, -1)

    # Compute cosine similarities to all others (row-wise only)
    text_sim = cosine_similarity(target_tfidf, tfidf_matrix).ravel()
    num_sim = cosine_similarity(target_num, num_matrix).ravel()
    combined_sim = 0.6 * text_sim + 0.4 * num_sim

    # Sort and select top-n excluding itself
    similar_idx = np.argsort(combined_sim)[::-1][1:top_n + 1]

    recommendations = df.loc[similar_idx, ["product_id", "category"]].copy()
    recommendations["similarity_score"] = combined_sim[similar_idx]
    return recommendations


# --------------------------------------------------
# MAIN TEST
# --------------------------------------------------
if __name__ == "__main__":
    sample_id = "1e9e8ef04dbcff4541ed26657ea517e5"
    try:
        recs = recommend_similar_products(sample_id, top_n=5)
        print("\nRecommended Products:\n", recs)
    except Exception as e:
        print("Error:", e)
