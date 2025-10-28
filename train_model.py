# ==========================================
# train_model.py
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pickle

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# ------------------------------
# Load Data
# ------------------------------
customers = pd.read_csv('olist_customers_dataset.csv')
geolocation = pd.read_csv('olist_geolocation_dataset.csv')
order_items = pd.read_csv('olist_order_items_dataset.csv')
order_payments = pd.read_csv('olist_order_payments_dataset.csv')
order_reviews = pd.read_csv('olist_order_reviews_dataset.csv')
orders = pd.read_csv('olist_orders_dataset.csv')
products = pd.read_csv('olist_products_dataset.csv')
sellers = pd.read_csv('olist_sellers_dataset.csv')
category_translation = pd.read_csv('product_category_name_translation.csv')

print("All datasets loaded successfully")

# ------------------------------
# Prepare Seller Data
# ------------------------------
def prepare_seller_data():
    orders_payments = orders.merge(
        order_payments[['order_id', 'payment_value']],
        on='order_id',
        how='left'
    )

    orders_items_payments = order_items.merge(
        orders_payments[['order_id', 'order_purchase_timestamp', 'payment_value']],
        on='order_id',
        how='left'
    ).merge(
        order_reviews[['order_id', 'review_score']],
        on='order_id',
        how='left'
    )

    seller_stats = orders_items_payments.groupby('seller_id').agg({
        'order_id': 'nunique',
        'order_item_id': 'count',
        'price': 'mean',
        'freight_value': 'mean',
        'payment_value': 'sum',
        'review_score': 'mean'
    }).reset_index()

    seller_stats.columns = [
        'seller_id', 'total_orders', 'total_products_sold',
        'avg_price', 'avg_freight', 'total_revenue', 'avg_review_score'
    ]

    seller_data = seller_stats.merge(sellers, on='seller_id', how='left')

    seller_data.fillna({
        'avg_review_score': seller_data['avg_review_score'].mean(),
        'avg_price': seller_data['avg_price'].mean(),
        'avg_freight': seller_data['avg_freight'].mean(),
        'total_revenue': 0
    }, inplace=True)

    return seller_data


seller_data = prepare_seller_data()
print("Seller data prepared successfully")

# ------------------------------
# Create Target Variable
# ------------------------------
median_revenue = seller_data['total_revenue'].median()
seller_data['profitable'] = (seller_data['total_revenue'] > median_revenue).astype(int)
print("Target variable created:")
print(seller_data['profitable'].value_counts())

# ------------------------------
# Split Data
# ------------------------------
features = ['total_orders', 'total_products_sold', 'avg_price', 'avg_freight', 'avg_review_score']
X = seller_data[features]
y = seller_data['profitable']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------
# Train Model
# ------------------------------
model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    ))
])

model.fit(X_train, y_train)

# ------------------------------
# Evaluate Model
# ------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Evaluation:")
print("Accuracy:", accuracy)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print("Cross-validation scores:", cv_scores)
print("Mean CV accuracy:", cv_scores.mean())

# ------------------------------
# Save Model and Data
# ------------------------------
with open('seller_profitability_model_simple.pkl', 'wb') as f:
    pickle.dump(model, f)

seller_data.to_csv('cleaned_seller_data.csv', index=False)
print("Model and data saved successfully")

# ------------------------------
# Visualization
# ------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

rf = model.named_steps['clf']
importances = rf.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values('Importance', ascending=True)

axes[0].barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
axes[0].set_title('Feature Importance')
axes[0].set_xlabel('Importance')

ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=axes[1], cmap='Blues')
axes[1].set_title('Confusion Matrix')

plt.tight_layout()
plt.show()

print("\nAdditional Analysis:")
print(f"Total sellers: {len(seller_data)}")
print(f"Profitable sellers: {seller_data['profitable'].sum()} ({seller_data['profitable'].mean()*100:.1f}%)")
print(f"Median revenue: BRL {median_revenue:,.2f}")

print("\nTop Features by Importance:")
for feature, importance in zip(features, importances):
    print(f"{feature}: {importance:.3f}")

# ------------------------------
# Explainability (SHAP)
# ------------------------------
print("\nGenerating SHAP explanations...")
explainer = shap.Explainer(rf, X_train)
shap_values = explainer(X_test)

# Summary Plot
shap.summary_plot(shap_values, X_test, feature_names=features, show=True)

# Mean Absolute SHAP Values
mean_shap = np.abs(shap_values.values).mean(axis=0)
shap_importance = pd.DataFrame({
    'Feature': features,
    'Mean_SHAP_Value': mean_shap
}).sort_values(by='Mean_SHAP_Value', ascending=False)

print("\nFeature Impact on Profitability (based on SHAP values):")
print(shap_importance)
