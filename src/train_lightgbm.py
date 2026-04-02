# https://www.geeksforgeeks.org/machine-learning/regression-using-lightgbm/
# https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html
# Ke et al. (2017) - LightGBM: A Highly Efficient Gradient Boosting Decision Tree

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from load_data import load_csv

train = load_csv("esa_train.csv")
test = load_csv("esa_test.csv")

# Drop columns with too many missing values (>50%)
# Can't fill that much data
threshold = 0.5
missing_frac = train.isnull().mean()
cols_to_drop = missing_frac[missing_frac > threshold].index.tolist()
train = train.drop(columns=cols_to_drop)
test = test.drop(columns=cols_to_drop)

# Encode c_object_type (text like PAYLOAD, DEBRIS) into numbers
# Fit categories on train only, then apply the same mapping to test
# This prevents test data from leaking into the encoding
if "c_object_type" in train.columns:
    categories = train["c_object_type"].astype("category").cat.categories
    train["c_object_type"] = pd.Categorical(train["c_object_type"], categories=categories).codes
    test["c_object_type"] = pd.Categorical(test["c_object_type"], categories=categories).codes

# Fill remaining missing values with median from training set
# Using train medians for both prevents test data from influencing the model
train_medians = train.median(numeric_only=True)
train = train.fillna(train_medians)
test = test.fillna(train_medians)

TARGET = "risk"
ID_COLS = ["event_id", "mission_id"]

# Remove max_risk_estimate and max_risk_scaling
# These are derived from the target - including them gave R² = 0.9998 
# Removing them dropped R² to 0.85 
# https://www.geeksforgeeks.org/machine-learning/what-is-data-leakage/
LEAKY_COLS = ["max_risk_estimate", "max_risk_scaling"]

FEATURE_COLS = [c for c in train.columns if c not in [TARGET] + ID_COLS + LEAKY_COLS]

X_train = train[FEATURE_COLS]
y_train = train[TARGET]
X_test = test[FEATURE_COLS]
y_test = test[TARGET]

print(f"\nTraining LightGBM on {len(FEATURE_COLS)} features...")

# LightGBM configuration
# n_estimators=200: build 200 decision trees
# learning_rate=0.05: conservative rate to prevent overfitting
# num_leaves=31: controls tree complexity (default, works well)
# random_state=42: this will make the results reproducible
model = LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42
)

model.fit(X_train, y_train)
preds = model.predict(X_test)

# Calculate metrics
rmse = mean_squared_error(y_test, preds) ** 0.5
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print("\n=== RESULTS ===")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"R²:   {r2:.4f}")

# Save the trained model
import os
os.makedirs("results/models", exist_ok=True)
joblib.dump(model, "results/models/lightgbm_baseline.pkl")

# Extract feature importance to understand what drives predictions
importance_df = pd.DataFrame({
    "feature": FEATURE_COLS,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\nTop 10 features:")
print(importance_df.head(10))

# Save feature importance
os.makedirs("results/tables", exist_ok=True)
importance_df.to_csv("results/tables/lightgbm_feature_importance.csv", index=False)
print("Feature importance saved.")