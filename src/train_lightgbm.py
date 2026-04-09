# https://www.geeksforgeeks.org/machine-learning/regression-using-lightgbm/
# https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html
# Ke et al. (2017) - LightGBM: A Highly Efficient Gradient Boosting Decision Tree

import os
import csv
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from load_data import load_csv

train = load_csv("esa_train.csv")
test = load_csv("esa_test.csv")

# Drop columns with too many missing values (>50%)
# Can't reliably fill that much missing data without introducing bias
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
# which is data leakage, not real performance
# Removing them gives an honest R² = 0.85
# https://www.geeksforgeeks.org/machine-learning/what-is-data-leakage/
LEAKY_COLS = ["max_risk_estimate", "max_risk_scaling"]

FEATURE_COLS = [c for c in train.columns if c not in [TARGET] + ID_COLS + LEAKY_COLS]

X_train = train[FEATURE_COLS]
y_train = train[TARGET]
X_test = test[FEATURE_COLS]
y_test = test[TARGET]

print(f"\nTraining LightGBM on {len(FEATURE_COLS)} features...")

# LightGBM configuration
# n_estimators=200: build 200 decision trees (enough for this dataset size)
# learning_rate=0.05: conservative learning rate to prevent overfitting
# num_leaves=31: default, controls tree complexity
# random_state=42: ensures results are reproducible
# https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html
model = LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42
)

model.fit(X_train, y_train)
preds = model.predict(X_test)

# Evaluate on final CDM per event only (smallest time_to_tca)
# LightGBM predicts a risk value for every row, but we only care about
# the prediction for the last CDM per event - the one closest to TCA
# This matches BiLSTM and physics baseline which also predict one value per event
# Ensures fair comparison across all models
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html
test = test.copy()
test['preds'] = preds
test_final = test.sort_values('time_to_tca').groupby('event_id').first().reset_index()
y_test_final = test_final[TARGET]
preds_final = test_final['preds']

rmse = mean_squared_error(y_test_final, preds_final) ** 0.5
mae = mean_absolute_error(y_test_final, preds_final)
r2 = r2_score(y_test_final, preds_final)

print("\n=== RESULTS (per event, final CDM only) ===")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"R²:   {r2:.4f}")

# Save the trained model
os.makedirs("results/models", exist_ok=True)
joblib.dump(model, "results/models/lightgbm_baseline.pkl")
print("Model saved to results/models/lightgbm_baseline.pkl")

# Append to shared metrics file so all models can be compared in one place
os.makedirs("results", exist_ok=True)
metrics_path = "results/model_metrics.csv"
file_exists = os.path.isfile(metrics_path)
with open(metrics_path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["model", "RMSE", "MAE", "R2"])
    if not file_exists:
        writer.writeheader()
    writer.writerow({"model": "LightGBM", "RMSE": rmse, "MAE": mae, "R2": r2})
print(f"Metrics appended to {metrics_path}")

# Save per-event predictions for dashboard and evaluation framework
os.makedirs("results/predictions", exist_ok=True)
pred_out = pd.DataFrame({
    'event_id': test_final['event_id'],
    'actual_risk': y_test_final.values,
    'predicted_risk': preds_final.values
})
pred_out.to_csv("results/predictions/lightgbm_predictions.csv", index=False)
print("Predictions saved to results/predictions/lightgbm_predictions.csv")

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
print("Feature importance saved to results/tables/lightgbm_feature_importance.csv")