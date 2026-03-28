import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from load_data import load_csv

train = load_csv("esa_train.csv")
test = load_csv("esa_test.csv")

# Drop columns where more than 50% of values are missing
threshold = 0.5
missing_frac = train.isnull().mean()
cols_to_drop = missing_frac[missing_frac > threshold].index.tolist()
train = train.drop(columns=cols_to_drop)
test = test.drop(columns=cols_to_drop)

# Encode the only non-numeric column
# Categories are fitted on train only then applied to test to keep mapping consistent
if "c_object_type" in train.columns:
    categories = train["c_object_type"].astype("category").cat.categories
    train["c_object_type"] = pd.Categorical(train["c_object_type"], categories=categories).codes
    test["c_object_type"] = pd.Categorical(test["c_object_type"], categories=categories).codes

# Fill missing values using training medians
# Test set uses train medians to avoid data leakage
train_medians = train.median(numeric_only=True)
train = train.fillna(train_medians)
test = test.fillna(train_medians)

TARGET = "risk"
ID_COLS = ["event_id", "mission_id"]

# max_risk_estimate and max_risk_scaling are removed as they are direct proxies
# for the target variable, including them would cause target leakage and
# give unrealistically high scores that wouldn't hold in the real world
# Reference: https://www.geeksforgeeks.org/machine-learning/what-is-data-leakage/
LEAKY_COLS = ["max_risk_estimate", "max_risk_scaling"]

FEATURE_COLS = [c for c in train.columns if c not in [TARGET] + ID_COLS + LEAKY_COLS]

X_train = train[FEATURE_COLS]
y_train = train[TARGET]
X_test = test[FEATURE_COLS]
y_test = test[TARGET]

print("\nTraining LightGBM...")
print(f"Number of features: {len(FEATURE_COLS)}")

model = LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42
)

model.fit(X_train, y_train)

preds = model.predict(X_test)

rmse = mean_squared_error(y_test, preds) ** 0.5
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print("\n=== LIGHTGBM RESULTS ===")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"R²:   {r2:.4f}")

joblib.dump(model, "results/models/lightgbm_baseline.pkl")
print("\nModel saved to results/models/lightgbm_baseline.pkl")