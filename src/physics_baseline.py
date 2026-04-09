"""
Physics Baseline using ESA's max_risk_estimate

Uses ESA's provided max_risk_estimate as the physics baseline for comparison.
This value is derived from operational conjunction assessment processes using
orbital mechanics - it represents what a physics-based system would output.

Note: max_risk_estimate is excluded from LightGBM and BiLSTM as a leaky feature,
but it is valid to use here as the physics baseline itself - this is what we are
comparing our ML models against.

Uriot et al. (2021): Spacecraft collision avoidance challenge:
https://link.springer.com/content/pdf/10.1007/s42064-021-0101-5.pdf
ESA Challenge Data: https://kelvins.esa.int/collision-avoidance-challenge/data/
"""

import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from load_data import load_csv

print("Loading data...")
train = load_csv("esa_train.csv")
test = load_csv("esa_test.csv")

# Evaluate on final CDM per event only (smallest time_to_tca)
# The ESA dataset description says the target is the risk at the last CDM
# We filter to just that row per event so this baseline is evaluated
# on the same unit as LightGBM and BiLSTM - one prediction per event
# https://kelvins.esa.int/collision-avoidance-challenge/data/
test_final = test.sort_values('time_to_tca').groupby('event_id').first().reset_index()

y_test = test_final['risk']
y_pred = test_final['max_risk_estimate']

print(f"\nEvaluating on {len(test_final)} events (one per event, final CDM)...")

# Calculate metrics
rmse = mean_squared_error(y_test, y_pred) ** 0.5
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n=== Physics Baseline (ESA max_risk_estimate) ===")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"R²:   {r2:.4f}")

# Append to shared metrics file so all models can be compared in one place
os.makedirs("results", exist_ok=True)
metrics_path = "results/model_metrics.csv"
file_exists = os.path.isfile(metrics_path)
with open(metrics_path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["model", "RMSE", "MAE", "R2"])
    if not file_exists:
        writer.writeheader()
    writer.writerow({"model": "PhysicsBaseline", "RMSE": rmse, "MAE": mae, "R2": r2})
print(f"Metrics appended to {metrics_path}")

# Save per-event predictions for dashboard and evaluation framework
os.makedirs("results/predictions", exist_ok=True)
pred_out = pd.DataFrame({
    'event_id': test_final['event_id'],
    'actual_risk': y_test.values,
    'predicted_risk': y_pred.values
})
pred_out.to_csv("results/predictions/physics_baseline_predictions.csv", index=False)
print("Predictions saved to results/predictions/physics_baseline_predictions.csv")

# Plot predicted vs actual and residuals
os.makedirs("results/plots", exist_ok=True)
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.3, s=5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Risk')
plt.ylabel('Predicted Risk (max_risk_estimate)')
plt.title(f'Physics Baseline (R² = {r2:.4f})')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.3, s=5)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Risk')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/plots/physics_baseline_evaluation.png', dpi=300)
print("Plot saved to results/plots/physics_baseline_evaluation.png")