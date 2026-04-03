"""
Physics-Inspired Baseline for Collision Risk Prediction

Uses miss_distance and relative_speed from ESA dataset to create
a simple baseline for comparing against ML models.

- ESA Collision Avoidance Challenge: https://kelvins.esa.int/collision-avoidance-challenge/data/
- CCSDS Conjunction Data Message Standard (miss_distance definition): https://public.ccsds.org/Pubs/508x0b1e2c2.pdf
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from load_data import load_csv
import os

print("Loading data...")
train = load_csv("esa_train.csv")
test = load_csv("esa_test.csv")

# Simple risk function based on distance and speed
def estimate_risk(miss_distance, relative_speed):
# Simple heuristic:
# smaller miss_distance suggests higher collision risk
# higher relative_speed suggests a more severe event
# log scale is used to roughly align with ESA risk values
    safe_distance = np.maximum(miss_distance, 1.0)
    safe_speed = np.maximum(relative_speed, 1.0)
    
    risk = -2.0 * np.log10(safe_distance) + 0.5 * np.log10(safe_speed)
    return risk

# Calculate predictions
print("Calculating predictions...")
train['physics_pred'] = estimate_risk(train['miss_distance'], train['relative_speed'])
test['physics_pred'] = estimate_risk(test['miss_distance'], test['relative_speed'])

# Evaluate on test set
y_test = test['risk']
y_pred = test['physics_pred']

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nPhysics Baseline Results:")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"R²:   {r2:.4f}")

# Compare to LightGBM 
lightgbm_r2 = 0.8468
print(f"\nComparison:")
print(f"Physics baseline: R² = {r2:.4f}")
print(f"LightGBM:        R² = {lightgbm_r2:.4f}")

# Save predictions
os.makedirs("results/predictions", exist_ok=True)
test[['event_id', 'risk', 'physics_pred']].to_csv(
    "results/predictions/physics_baseline_predictions.csv", 
    index=False
)

# Plot results
os.makedirs("results/plots", exist_ok=True)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.3, s=5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Risk')
plt.ylabel('Predicted Risk')
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
print("\nSaved results to results/ folder")