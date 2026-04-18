# SHAP analysis for LightGBM model
# explains why the model made each prediction by showing
# how much each feature pushed the risk prediction up or down
# uses TreeExplainer which is fast and exact for tree based models
#
# https://www.geeksforgeeks.org/shap-values-in-machine-learning/
# https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html
# https://www.datacamp.com/tutorial/introduction-to-shap-values-machine-learning-interpretability

import os
import shap
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from load_data import load_csv

os.makedirs("results/plots", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)

TARGET = "risk"

print("Loading data and model...")
train = load_csv("esa_train.csv")
test  = load_csv("esa_test.csv")

# get final CDM per event - same evaluation as all other models
test_final  = test.sort_values('time_to_tca').groupby('event_id').first().reset_index()

# load trained lightgbm model
model = joblib.load("results/models/lightgbm_baseline.pkl")

# encode c_object_type to number before selecting features
# same way it was encoded during training
test_final['c_object_type'] = test_final['c_object_type'].astype('category').cat.codes

# use exact feature names the model was trained on - no guessing
feature_cols = model.feature_name_

X_test = test_final[feature_cols].copy()
X_test = X_test.fillna(X_test.median(numeric_only=True))
y_test = test_final[TARGET].values

print(f"Features: {len(feature_cols)}")
print(f"Test events: {len(X_test)}")

# TreeExplainer is the fast exact explainer for tree based models
# much faster than KernelExplainer and gives exact SHAP values
print("Computing SHAP values...")
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
print("SHAP values computed. Generating plots...")

# plot 1: summary bar plot
# shows mean absolute SHAP value per feature - overall feature importance
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_values, X_test,
    plot_type="bar",
    max_display=20,
    show=False
)
plt.title("LightGBM — Top 20 Features by Mean |SHAP Value|", fontsize=12)
plt.tight_layout()
plt.savefig("results/plots/shap_summary_bar.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved results/plots/shap_summary_bar.png")

# plot 2: beeswarm plot
# shows how each feature affects individual predictions
# colour = feature value (red = high, blue = low)
# x position = SHAP value (right = pushes risk up, left = pushes risk down)
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_values, X_test,
    plot_type="dot",
    max_display=20,
    show=False
)
plt.title("LightGBM — SHAP Beeswarm Plot (Top 20 Features)", fontsize=12)
plt.tight_layout()
plt.savefig("results/plots/shap_beeswarm.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved results/plots/shap_beeswarm.png")

# plot 3: top 3 feature dependence plots
# shows how the top 3 most important features relate to their SHAP values
# each dot is one event - shows non-linear relationships the model learned
shap_df       = pd.DataFrame(np.abs(shap_values), columns=feature_cols)
top3_features = shap_df.mean().nlargest(3).index.tolist()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, feat in enumerate(top3_features):
    feat_idx = list(feature_cols).index(feat)
    axes[i].scatter(
        X_test[feat],
        shap_values[:, feat_idx],
        alpha=0.3, s=8, c=shap_values[:, feat_idx],
        cmap='RdYlGn_r'
    )
    axes[i].axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    axes[i].set_xlabel(feat, fontsize=9)
    axes[i].set_ylabel('SHAP value', fontsize=9)
    axes[i].set_title(feat, fontsize=10)
    axes[i].grid(alpha=0.2, linestyle='--')
    axes[i].spines[['top', 'right']].set_visible(False)

fig.suptitle('LightGBM — Top 3 Feature Dependence Plots', fontsize=12)
plt.subplots_adjust(top=0.88)
plt.savefig("results/plots/shap_dependence.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved results/plots/shap_dependence.png")

# plot 4: waterfall for highest risk event
# shows exactly how the model built up its prediction for one specific event
# each bar shows how much one feature pushed the prediction up or down
predictions   = model.predict(X_test)
high_risk_idx = np.argmin(y_test)

plt.figure(figsize=(10, 6))
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[high_risk_idx],
        base_values=explainer.expected_value,
        data=X_test.iloc[high_risk_idx],
        feature_names=list(feature_cols)
    ),
    max_display=15,
    show=False
)
plt.title(
    f"LightGBM — Prediction Breakdown for Highest Risk Event\n"
    f"Actual: {y_test[high_risk_idx]:.3f}  Predicted: {predictions[high_risk_idx]:.3f}",
    fontsize=11
)
plt.tight_layout()
plt.savefig("results/plots/shap_waterfall_high_risk.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved results/plots/shap_waterfall_high_risk.png")

# save feature importance table
shap_summary = pd.DataFrame({
    'feature': feature_cols,
    'mean_abs_shap': np.abs(shap_values).mean(axis=0)
}).sort_values('mean_abs_shap', ascending=False)

shap_summary.to_csv("results/tables/shap_feature_importance.csv", index=False)
print("Saved results/tables/shap_feature_importance.csv")

print("\nTop 10 most important features by SHAP:")
print(shap_summary.head(10).to_string(index=False))
print("\nAll SHAP plots saved to results/plots/")