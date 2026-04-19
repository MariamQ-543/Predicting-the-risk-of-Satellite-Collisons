# Model evaluation and comparison script
# loads saved predictions from all 5 models and generates:
# - bar charts comparing R2 and RMSE across models
# - predicted vs actual scatter plots for each model
# - residual plots for each model
# - formatted comparison table
#
# https://www.geeksforgeeks.org/matplotlib-tutorial/
# https://www.geeksforgeeks.org/python-seaborn-tutorial/

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# output directories
os.makedirs("results/plots", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)

# consistent colour for each model across all plots
MODEL_COLOURS = {
    'PhysicsBaseline': '#6c757d',
    'LightGBM':        '#2196F3',
    'BiLSTM':          '#FF9800',
    'MLP':             '#9C27B0',
    'GNN':             '#4CAF50',
    'Transformer':     '#F44336'
}

MODEL_ORDER = ['PhysicsBaseline', 'LightGBM', 'BiLSTM', 'Transformer', 'MLP', 'GNN']

# prediction file paths - one per model
PRED_FILES = {
    'PhysicsBaseline': 'results/predictions/physics_baseline_predictions.csv',
    'LightGBM':        'results/predictions/lightgbm_predictions.csv',
    'BiLSTM':          'results/predictions/lstm_predictions.csv',
    'Transformer':     'results/predictions/transformer_predictions.csv',
    'MLP':             'results/predictions/mlp_predictions.csv',
    'GNN':             'results/predictions/gnn_predictions.csv',
}

# load predictions
print("Loading predictions...")
predictions = {}
metrics = {}

for model, path in PRED_FILES.items():
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found - skipping {model}")
        continue

    df = pd.read_csv(path)
    predictions[model] = df

    y_true = df['actual_risk'].values
    y_pred = df['predicted_risk'].values

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    metrics[model] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
    print(f"  {model}: R2={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")

# build metrics dataframe in fixed order
available = [m for m in MODEL_ORDER if m in metrics]
metrics_df = pd.DataFrame(metrics).T.loc[available]
metrics_df.index.name = 'Model'
metrics_df = metrics_df.reset_index()

# save formatted table
metrics_df.to_csv("results/tables/model_comparison_summary.csv", index=False)
print("\nSaved results/tables/model_comparison_summary.csv")

# plot 1: R2 bar chart
fig, ax = plt.subplots(figsize=(9, 5))

colours = [MODEL_COLOURS.get(m, '#333') for m in metrics_df['Model']]
bars = ax.bar(metrics_df['Model'], metrics_df['R2'], color=colours,
              edgecolor='white', linewidth=0.8, width=0.6)

# add value labels on bars
for bar, val in zip(bars, metrics_df['R2']):
    ypos = bar.get_height() + 0.02 if val >= 0 else bar.get_height() - 0.08
    ax.text(bar.get_x() + bar.get_width() / 2, ypos,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
ax.set_ylabel('R² Score', fontsize=11)
ax.set_title('Model Comparison — R² Score (higher is better)', fontsize=12, pad=12)
ax.set_ylim(min(metrics_df['R2'].min() - 0.3, -0.5), 1.05)
ax.tick_params(axis='x', labelsize=9)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig('results/plots/model_comparison_r2.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved results/plots/model_comparison_r2.png")

# plot 2: RMSE bar chart
fig, ax = plt.subplots(figsize=(9, 5))

bars = ax.bar(metrics_df['Model'], metrics_df['RMSE'], color=colours,
              edgecolor='white', linewidth=0.8, width=0.6)

for bar, val in zip(bars, metrics_df['RMSE']):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
            f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('RMSE (log risk units)', fontsize=11)
ax.set_title('Model Comparison — RMSE (lower is better)', fontsize=12, pad=12)
ax.tick_params(axis='x', labelsize=9)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig('results/plots/model_comparison_rmse.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved results/plots/model_comparison_rmse.png")

# plot 3: predicted vs actual for each model
n = len(available)
cols = 3
rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
axes = axes.flatten()

for i, model in enumerate(available):
    df   = predictions[model]
    ax   = axes[i]
    col  = MODEL_COLOURS.get(model, '#333')

    y_true = df['actual_risk'].values
    y_pred = df['predicted_risk'].values
    r2     = metrics[model]['R2']

    ax.scatter(y_true, y_pred, alpha=0.25, s=6, color=col)

    # perfect prediction line
    lim = [min(y_true.min(), y_pred.min()) - 1,
           max(y_true.max(), y_pred.max()) + 1]
    ax.plot(lim, lim, 'k--', linewidth=1, alpha=0.6, label='Perfect')

    ax.set_xlabel('Actual Risk', fontsize=9)
    ax.set_ylabel('Predicted Risk', fontsize=9)
    ax.set_title(f'{model}  (R²={r2:.3f})', fontsize=10)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.grid(alpha=0.2, linestyle='--')
    ax.spines[['top', 'right']].set_visible(False)

# hide unused subplots
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle('Predicted vs Actual Risk — All Models', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('results/plots/predicted_vs_actual_all.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved results/plots/predicted_vs_actual_all.png")

# plot 4: residual plots
fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
axes = axes.flatten()

for i, model in enumerate(available):
    df   = predictions[model]
    ax   = axes[i]
    col  = MODEL_COLOURS.get(model, '#333')

    y_true    = df['actual_risk'].values
    y_pred    = df['predicted_risk'].values
    residuals = y_true - y_pred

    ax.scatter(y_pred, residuals, alpha=0.25, s=6, color=col)
    ax.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.6)

    ax.set_xlabel('Predicted Risk', fontsize=9)
    ax.set_ylabel('Residual (Actual - Predicted)', fontsize=9)
    ax.set_title(f'{model} — Residuals', fontsize=10)
    ax.grid(alpha=0.2, linestyle='--')
    ax.spines[['top', 'right']].set_visible(False)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle('Residual Plots — All Models', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('results/plots/residuals_all.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved results/plots/residuals_all.png")

# plot 5: summary heatmap 
# normalise metrics so they're on the same scale for the heatmap
# R2: higher is better, RMSE/MAE: lower is better (invert so higher = better)
heatmap_df = metrics_df.set_index('Model')[['R2', 'RMSE', 'MAE']].copy()
heatmap_df['RMSE'] = -heatmap_df['RMSE']  # invert so higher = better
heatmap_df['MAE']  = -heatmap_df['MAE']

# min-max normalise each column to 0-1
for col in heatmap_df.columns:
    col_min = heatmap_df[col].min()
    col_max = heatmap_df[col].max()
    if col_max != col_min:
        heatmap_df[col] = (heatmap_df[col] - col_min) / (col_max - col_min)

fig, ax = plt.subplots(figsize=(7, 4))
sns.heatmap(heatmap_df, annot=True, fmt='.2f', cmap='RdYlGn',
            vmin=0, vmax=1, ax=ax, linewidths=0.5,
            cbar_kws={'label': 'Normalised score (higher = better)'})

ax.set_title('Model Performance Heatmap\n(normalised, higher = better for all metrics)',
             fontsize=11, pad=10)
ax.set_xlabel('')
ax.tick_params(axis='x', labelsize=10)
ax.tick_params(axis='y', labelsize=9, rotation=0)

plt.tight_layout()
plt.savefig('results/plots/model_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved results/plots/model_heatmap.png")

# final summary
print("\n" + "="*55)
print("FINAL MODEL COMPARISON")
print("="*55)
print(metrics_df.to_string(index=False))
print("\nAll plots saved to results/plots/")
print("Summary table saved to results/tables/model_comparison_summary.csv")