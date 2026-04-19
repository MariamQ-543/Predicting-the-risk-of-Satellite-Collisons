# Attention weight visualisation for BiLSTM model
# the BiLSTM uses an attention mechanism to learn which CDMs in the sequence
# matter most when predicting the final collision risk
# this script extracts those attention weights and visualises them
# so we can see what the model is "paying attention to"
# https://www.geeksforgeeks.org/artificial-intelligence/ml-attention-mechanism/
# https://machinelearningmastery.com/the-attention-mechanism-from-scratch/
 

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from load_data import load_csv

os.makedirs("results/plots", exist_ok=True)

TARGET  = "risk"
MAX_LEN = 23

# same features used during BiLSTM training
# selected based on LightGBM feature importance analysis
FEATURE_COLS = [
    'relative_position_r', 'miss_distance', 'relative_speed',
    'relative_velocity_t', 'relative_position_t', 'relative_position_n',
    'c_sigma_t', 'c_sigma_r', 'c_sigma_tdot', 'c_sigma_n',
    'mahalanobis_distance', 'c_position_covariance_det', 'c_ct_r',
    't_sigma_r', 't_sigma_t',
    'time_to_tca', 'geocentric_latitude', 'azimuth', 'elevation',
    'F10', 'F3M', 'AP'
]

def prepare_data():
    # load and sort CDMs ascending so sequence goes earliest to closest TCA
    test  = load_csv("esa_test.csv")
    test  = test.sort_values(['event_id', 'time_to_tca'], ascending=[True, True])
    test['time_delta'] = test.groupby('event_id')['time_to_tca'].diff().fillna(0).abs()

    train = load_csv("esa_train.csv")
    train = train.sort_values(['event_id', 'time_to_tca'], ascending=[True, True])
    train['time_delta'] = train.groupby('event_id')['time_to_tca'].diff().fillna(0).abs()

    current_features = FEATURE_COLS + ['time_delta']

    # fit on train only, apply to test - same as during training
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    train[current_features] = scaler.fit_transform(train[current_features].fillna(train[current_features].median()))
    test[current_features]  = scaler.transform(test[current_features].fillna(test[current_features].median()))

    return test, current_features

def build_sequences(df, feature_cols):
    # build padded sequences - same logic as training
    # pad at start so the most recent CDMs are always at the end
    x_list, y_list, tca_list, id_list = [], [], [], []

    for event_id, group in df.groupby('event_id'):
        group    = group.sort_values('time_to_tca', ascending=True)
        features = group[feature_cols].values
        target   = group.iloc[-1][TARGET]
        tca_vals = group['time_to_tca'].values

        if len(features) < MAX_LEN:
            pad      = np.zeros((MAX_LEN - len(features), len(feature_cols)))
            features = np.vstack([pad, features])
            tca_pad  = np.full(MAX_LEN - len(tca_vals), np.nan)
            tca_vals = np.concatenate([tca_pad, tca_vals])
        else:
            features = features[:MAX_LEN]
            tca_vals = tca_vals[:MAX_LEN]

        x_list.append(features)
        y_list.append(target)
        tca_list.append(tca_vals)
        id_list.append(event_id)

    return np.array(x_list), np.array(y_list), np.array(tca_list), np.array(id_list)

print("Loading data...")
test_df, features = prepare_data()
X_test, y_test, tca_seqs, event_ids = build_sequences(test_df, features)

print("Loading BiLSTM model...")
keras.config.enable_unsafe_deserialization()
model = keras.models.load_model("results/models/best_bilstm.keras")
model.summary()

# find the attention layer in the model
# look for a Dense layer that outputs shape (batch, timesteps, 1)
# that is the layer that computes one attention score per timestep
attention_layer = None
for layer in model.layers:
    if 'attention' in layer.name.lower() or 'dense' in layer.name.lower():
        if len(layer.output.shape) == 3 and layer.output.shape[-1] == 1:
            attention_layer = layer
            break

# fallback in case layer naming is different
if attention_layer is None:
    for layer in model.layers:
        if len(layer.output.shape) == 3:
            attention_layer = layer

print(f"Attention layer found: {attention_layer.name} shape: {attention_layer.output.shape}")

# sub-model to extract attention weights alongside predictions
attention_model = keras.Model(
    inputs=model.input,
    outputs=[model.output, attention_layer.output]
)

print("Computing attention weights for all test events...")
predictions, attention_weights = attention_model.predict(X_test, verbose=0)
predictions = predictions.ravel()

# squeeze to (n_events, timesteps)
if len(attention_weights.shape) == 3:
    attention_weights = attention_weights[:, :, 0]

# normalise so weights per event sum to 1 - makes them comparable across events
attention_weights = np.abs(attention_weights)
row_sums = attention_weights.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
attention_weights = attention_weights / row_sums

# x axis labels - 0 is earliest CDM, 22 is closest to TCA
timestep_labels = [f"t-{MAX_LEN - i - 1}" if i < MAX_LEN - 1 else "TCA" for i in range(MAX_LEN)]

# plot 1: average attention across all test events
# shows which CDM positions the model generally focuses on
# if the model is using temporal context it should peak near the end (close to TCA)
mean_attention = attention_weights.mean(axis=0)

fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(range(MAX_LEN), mean_attention, color='steelblue', alpha=0.8, edgecolor='white')
ax.set_xlabel('CDM Position (left = earliest, right = closest to TCA)', fontsize=10)
ax.set_ylabel('Mean Attention Weight', fontsize=10)
ax.set_title('BiLSTM Average Attention Weights Across All Test Events\n'
             'Higher = model focused more on that CDM position', fontsize=11)
ax.set_xticks(range(0, MAX_LEN, 2))
ax.set_xticklabels([timestep_labels[i] for i in range(0, MAX_LEN, 2)], fontsize=8, rotation=45)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig("results/plots/attention_mean.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved results/plots/attention_mean.png")

# plot 2: attention heatmap for a sample of events
# each row is one event, each column is a CDM position
# brighter colour = higher attention = model focused more on that CDM
# pick 5 high risk, 5 medium risk, 5 low risk to compare patterns
n_samples  = 15
sorted_idx = np.argsort(y_test)
high_idx   = sorted_idx[:5]
mid_idx    = sorted_idx[len(sorted_idx)//2 - 2: len(sorted_idx)//2 + 3]
low_idx    = sorted_idx[-5:]
sample_idx = np.concatenate([high_idx, mid_idx, low_idx])

sample_attn = attention_weights[sample_idx]
sample_risk = y_test[sample_idx]
sample_pred = predictions[sample_idx]

row_labels = [f"Event {i+1} | actual={r:.1f} pred={p:.1f}"
              for i, (r, p) in enumerate(zip(sample_risk, sample_pred))]

fig, ax = plt.subplots(figsize=(14, 6))
im = ax.imshow(sample_attn, aspect='auto', cmap='YlOrRd', interpolation='nearest')
plt.colorbar(im, ax=ax, label='Attention Weight')
ax.set_xlabel('CDM Position (left = earliest, right = closest to TCA)', fontsize=10)
ax.set_ylabel('Event', fontsize=10)
ax.set_title('BiLSTM Attention Heatmap for Sample Events\n'
             'Top 5 = highest risk, Middle 5 = medium risk, Bottom 5 = lowest risk',
             fontsize=11)
ax.set_xticks(range(0, MAX_LEN, 2))
ax.set_xticklabels([timestep_labels[i] for i in range(0, MAX_LEN, 2)], fontsize=8, rotation=45)
ax.set_yticks(range(n_samples))
ax.set_yticklabels(row_labels, fontsize=7)
plt.tight_layout()
plt.savefig("results/plots/attention_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved results/plots/attention_heatmap.png")

# plot 3: attention weights for the 5 highest risk events individually
# shows exactly which CDMs each high risk event focused on
# cdm index 0 = first CDM received, last index = closest to TCA
fig, axes = plt.subplots(1, 5, figsize=(16, 4), sharey=True)
for i, idx in enumerate(high_idx):
    tca_vals = tca_seqs[idx]
    attn     = attention_weights[idx]
    valid    = ~np.isnan(tca_vals)  # exclude padded positions

    axes[i].bar(range(sum(valid)), attn[valid], color='crimson', alpha=0.7)
    axes[i].set_title(f"Risk={y_test[idx]:.1f}\nPred={predictions[idx]:.1f}", fontsize=8)
    axes[i].set_xlabel('CDM index', fontsize=7)
    if i == 0:
        axes[i].set_ylabel('Attention Weight', fontsize=8)
    axes[i].grid(axis='y', alpha=0.3, linestyle='--')
    axes[i].spines[['top', 'right']].set_visible(False)

fig.suptitle('BiLSTM Attention Weights for 5 Highest Risk Events', fontsize=11)
plt.subplots_adjust(top=0.85)
plt.savefig("results/plots/attention_high_risk.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved results/plots/attention_high_risk.png")

print("\nAll attention plots saved to results/plots/")
print(f"Mean attention peaks at timestep: {np.argmax(mean_attention)} "
      f"(0=earliest CDM, {MAX_LEN-1}=closest to TCA)")