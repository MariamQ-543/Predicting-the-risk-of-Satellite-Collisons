"""
BiLSTM with Attention
- https://machinelearningmastery.com/how-to-reshape-data-for-long-short-term-memory-networks-in-keras/
- https://analyticsvidhya.com/blog/2023/06/time-series-forecasting-using-attention-mechanism/
- https://www.tensorflow.org/api_docs/python/tf/keras/layers/Bidirectional
- https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
"""

import os
import csv
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from load_data import load_csv

# Settings
TARGET = "risk"
MAX_LEN = 23

# Features selected based on LightGBM feature importance analysis
# Using top predictors confirmed by the tree-based model to ensure
# the LSTM has access to the most informative signals
# Includes all three modalities present in the ESA dataset:
# - Conjunction geometry (relative position, velocity, miss distance)
# - Uncertainty / covariance terms (sigma values, Mahalanobis distance)
# - Space weather indices (F10, F3M, AP) - multimodal component
FEATURE_COLS = [
    # Conjunction geometry - top LightGBM features
    'relative_position_r', 'miss_distance', 'relative_speed',
    'relative_velocity_t', 'relative_position_t', 'relative_position_n',
    # Uncertainty / covariance - how well we know the orbits
    'c_sigma_t', 'c_sigma_r', 'c_sigma_tdot', 'c_sigma_n',
    'mahalanobis_distance', 'c_position_covariance_det', 'c_ct_r',
    't_sigma_r', 't_sigma_t',
    # Orbital context
    'time_to_tca', 'geocentric_latitude', 'azimuth', 'elevation',
    # Space weather indices - atmospheric drag affects orbit uncertainty
    # F10: solar radio flux, F3M: 81-day mean, AP: geomagnetic activity
    'F10', 'F3M', 'AP'
]

def prepare_data():
    """Load data, add time_delta, fill missing values, scale features."""
    train = load_csv("esa_train.csv")
    test = load_csv("esa_test.csv")

    # Sort ascending so each event goes from earliest CDM -> closest to TCA
    # This is the correct temporal order for LSTM to learn from
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_values.html
    train = train.sort_values(['event_id', 'time_to_tca'], ascending=[True, True])
    test = test.sort_values(['event_id', 'time_to_tca'], ascending=[True, True])

    # Time gap between consecutive CDMs within each event
    # Gives the model information about how frequently updates arrived
    train['time_delta'] = train.groupby('event_id')['time_to_tca'].diff().fillna(0).abs()
    test['time_delta'] = test.groupby('event_id')['time_to_tca'].diff().fillna(0).abs()

    current_features = FEATURE_COLS + ['time_delta']

    # Fill missing values using train medians only
    # Prevents test data leaking into preprocessing
    # https://www.geeksforgeeks.org/machine-learning/what-is-data-leakage/
    train_medians = train.median(numeric_only=True)
    train = train.fillna(train_medians)
    test = test.fillna(train_medians)

    # Scale features - LSTM is sensitive to feature magnitude
    # Fit scaler on train only, transform both
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    scaler = StandardScaler()
    train[current_features] = scaler.fit_transform(train[current_features])
    test[current_features] = scaler.transform(test[current_features])

    return train, test, current_features

def create_sequences(df, feature_cols, target_col):
    """
    Convert each conjunction event into a fixed-length sequence.

    Each event has multiple CDMs over time. We treat these as timesteps.
    Target is the risk value at the final CDM (closest to TCA).
    This matches what LightGBM and physics baseline also predict,
    ensuring fair comparison across all models.

    Padding is added at the START so that the most recent CDMs
    (closest to TCA) are always at the end of the sequence.
    """
    x_list, y_list, id_list = [], [], []

    for event_id, group in df.groupby('event_id'):
        # Sort ascending: earliest CDM first, closest to TCA last
        group = group.sort_values('time_to_tca', ascending=True)
        features = group[feature_cols].values

        # Target = risk at the CDM closest to TCA (last row after ascending sort)
        # This is the final risk value the ESA competition asks us to predict
        # https://kelvins.esa.int/collision-avoidance-challenge/data/
        target_val = group.iloc[-1][target_col]

        # Pad at the START so recent CDMs are always at position [-1]
        # Trim from the start if sequence is too long (keeps most recent CDMs)
        if len(features) < MAX_LEN:
            pad = np.zeros((MAX_LEN - len(features), len(feature_cols)))
            features = np.vstack([pad, features])
        else:
            features = features[:MAX_LEN]

        x_list.append(features)
        y_list.append(target_val)
        id_list.append(event_id)

    return np.array(x_list), np.array(y_list), np.array(id_list)

def build_model(input_shape):
    """
    BiLSTM + Attention model.

    Bidirectional LSTM reads the CDM sequence in both directions,
    capturing both how risk evolved from the start and how it
    approaches TCA from the end.
    Schuster & Paliwal (1997) - Bidirectional Recurrent Neural Networks

    Attention layer learns which CDMs in the sequence matter most
    for predicting the final risk.
    Bahdanau et al. (2015) - Neural Machine Translation by Jointly
    Learning to Align and Translate
    """
    inputs = layers.Input(shape=input_shape)

    # Masking ignores the zero-padded timesteps at the start
    # so the model only learns from real CDM observations
    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Masking
    masking = layers.Masking(mask_value=0.0)(inputs)

    # BiLSTM layer 1: 64 units each direction = 128 total output features
    lstm_1 = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(masking)

    # BiLSTM layer 2: 32 units each direction = 64 total output features
    lstm_2 = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(lstm_1)

    # Attention mechanism
    # Learns a weight for each timestep - which CDMs are most informative
    att_weights = layers.Dense(1, activation='tanh')(lstm_2)
    att_weights = layers.Flatten()(att_weights)
    att_weights = layers.Activation('softmax', name='attention_weights')(att_weights)

    # Get output dim dynamically so this doesn't break if LSTM units change
    # lstm_2 output = 32 units * 2 directions = 64
    lstm_out_dim = lstm_2.shape[-1]
    att_weights = layers.RepeatVector(lstm_out_dim)(att_weights)
    att_weights = layers.Permute([2, 1])(att_weights)

    # Multiply attention weights with LSTM output then sum across timesteps
    # Result: single vector summarising the whole sequence
    weighted_seq = layers.Multiply()([lstm_2, att_weights])
    summary_vec = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(weighted_seq)

    # Final prediction layers
    dense = layers.Dense(32, activation='relu')(summary_vec)
    outputs = layers.Dense(1, activation='linear')(dense)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

if __name__ == "__main__":
    # 1. prepare data
    train_df, test_df, features = prepare_data()

    # 2. Split by event_id to prevent data leakage between train and validation
    # Splitting by row would let the model see future CDMs from the same event
    # https://www.geeksforgeeks.org/machine-learning/what-is-data-leakage/
    unique_events = train_df['event_id'].unique()
    train_ids, val_ids = train_test_split(unique_events, test_size=0.1, random_state=42)

    val_data = train_df[train_df['event_id'].isin(val_ids)]
    train_data = train_df[train_df['event_id'].isin(train_ids)]

    X_train, y_train, _ = create_sequences(train_data, features, TARGET)
    X_val, y_val, _ = create_sequences(val_data, features, TARGET)
    X_test, y_test, test_ids = create_sequences(test_df, features, TARGET)

    print(f"Shapes (Train/Val/Test): {X_train.shape}, {X_val.shape}, {X_test.shape}")

    # 3. train model
    model = build_model((MAX_LEN, len(features)))

    os.makedirs("results/models", exist_ok=True)
    my_callbacks = [
        # patience=8: wait 8 epochs without improvement before stopping
        # more patience than before since loss was still slowly improving
        callbacks.EarlyStopping(patience=8, restore_best_weights=True),
        # Save the best model checkpoint during training
        callbacks.ModelCheckpoint("results/models/best_bilstm.h5", save_best_only=True)
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,  # increased from 30 - model needs more time to converge
        batch_size=32,
        callbacks=my_callbacks
    )

    # 4. evaluate on test set
    # X_test contains one sequence per event, y_test is the final risk per event
    # This is the same evaluation unit as LightGBM and physics baseline
    preds = model.predict(X_test).ravel()
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"\nFinal Test Results = RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

    # 5. save results
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/predictions", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Shared metrics file so all models can be compared in one place
    metrics_path = "results/model_metrics.csv"
    file_exists = os.path.isfile(metrics_path)
    with open(metrics_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "RMSE", "MAE", "R2"])
        if not file_exists:
            writer.writeheader()
        writer.writerow({"model": "BiLSTM", "RMSE": rmse, "MAE": mae, "R2": r2})
    print(f"Metrics appended to {metrics_path}")

    # Save per-event predictions for dashboard and evaluation framework
    pred_df = pd.DataFrame({
        'event_id': test_ids,
        'actual_risk': y_test,
        'predicted_risk': preds
    })
    pred_df.to_csv("results/predictions/lstm_predictions.csv", index=False)
    print("Predictions saved to results/predictions/lstm_predictions.csv")