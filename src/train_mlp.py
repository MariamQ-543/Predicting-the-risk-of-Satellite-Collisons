# MLP (Multi-Layer Perceptron) for satellite collision risk prediction
# Uses the same flat feature input as LightGBM but learns through neural layers
# instead of decision trees - allows direct comparison between model types
#
# Tutorials used:
# https://www.geeksforgeeks.org/multi-layer-perceptron-learning-in-tensorflow/
# https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/
# https://www.geeksforgeeks.org/ml-neural-network-implementation-in-c-from-scratch/
# https://medium.com/@brijesh_soni/understanding-dropout-in-neural-networks-a-comprehensive-guide-pretty-good-explanation-dae3966fcfdb
# https://www.geeksforgeeks.org/batch-normalization-in-deep-learning/
# https://www.geeksforgeeks.org/machine-learning/what-is-data-leakage/

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

TARGET = "risk"
ID_COLS = ["event_id", "mission_id"]

# removing these because they are derived from the target variable
# including them caused R2 = 0.9998 which was data leakage not real performance
# https://www.geeksforgeeks.org/machine-learning/what-is-data-leakage/
LEAKY_COLS = ["max_risk_estimate", "max_risk_scaling"]

def prepare_data():
    train = load_csv("esa_train.csv")
    test = load_csv("esa_test.csv")

    # drop columns where more than 50% of values are missing
    # cant reliably fill that much data without introducing bias
    threshold = 0.5
    missing_frac = train.isnull().mean()
    cols_to_drop = missing_frac[missing_frac > threshold].index.tolist()
    train = train.drop(columns=cols_to_drop)
    test = test.drop(columns=cols_to_drop)

    # encode c_object_type (text like PAYLOAD, DEBRIS) into numbers
    # fit on train only so test categories dont leak into training
    if "c_object_type" in train.columns:
        categories = train["c_object_type"].astype("category").cat.categories
        train["c_object_type"] = pd.Categorical(
            train["c_object_type"], categories=categories).codes
        test["c_object_type"] = pd.Categorical(
            test["c_object_type"], categories=categories).codes

    # fill remaining missing values with medians from train set only
    train_medians = train.median(numeric_only=True)
    train = train.fillna(train_medians)
    test = test.fillna(train_medians)

    feature_cols = [c for c in train.columns
                    if c not in [TARGET] + ID_COLS + LEAKY_COLS]

    # filter to final CDM per event (smallest time_to_tca = closest to TCA)
    # same evaluation unit as LightGBM, BiLSTM and physics baseline
    # so the comparison between models is fair
    train_final = (train.sort_values('time_to_tca')
                   .groupby('event_id').first().reset_index())
    test_final = (test.sort_values('time_to_tca')
                  .groupby('event_id').first().reset_index())

    X_train = train_final[feature_cols].values
    y_train = train_final[TARGET].values
    X_test = test_final[feature_cols].values
    y_test = test_final[TARGET].values
    test_event_ids = test_final['event_id'].values

    # scale features so all inputs are on similar scale
    # neural networks are sensitive to feature magnitude unlike tree-based models
    # fit on train only, then apply same scaling to test
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test, test_event_ids, feature_cols

def build_model(input_dim):
    # MLP with 4 hidden layers, getting smaller towards the output
    # BatchNorm stabilises training between layers
    # Dropout randomly switches off neurons during training to prevent overfitting
    # https://www.geeksforgeeks.org/batch-normalization-in-deep-learning/
    # https://medium.com/@brijesh_soni/understanding-dropout-in-neural-networks-a-comprehensive-guide-pretty-good-explanation-dae3966fcfdb

    inputs = layers.Input(shape=(input_dim,))

    # layer 1 - 256 neurons, learns broad feature combinations
    x = layers.Dense(256)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)

    # layer 2 - 128 neurons
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)

    # layer 3 - 64 neurons
    x = layers.Dense(64)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)

    # layer 4 - 32 neurons, final compression before output
    x = layers.Dense(32, activation='relu')(x)

    # output layer - linear activation because this is regression not classification
    outputs = layers.Dense(1, activation='linear')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

if __name__ == "__main__":
    X_train, y_train, X_test, y_test, test_ids, feature_cols = prepare_data()

    print(f"\nTraining MLP on {X_train.shape[1]} features")
    print(f"Train events: {X_train.shape[0]}, Test events: {X_test.shape[0]}")

    # 10% validation split, same ratio as BiLSTM
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42)

    model = build_model(X_train.shape[1])

    os.makedirs("results/models", exist_ok=True)
    my_callbacks = [
        # stop training early if validation loss doesnt improve for 10 epochs
        callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        callbacks.ModelCheckpoint(
            "results/models/best_mlp.keras", save_best_only=True)
    ]

    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=64,
        callbacks=my_callbacks,
        verbose=1
    )

    # evaluate on test set - one prediction per event, same as all other models
    preds = model.predict(X_test).ravel()
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"\n=== MLP Results ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R2:   {r2:.4f}")

    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/predictions", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # append to shared metrics file
    metrics_path = "results/model_metrics.csv"
    file_exists = os.path.isfile(metrics_path)
    with open(metrics_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "RMSE", "MAE", "R2"])
        if not file_exists:
            writer.writeheader()
        writer.writerow({"model": "MLP", "RMSE": rmse, "MAE": mae, "R2": r2})
    print(f"Metrics appended to {metrics_path}")

    pred_df = pd.DataFrame({
        'event_id': test_ids,
        'actual_risk': y_test,
        'predicted_risk': preds
    })
    pred_df.to_csv("results/predictions/mlp_predictions.csv", index=False)
    print("Predictions saved to results/predictions/mlp_predictions.csv")

    print("\nAll models so far:")
    print(pd.read_csv(metrics_path))