# Transformer Encoder for satellite collision risk prediction
# uses the same CDM sequences as BiLSTM but processes all timesteps
# simultaneously using self-attention instead of sequentially
# this lets it directly learn which CDMs are most relevant to final risk
# regardless of their position in the sequence
#
# tutorials used:
# https://www.geeksforgeeks.org/transformer-neural-network/
# https://www.tensorflow.org/text/tutorials/transformer
# https://machinelearningmastery.com/the-transformer-model/
# https://www.datacamp.com/tutorial/how-transformers-work
# https://www.geeksforgeeks.org/ml-neural-network-implementation-in-c-from-scratch/

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
MAX_LEN = 23

# same features as BiLSTM so comparison between the two is fair
# selected based on LightGBM feature importance
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
    # same preprocessing as BiLSTM - sort ascending so sequence goes
    # from earliest CDM to closest to TCA
    train = load_csv("esa_train.csv")
    test = load_csv("esa_test.csv")

    train = train.sort_values(['event_id', 'time_to_tca'], ascending=[True, True])
    test = test.sort_values(['event_id', 'time_to_tca'], ascending=[True, True])

    train['time_delta'] = train.groupby('event_id')['time_to_tca'].diff().fillna(0).abs()
    test['time_delta'] = test.groupby('event_id')['time_to_tca'].diff().fillna(0).abs()

    current_features = FEATURE_COLS + ['time_delta']

    # fill missing with train medians only to prevent leakage
    # https://www.geeksforgeeks.org/machine-learning/what-is-data-leakage/
    train_medians = train.median(numeric_only=True)
    train = train.fillna(train_medians)
    test = test.fillna(train_medians)

    scaler = StandardScaler()
    train[current_features] = scaler.fit_transform(train[current_features])
    test[current_features] = scaler.transform(test[current_features])

    return train, test, current_features

def create_sequences(df, feature_cols, target_col):
    # same sequence building as BiLSTM for fair comparison
    # target = risk at final CDM (closest to TCA)
    # pad at start so recent CDMs are always at the end
    x_list, y_list, id_list = [], [], []

    for event_id, group in df.groupby('event_id'):
        group = group.sort_values('time_to_tca', ascending=True)
        features = group[feature_cols].values
        target_val = group.iloc[-1][target_col]

        if len(features) < MAX_LEN:
            pad = np.zeros((MAX_LEN - len(features), len(feature_cols)))
            features = np.vstack([pad, features])
        else:
            features = features[:MAX_LEN]

        x_list.append(features)
        y_list.append(target_val)
        id_list.append(event_id)

    return np.array(x_list), np.array(y_list), np.array(id_list)

def positional_encoding(max_len, d_model):
    # positional encoding tells the transformer the order of the CDMs
    # without this the model wouldnt know which CDM came first
    # uses sine and cosine functions at different frequencies
    # https://www.geeksforgeeks.org/transformer-neural-network/
    positions = np.arange(max_len)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]

    angles = positions / np.power(10000, (2 * (dims // 2)) / np.float32(d_model))

    # apply sin to even indices, cos to odd indices
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])

    return tf.cast(angles[np.newaxis, :, :], dtype=tf.float32)

def build_model(input_shape, num_heads=4, ff_dim=128, num_layers=2, dropout=0.1):
    # Transformer encoder architecture
    # num_heads: number of attention heads - each learns different relationships
    # ff_dim: size of feed-forward layer inside each transformer block
    # num_layers: number of stacked transformer encoder layers
    # https://machinelearningmastery.com/the-transformer-model/

    seq_len, feat_dim = input_shape

    inputs = layers.Input(shape=input_shape)

    # project input features up to d_model dimension
    # d_model must be divisible by num_heads for multi-head attention
    d_model = 64
    x = layers.Dense(d_model)(inputs)

    # add positional encoding so model knows the order of CDMs
    pos_enc = positional_encoding(seq_len, d_model)
    x = x + pos_enc

    # stack transformer encoder layers
    # each layer has:
    # 1. multi-head self-attention (all CDMs attend to each other)
    # 2. feed-forward network
    # 3. layer normalisation after each
    # https://www.datacamp.com/tutorial/how-transformers-work
    for _ in range(num_layers):
        # multi-head self-attention
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout
        )(x, x)
        attn_output = layers.Dropout(dropout)(attn_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

        # feed-forward network
        ff_output = layers.Dense(ff_dim, activation='relu')(x)
        ff_output = layers.Dense(d_model)(ff_output)
        ff_output = layers.Dropout(dropout)(ff_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ff_output)

    # global average pooling collapses the 23 timesteps into one vector
    # by averaging across the sequence dimension
    x = layers.GlobalAveragePooling1D()(x)

    # final prediction head
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1, activation='linear')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

if __name__ == "__main__":
    # 1. prepare data - identical pipeline to BiLSTM
    train_df, test_df, features = prepare_data()

    # 2. split by event_id to prevent leakage
    unique_events = train_df['event_id'].unique()
    train_ids, val_ids = train_test_split(unique_events, test_size=0.1, random_state=42)

    val_data = train_df[train_df['event_id'].isin(val_ids)]
    train_data = train_df[train_df['event_id'].isin(train_ids)]

    X_train, y_train, _ = create_sequences(train_data, features, TARGET)
    X_val, y_val, _ = create_sequences(val_data, features, TARGET)
    X_test, y_test, test_ids = create_sequences(test_df, features, TARGET)

    print(f"Shapes (Train/Val/Test): {X_train.shape}, {X_val.shape}, {X_test.shape}")

    # 3. build and train model
    model = build_model((MAX_LEN, len(features)))
    model.summary()

    os.makedirs("results/models", exist_ok=True)
    my_callbacks = [
        callbacks.EarlyStopping(patience=8, restore_best_weights=True),
        callbacks.ModelCheckpoint(
            "results/models/best_transformer.keras", save_best_only=True)
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=my_callbacks,
        verbose=1
    )

    # 4. evaluate - same unit as all other models (one prediction per event)
    preds = model.predict(X_test).ravel()
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"\n=== Transformer Results ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R2:   {r2:.4f}")

    # 5. save results
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/predictions", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    metrics_path = "results/model_metrics.csv"
    file_exists = os.path.isfile(metrics_path)
    with open(metrics_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "RMSE", "MAE", "R2"])
        if not file_exists:
            writer.writeheader()
        writer.writerow({"model": "Transformer", "RMSE": rmse, "MAE": mae, "R2": r2})
    print(f"Metrics appended to {metrics_path}")

    pred_df = pd.DataFrame({
        'event_id': test_ids,
        'actual_risk': y_test,
        'predicted_risk': preds
    })
    pred_df.to_csv("results/predictions/transformer_predictions.csv", index=False)
    print("Predictions saved to results/predictions/transformer_predictions.csv")

    print("\nAll models so far:")
    print(pd.read_csv(metrics_path))