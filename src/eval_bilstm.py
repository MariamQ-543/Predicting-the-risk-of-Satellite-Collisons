import os
import csv
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Load saved predictions from the last training run
pred_path = "results/predictions/lstm_predictions.csv"

if not os.path.exists(pred_path):
    print("ERROR: lstm_predictions.csv not found.")
    print("You need to retrain the model first by running train_lstm.py")
else:
    preds_df = pd.read_csv(pred_path)

    y_test = preds_df['actual_risk'].values
    preds = preds_df['predicted_risk'].values

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"BiLSTM Results -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

    # Append to shared metrics file
    os.makedirs("results", exist_ok=True)
    metrics_path = "results/model_metrics.csv"
    file_exists = os.path.isfile(metrics_path)

    with open(metrics_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "RMSE", "MAE", "R2"])
        if not file_exists:
            writer.writeheader()
        writer.writerow({"model": "BiLSTM", "RMSE": rmse, "MAE": mae, "R2": r2})

    print(f"Metrics appended to {metrics_path}")
    print("\nCurrent model_metrics.csv:")
    print(pd.read_csv(metrics_path))