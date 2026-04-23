# Graph Neural Network for satellite collision risk prediction
# each conjunction event is a node in the graph
# instead of connecting every event to every other event,
# each event only connects to a few nearby events from the same mission
# if events from the same ESA mission happen close together,
# they might have similar conjunction patterns / orbital conditions
#
# https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html
# https://www.geeksforgeeks.org/deep-learning/graph-neural-networks-with-pytorch/
# https://www.geeksforgeeks.org/deep-learning/graph-convolutional-networks-gcns-architectural-insights-and-applications/

import os
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from load_data import load_csv

TARGET = "risk"
NEIGHBOURS = 5  # how many nearby events to connect to on each side

# remove columns that would leak the answer
# max_risk_estimate and max_risk_scaling gave fake super high results before
LEAKAGE_COLS = [
    "max_risk_estimate",
    "max_risk_scaling",
    TARGET,
    "event_id",
    "mission_id"
]


def prepare_data():
    # load same ESA train/test data as the other models
    train = load_csv("esa_train.csv")
    test = load_csv("esa_test.csv")

    # encode the text column as numbers
    # fit categories on train first, then use the same categories on test
    if "c_object_type" in train.columns:
        categories = train["c_object_type"].astype("category").cat.categories
        train["c_object_type"] = pd.Categorical(train["c_object_type"], categories=categories).codes
        test["c_object_type"] = pd.Categorical(test["c_object_type"], categories=categories).codes

    # get one row per event
    # use the final CDM closest to TCA
    # that keeps this model fair with the other event-level models
    train_final = (
        train.sort_values("time_to_tca")
        .groupby("event_id")
        .first()
        .reset_index()
    )

    test_final = (
        test.sort_values("time_to_tca")
        .groupby("event_id")
        .first()
        .reset_index()
    )

    # choose feature columns
    # drop leakage columns and keep only numeric ones
    drop_cols = [c for c in LEAKAGE_COLS if c in train_final.columns]
    feature_cols = [c for c in train_final.columns if c not in drop_cols]

    feature_cols = [
        c for c in feature_cols
        if pd.api.types.is_numeric_dtype(train_final[c])
    ]

    # fill missing values using train medians only
    # don't use test info here
    train_medians = train_final[feature_cols].median()
    train_final[feature_cols] = train_final[feature_cols].fillna(train_medians)
    test_final[feature_cols] = test_final[feature_cols].fillna(train_medians)

    # scale features because this is still a neural network model
    scaler = StandardScaler()
    train_final[feature_cols] = scaler.fit_transform(train_final[feature_cols])
    test_final[feature_cols] = scaler.transform(test_final[feature_cols])

    return train_final, test_final, feature_cols


def build_graph(df, feature_cols, k=5):
    # build graph where:
    # - each node = one conjunction event
    # - node features = the same numeric features used by other models
    # - target = final risk for that event
    #
    # edges:
    # only connect each event to a few nearby events from the same mission
    # nearby here means nearby in time order using time_to_tca
    #
    # this is better than connecting every event to every other event
    # because that was too slow and too dense

    df = df.reset_index(drop=True)

    # node features
    x = torch.tensor(df[feature_cols].values, dtype=torch.float)

    # target values
    y = torch.tensor(df[TARGET].values, dtype=torch.float)

    edge_list = []

    # build edges mission by mission
    # so events only connect to other events from the same ESA satellite / mission
    for mission_id, group in df.groupby("mission_id"):
        # sort by time_to_tca so each mission has a simple time order
        # this is not the same as the BiLSTM sequence inside one event
        # here we are ordering whole events inside the same mission
        group = group.sort_values("time_to_tca").reset_index()

        # these are the real row indices from the full dataframe
        indices = group["index"].tolist()

        # for each event, connect to a few neighbours around it
        for pos, node_idx in enumerate(indices):
            start = max(0, pos - k)
            end = min(len(indices), pos + k + 1)

            for neighbour_pos in range(start, end):
                neighbour_idx = indices[neighbour_pos]

                if node_idx != neighbour_idx:
                    edge_list.append([node_idx, neighbour_idx])

    # fallback just in case graph ends up empty
    if len(edge_list) == 0:
        edge_list = [[i, i] for i in range(len(df))]

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index, y=y)


class GCNModel(nn.Module):
    # simple GCN model
    # graph layers do message passing between connected events
    # then a small MLP predicts the final risk

    def __init__(self, input_dim, hidden_dim=64):
        super(GCNModel, self).__init__()

        # first graph convolution layer
        self.gcn1 = GCNConv(input_dim, hidden_dim)

        # second graph convolution layer
        self.gcn2 = GCNConv(hidden_dim, hidden_dim // 2)

        # prediction head
        self.fc1 = nn.Linear(hidden_dim // 2, 32)
        self.fc2 = nn.Linear(32, 1)

        self.dropout = nn.Dropout(0.3)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # first round of message passing
        x = F.relu(self.gcn1(x, edge_index))
        x = self.dropout(x)

        # second round of message passing
        x = F.relu(self.gcn2(x, edge_index))

        # final prediction layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x.squeeze()


if __name__ == "__main__":
    print("Loading and preprocessing data...")
    train_final, test_final, feature_cols = prepare_data()

    print(f"Train events: {len(train_final)}")
    print(f"Test events:  {len(test_final)}")
    print(f"Features:     {len(feature_cols)}")

    # split train into train/val by event
    # do this before graph building so splits stay separate
    unique_events = train_final["event_id"].unique()
    np.random.seed(42)

    val_size = int(len(unique_events) * 0.1)
    val_events = np.random.choice(unique_events, val_size, replace=False)
    train_events = np.setdiff1d(unique_events, val_events)

    train_df = train_final[train_final["event_id"].isin(train_events)].reset_index(drop=True)
    val_df = train_final[train_final["event_id"].isin(val_events)].reset_index(drop=True)

    print("\nBuilding graphs...")
    train_graph = build_graph(train_df, feature_cols, k=NEIGHBOURS)
    val_graph = build_graph(val_df, feature_cols, k=NEIGHBOURS)
    test_graph = build_graph(test_final, feature_cols, k=NEIGHBOURS)

    print(f"Train graph: {train_graph.num_nodes} nodes, {train_graph.num_edges} edges")
    print(f"Val graph:   {val_graph.num_nodes} nodes, {val_graph.num_edges} edges")
    print(f"Test graph:  {test_graph.num_nodes} nodes, {test_graph.num_edges} edges")

    # build model
    model = GCNModel(input_dim=len(feature_cols), hidden_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0
    best_state = None

    print("\nTraining GNN...")
    for epoch in range(80):
        model.train()
        optimizer.zero_grad()

        preds = model(train_graph)
        loss = criterion(preds, train_graph.y)

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(val_graph)
            val_loss = criterion(val_preds, val_graph.y).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # evaluate best saved model on test graph
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        test_preds = model(test_graph).numpy()

    y_test = test_graph.y.numpy()

    rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    mae = mean_absolute_error(y_test, test_preds)
    r2 = r2_score(y_test, test_preds)

    print(f"\nGNN Test Results:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R2:   {r2:.4f}")

    # save model
    os.makedirs("results/models", exist_ok=True)
    torch.save(best_state, "results/models/best_gnn.pt")
    print("Model saved to results/models/best_gnn.pt")

    # save metrics
    os.makedirs("results", exist_ok=True)
    metrics_path = "results/model_metrics.csv"
    file_exists = os.path.isfile(metrics_path)

    with open(metrics_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "RMSE", "MAE", "R2"])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "model": "GNN",
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        })

    print(f"Metrics saved to {metrics_path}")

    # save predictions
    os.makedirs("results/predictions", exist_ok=True)
    pred_df = pd.DataFrame({
        "event_id": test_final["event_id"].values,
        "actual_risk": y_test,
        "predicted_risk": test_preds
    })
    pred_df.to_csv("results/predictions/gnn_predictions.csv", index=False)
    print("Predictions saved to results/predictions/gnn_predictions.csv")