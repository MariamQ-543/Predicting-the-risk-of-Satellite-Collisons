# Graph Attention Network (GAT) for satellite collision risk prediction
# models the 19 ESA missions as a graph - each mission is a node
# fully connected graph - all missions connected to each other
# GAT attention weights learn which connections actually matter
# each node learns from its own features AND neighbouring missions
# via message passing
#
# tutorials used:
# https://www.geeksforgeeks.org/deep-learning/graph-neural-networks-with-pytorch/
# https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html
# https://www.datacamp.com/tutorial/comprehensive-introduction-graph-neural-networks-gnns-tutorial
# https://medium.com/towards-data-science/hands-on-graph-neural-networks-with-pytorch-pytorch-geometric-359487e221a8
# https://www.baeldung.com/cs/graph-attention-networks
# https://medium.com/@farzad.karami/understanding-graph-attention-networks-a-practical-exploration-cf033a8f3d9d
# https://www.dgl.ai/dgl_docs/en/2.0.x/tutorials/models/1_gnn/9_gat.html

import os
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from load_data import load_csv

TARGET = "risk"
ID_COLS = ["event_id", "mission_id"]
LEAKY_COLS = ["max_risk_estimate", "max_risk_scaling"]

# features aggregated per mission to describe each graph node
# using mean across all CDMs for that mission
NODE_FEATURE_COLS = [
    'miss_distance', 'relative_speed', 'relative_position_r',
    'mahalanobis_distance', 'c_sigma_t', 'c_sigma_r',
    'time_to_tca', 'F10', 'AP'
]

# event-level features used for the final prediction
# same top features confirmed by LightGBM importance
EVENT_FEATURE_COLS = [
    'miss_distance', 'relative_speed', 'relative_position_r',
    'relative_position_t', 'relative_position_n',
    'mahalanobis_distance', 'c_sigma_t', 'c_sigma_r',
    'c_sigma_tdot', 'c_sigma_n', 'c_position_covariance_det',
    'c_ct_r', 't_sigma_r', 't_sigma_t',
    'time_to_tca', 'geocentric_latitude', 'azimuth', 'elevation',
    'F10', 'F3M', 'AP'
]

def build_graph(df, node_feature_cols):
    # builds a graph where each of the 19 ESA missions is a node
    # fully connected - all missions connected to each other
    # with only 19 nodes this is fine (19x18 = 342 edges)
    # GAT attention weights learn which connections actually matter
    # node features are the mean CDM values for each mission

    missions = sorted(df['mission_id'].unique())
    mission_to_idx = {m: i for i, m in enumerate(missions)}
    num_nodes = len(missions)

    # build node features by averaging CDM values per mission
    node_features = []
    for mission in missions:
        mission_data = df[df['mission_id'] == mission]
        features = mission_data[node_feature_cols].mean().values
        node_features.append(features)

    node_features = np.array(node_features, dtype=np.float32)

    # fill any nans from missions missing a feature
    col_means = np.nanmean(node_features, axis=0)
    for i in range(node_features.shape[1]):
        mask = np.isnan(node_features[:, i])
        node_features[mask, i] = col_means[i]

    # fully connected graph - every mission connected to every other mission
    # the attention mechanism learns which connections are important
    # so we dont need to manually decide which missions are related
    edge_set = set()
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edge_set.add((i, j))

    edges = list(edge_set)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.tensor(node_features, dtype=torch.float)

    return Data(x=x, edge_index=edge_index), mission_to_idx, missions


class GATModel(nn.Module):
    # GAT with 2 message passing layers
    # each mission node looks at its neighbours and learns which ones
    # matter most using attention weights - similar to the attention in BiLSTM
    # after message passing, combines node embedding with event features
    # and predicts risk through a small MLP

    def __init__(self, node_feat_dim, event_feat_dim, hidden_dim=32):
        super(GATModel, self).__init__()

        # GAT layer 1 - heads=4 means 4 separate attention mechanisms
        # each head learns different aspects of the neighbourhood
        self.gat1 = GATConv(node_feat_dim, hidden_dim, heads=4, dropout=0.3)

        # GAT layer 2 - concat=False averages the 4 heads into one
        self.gat2 = GATConv(hidden_dim * 4, hidden_dim, heads=1,
                            concat=False, dropout=0.3)

        # MLP head combines the node embedding with event-specific features
        # event features give info about this specific conjunction
        # not just the mission's general profile
        combined_dim = hidden_dim + event_feat_dim
        self.fc1 = nn.Linear(combined_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, data, event_features, mission_indices):
        x, edge_index = data.x, data.edge_index

        # message passing - nodes learn from their neighbours
        x = F.elu(self.gat1(x, edge_index))
        x = self.dropout(x)
        x = F.elu(self.gat2(x, edge_index))

        # get the node embedding for the mission in each event
        node_embeddings = x[mission_indices]

        # combine node embedding with event-specific features
        combined = torch.cat([node_embeddings, event_features], dim=1)

        # MLP prediction head
        out = F.relu(self.fc1(combined))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        return out.squeeze()


def prepare_data():
    # same preprocessing as other models - drop high missing cols,
    # encode object type, fill with train medians, scale features
    # filter to final CDM per event for fair comparison

    train = load_csv("esa_train.csv")
    test = load_csv("esa_test.csv")

    # drop columns with more than 50% missing values
    threshold = 0.5
    missing_frac = train.isnull().mean()
    cols_to_drop = missing_frac[missing_frac > threshold].index.tolist()
    train = train.drop(columns=cols_to_drop)
    test = test.drop(columns=cols_to_drop)

    # encode c_object_type - fit on train only to prevent leakage
    if "c_object_type" in train.columns:
        categories = train["c_object_type"].astype("category").cat.categories
        train["c_object_type"] = pd.Categorical(
            train["c_object_type"], categories=categories).codes
        test["c_object_type"] = pd.Categorical(
            test["c_object_type"], categories=categories).codes

    # fill missing with train medians only
    train_medians = train.median(numeric_only=True)
    train = train.fillna(train_medians)
    test = test.fillna(train_medians)

    # filter to final CDM per event (smallest time_to_tca)
    # same evaluation unit as LightGBM, MLP, BiLSTM and physics baseline
    train_final = (train.sort_values('time_to_tca')
                   .groupby('event_id').first().reset_index())
    test_final = (test.sort_values('time_to_tca')
                  .groupby('event_id').first().reset_index())

    # only keep cols that exist after the missing value drop
    event_cols = [c for c in EVENT_FEATURE_COLS if c in train_final.columns]
    node_cols = [c for c in NODE_FEATURE_COLS if c in train.columns]

    # scale event features
    scaler = StandardScaler()
    train_final[event_cols] = scaler.fit_transform(train_final[event_cols])
    test_final[event_cols] = scaler.transform(test_final[event_cols])

    # scale node features using full train data
    node_scaler = StandardScaler()
    train[node_cols] = node_scaler.fit_transform(train[node_cols])
    test[node_cols] = node_scaler.transform(test[node_cols])

    return train, test, train_final, test_final, event_cols, node_cols


if __name__ == "__main__":
    # 1. prepare data
    train, test, train_final, test_final, event_cols, node_cols = prepare_data()

    # 2. build graph from training data
    graph_data, mission_to_idx, missions = build_graph(train, node_cols)

    print(f"\nGraph built:")
    print(f"  Nodes (missions): {graph_data.num_nodes}")
    print(f"  Edges: {graph_data.num_edges}")
    print(f"  Node feature dim: {graph_data.num_node_features}")

    # 3. prepare tensors for training
    def get_tensors(df_final):
        mission_indices = df_final['mission_id'].map(
            lambda m: mission_to_idx.get(m, 0)).values

        X_events = torch.tensor(df_final[event_cols].values, dtype=torch.float)
        y = torch.tensor(df_final[TARGET].values, dtype=torch.float)
        mission_idx_tensor = torch.tensor(mission_indices, dtype=torch.long)

        return X_events, y, mission_idx_tensor

    # 90/10 train/val split by event
    unique_events = train_final['event_id'].unique()
    np.random.seed(42)
    val_size = int(len(unique_events) * 0.1)
    val_events = np.random.choice(unique_events, val_size, replace=False)
    train_events = np.setdiff1d(unique_events, val_events)

    train_df = train_final[train_final['event_id'].isin(train_events)]
    val_df = train_final[train_final['event_id'].isin(val_events)]

    X_train, y_train, idx_train = get_tensors(train_df)
    X_val, y_val, idx_val = get_tensors(val_df)
    X_test, y_test, idx_test = get_tensors(test_final)

    print(f"\nTrain/Val/Test events: {len(X_train)}/{len(X_val)}/{len(X_test)}")

    # 4. build and train model
    model = GATModel(
        node_feat_dim=graph_data.num_node_features,
        event_feat_dim=len(event_cols)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    best_model_state = None

    print("\nTraining GNN...")
    for epoch in range(150):
        model.train()
        optimizer.zero_grad()
        preds = model(graph_data, X_train, idx_train)
        loss = criterion(preds, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(graph_data, X_val, idx_val)
            val_loss = criterion(val_preds, y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {loss.item():.4f} "
                  f"| Val Loss: {val_loss:.4f}")

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # 5. evaluate on test set
    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        test_preds = model(graph_data, X_test, idx_test).numpy()

    y_test_np = y_test.numpy()
    rmse = np.sqrt(mean_squared_error(y_test_np, test_preds))
    mae = mean_absolute_error(y_test_np, test_preds)
    r2 = r2_score(y_test_np, test_preds)

    print(f"\n=== GNN Results ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R2:   {r2:.4f}")

    # 6. save results
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/predictions", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    torch.save(best_model_state, "results/models/best_gnn.pt")

    metrics_path = "results/model_metrics.csv"
    file_exists = os.path.isfile(metrics_path)
    with open(metrics_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "RMSE", "MAE", "R2"])
        if not file_exists:
            writer.writeheader()
        writer.writerow({"model": "GNN", "RMSE": rmse, "MAE": mae, "R2": r2})
    print(f"Metrics appended to {metrics_path}")

    pred_df = pd.DataFrame({
        'event_id': test_final['event_id'].values,
        'actual_risk': y_test_np,
        'predicted_risk': test_preds
    })
    pred_df.to_csv("results/predictions/gnn_predictions.csv", index=False)
    print("Predictions saved to results/predictions/gnn_predictions.csv")

    print("\nAll models:")
    print(pd.read_csv(metrics_path))