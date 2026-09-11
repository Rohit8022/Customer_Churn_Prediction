"""
Train the Customer Churn ANN using PyTorch.

This replaces the Keras/TensorFlow training code from experiments.ipynb.
Preprocessing (drop columns, label encode Gender, one-hot encode Geography,
scale features) is kept identical to the original so the saved
scaler.pkl / label_encoder_gender.pkl / onehot_encoder_geo.pkl are still
compatible with app.py.

Usage:
    python train.py --data Churn_Modelling.csv --epochs 100
"""

import argparse
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from model import ChurnANN


def load_and_preprocess(csv_path: str):
    data = pd.read_csv(csv_path)

    # Drop irrelevant columns
    data = data.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

    # Label encode Gender
    label_encoder_gender = LabelEncoder()
    data["Gender"] = label_encoder_gender.fit_transform(data["Gender"])

    # One-hot encode Geography
    onehot_encoder_geo = OneHotEncoder()
    geo_encoded = onehot_encoder_geo.fit_transform(data[["Geography"]]).toarray()
    geo_encoded_df = pd.DataFrame(
        geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
    )
    data = pd.concat(
        [data.drop("Geography", axis=1).reset_index(drop=True), geo_encoded_df],
        axis=1,
    )

    # Split features / target
    X = data.drop("Exited", axis=1)
    y = data["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return (
        X_train,
        X_test,
        y_train.values,
        y_test.values,
        label_encoder_gender,
        onehot_encoder_geo,
        scaler,
    )


def train(args):
    # Fix random seeds for reproducibility (weight init, DataLoader shuffling, etc.)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    (
        X_train,
        X_test,
        y_train,
        y_test,
        label_encoder_gender,
        onehot_encoder_geo,
        scaler,
    ) = load_and_preprocess(args.data)

    # Save encoders/scaler (same filenames as the original project so
    # app.py can load them unchanged)
    with open("label_encoder_gender.pkl", "wb") as f:
        pickle.dump(label_encoder_gender, f)
    with open("onehot_encoder_geo.pkl", "wb") as f:
        pickle.dump(onehot_encoder_geo, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    train_ds = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True
    )

    model = ChurnANN(input_dim=X_train.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        train_loss = running_loss / len(train_ds)

        # Validation
        model.eval()
        with torch.no_grad():
            X_test_dev = X_test_t.to(device)
            y_test_dev = y_test_t.to(device)
            val_logits = model(X_test_dev)
            val_loss = criterion(val_logits, y_test_dev).item()
            val_preds = (torch.sigmoid(val_logits) > 0.5).float()
            val_acc = (val_preds == y_test_dev).float().mean().item()

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )

        # Early stopping (mirrors Keras EarlyStopping(patience=10, restore_best_weights=True))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (best val_loss={best_val_loss:.4f})")
                break

    # Restore best weights before saving
    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(
        {"state_dict": model.state_dict(), "input_dim": X_train.shape[1]},
        args.output,
    )
    print(f"Saved trained model to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train churn prediction ANN (PyTorch)")
    parser.add_argument("--data", type=str, default="Churn_Modelling.csv")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--output", type=str, default="model.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)
