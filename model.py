"""
PyTorch model definition for Customer Churn Prediction.

Mirrors the original Keras architecture:
    Dense(64, relu)  -> Dense(32, relu) -> Dense(1, sigmoid)

Kept in its own module so both train.py and app.py (Streamlit) can import
the exact same class when saving/loading weights.
"""

import torch
import torch.nn as nn


class ChurnANN(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            # No sigmoid here on purpose -> we train with BCEWithLogitsLoss
            # (numerically more stable). Sigmoid is applied at inference time
            # via `predict_proba`.
        )

    def forward(self, x):
        return self.net(x)  # returns raw logits

    @torch.no_grad()
    def predict_proba(self, x):
        self.eval()
        logits = self.forward(x)
        return torch.sigmoid(logits)
