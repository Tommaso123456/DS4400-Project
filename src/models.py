"""
Model definitions and training for credit card fraud detection.
Models: Logistic Regression, Random Forest, Gradient Boosting (XGBoost), Neural Network.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        tree_method="hist",   # memory-efficient histogram method
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return model



# Neural Network
class FraudNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


class NeuralNetWrapper:
    """Sklearn-style wrapper so the NN fits the same evaluation interface."""

    def __init__(self, input_dim: int, epochs: int = 20, batch_size: int = 512, lr: float = 1e-3):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X_train, y_train):
        X_t = torch.tensor(np.array(X_train), dtype=torch.float32).to(self.device)
        y_t = torch.tensor(np.array(y_train), dtype=torch.float32).to(self.device)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = FraudNet(self.input_dim).to(self.device)

        # weight positive class to handle imbalance
        pos_weight = torch.tensor([(y_t == 0).sum() / (y_t == 1).sum()]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(self.epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()

        return self

    def predict_proba(self, X_test):
        self.model.eval()
        X_t = torch.tensor(np.array(X_test), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
            probs = torch.sigmoid(logits).cpu().numpy()
        return np.column_stack([1 - probs, probs])

    def predict(self, X_test, threshold: float = 0.5):
        probs = self.predict_proba(X_test)[:, 1]
        return (probs >= threshold).astype(int)


def train_neural_network(X_train, y_train, epochs: int = 20):
    input_dim = X_train.shape[1]
    wrapper = NeuralNetWrapper(input_dim=input_dim, epochs=epochs)
    wrapper.fit(X_train, y_train)
    return wrapper
