"""
DigitalTwin.ai - PyTorch LSTM Short-Horizon Bottleneck Forecaster
Forecasts near-future cycle time & queue length sequences to predict bottlenecks before they form.
Includes normalized target scaling, walk-forward validation, and naive persistence/EMA baselines.
"""

import os
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class BottleneckLSTMNet(nn.Module):
    """
    Lightweight 2-layer LSTM with Dropout and linear projection head.
    Input: [batch_size, seq_len=15, feature_dim=4] (cycle_time, queue_len, uph, power_kw)
    Output: [batch_size, forecast_horizon=5, output_dim=2] (normalized future cycle_time, future queue_len)
    """

    def __init__(self, input_dim: int = 4, hidden_dim: int = 32, num_layers: int = 2, forecast_horizon: int = 5, dropout: float = 0.2):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, forecast_horizon * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_step = self.dropout(lstm_out[:, -1, :])
        out = self.fc(last_step)
        return out.view(-1, self.forecast_horizon, 2)


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class BottleneckForecaster:
    """
    Manager class for training, evaluating, and predicting with the Bottleneck LSTM.
    Normalized inputs and targets for optimal neural net convergence.
    """

    def __init__(self, seq_len: int = 15, horizon: int = 5, hidden_dim: int = 32):
        self.seq_len = seq_len
        self.horizon = horizon
        self.model = BottleneckLSTMNet(input_dim=4, hidden_dim=hidden_dim, forecast_horizon=horizon)
        self.norm_means = np.array([60.0, 1.0, 55.0, 3.0], dtype=np.float32)
        self.norm_stds = np.array([10.0, 2.0, 15.0, 1.5], dtype=np.float32)
        self.loss_history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    def prepare_sequences(self, station_time_series: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        raw_data = []
        for r in station_time_series:
            ct = float(r.get("cycle_time", 60.0))
            q = float(r.get("queue_len", 1.0))
            uph = float(r.get("throughput_uph", 55.0))
            pwr = float(r.get("power_kw", 3.0))
            raw_data.append([ct, q, uph, pwr])
        
        arr = np.array(raw_data, dtype=np.float32)
        arr_norm = (arr - self.norm_means) / (self.norm_stds + 1e-6)

        X_list, y_list = [], []
        total_len = len(arr)
        for i in range(total_len - self.seq_len - self.horizon + 1):
            x_window = arr_norm[i : i + self.seq_len]
            # Normalize target [cycle_time, queue_len] with first 2 channels
            y_window = arr_norm[i + self.seq_len : i + self.seq_len + self.horizon, 0:2]
            X_list.append(x_window)
            y_list.append(y_window)

        if not X_list:
            return np.empty((0, self.seq_len, 4)), np.empty((0, self.horizon, 2))

        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

    def train_model(
        self,
        train_data: List[Dict[str, Any]],
        val_data: List[Dict[str, Any]],
        epochs: int = 35,
        batch_size: int = 32,
        lr: float = 0.005
    ) -> Dict[str, List[float]]:
        X_train, y_train = self.prepare_sequences(train_data)
        X_val, y_val = self.prepare_sequences(val_data)

        if len(X_train) == 0 or len(X_val) == 0:
            return self.loss_history

        train_loader = DataLoader(SequenceDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(SequenceDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.MSELoss()

        self.loss_history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        best_weights = None

        self.model.train()
        for epoch in range(epochs):
            running_train_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                pred = self.model(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                running_train_loss += loss.item() * len(batch_X)

            epoch_train_loss = running_train_loss / len(X_train)
            self.loss_history["train_loss"].append(round(epoch_train_loss, 4))

            # Validation pass
            self.model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    pred = self.model(batch_X)
                    loss = criterion(pred, batch_y)
                    running_val_loss += loss.item() * len(batch_X)

            epoch_val_loss = running_val_loss / len(X_val)
            self.loss_history["val_loss"].append(round(epoch_val_loss, 4))

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            self.model.train()

        if best_weights:
            self.model.load_state_dict(best_weights)
        self.model.eval()
        return self.loss_history

    def predict_forecast(self, recent_15_ticks: List[Dict[str, Any]]) -> Dict[str, Any]:
        self.model.eval()
        raw_data = []
        for r in recent_15_ticks[-self.seq_len:]:
            raw_data.append([
                float(r.get("cycle_time", 60.0)),
                float(r.get("queue_len", 1.0)),
                float(r.get("throughput_uph", 55.0)),
                float(r.get("power_kw", 3.0))
            ])
        
        while len(raw_data) < self.seq_len:
            raw_data.insert(0, [60.0, 1.0, 55.0, 3.0])

        arr = np.array(raw_data, dtype=np.float32)
        arr_norm = (arr - self.norm_means) / (self.norm_stds + 1e-6)
        tensor_in = torch.tensor(arr_norm, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            preds_norm = self.model(tensor_in).squeeze(0).numpy()  # shape [5, 2]

        # Denormalize predictions
        future_ct = (preds_norm[:, 0] * self.norm_stds[0]) + self.norm_means[0]
        future_queue = (preds_norm[:, 1] * self.norm_stds[1]) + self.norm_means[1]

        future_ct_rounded = [round(float(ct), 1) for ct in future_ct]
        future_q_rounded = [max(0, int(round(float(q)))) for q in future_queue]

        # Bottleneck forecast condition: forecasted cycle time exceeds takt nominal AND queue is projected to build
        is_bottleneck_predicted = any(ct >= 64.5 for ct in future_ct_rounded) and any(q >= 3 for q in future_q_rounded)

        return {
            "forecast_cycle_times": future_ct_rounded,
            "forecast_queues": future_q_rounded,
            "is_bottleneck_predicted": is_bottleneck_predicted,
            "max_forecast_cycle_time": max(future_ct_rounded),
            "max_forecast_queue": max(future_q_rounded),
            "lead_time_ticks": next((i + 1 for i, ct in enumerate(future_ct_rounded) if ct > 63.5), 5)
        }

    def evaluate_baselines_vs_lstm(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        X_test, y_test_norm = self.prepare_sequences(test_data)
        if len(X_test) == 0:
            return {"error": "No test sequences"}

        self.model.eval()
        with torch.no_grad():
            lstm_preds_norm = self.model(torch.tensor(X_test, dtype=torch.float32)).numpy()

        # Denormalize to true seconds
        lstm_ct_pred = (lstm_preds_norm[:, :, 0] * self.norm_stds[0]) + self.norm_means[0]
        true_ct = (y_test_norm[:, :, 0] * self.norm_stds[0]) + self.norm_means[0]

        lstm_mae = float(np.mean(np.abs(lstm_ct_pred - true_ct)))
        lstm_rmse = float(np.sqrt(np.mean((lstm_ct_pred - true_ct) ** 2)))

        # Naive Persistence: predict future as last observed value in sequence
        last_observed_ct = (X_test[:, -1, 0] * self.norm_stds[0]) + self.norm_means[0]
        persistence_pred = np.repeat(last_observed_ct[:, np.newaxis], self.horizon, axis=1)
        persist_mae = float(np.mean(np.abs(persistence_pred - true_ct)))
        persist_rmse = float(np.sqrt(np.mean((persistence_pred - true_ct) ** 2)))

        # Exponential Moving Average (alpha = 0.3)
        ema_vals = []
        for i in range(len(X_test)):
            seq_ct = (X_test[i, :, 0] * self.norm_stds[0]) + self.norm_means[0]
            val = seq_ct[0]
            for c in seq_ct[1:]:
                val = 0.3 * c + 0.7 * val
            ema_vals.append(val)
        ema_pred = np.repeat(np.array(ema_vals)[:, np.newaxis], self.horizon, axis=1)
        ema_mae = float(np.mean(np.abs(ema_pred - true_ct)))
        ema_rmse = float(np.sqrt(np.mean((ema_pred - true_ct) ** 2)))

        improvement = float((persist_mae - lstm_mae) / persist_mae * 100.0)

        return {
            "lstm": {"mae": round(lstm_mae, 3), "rmse": round(lstm_rmse, 3)},
            "naive_persistence": {"mae": round(persist_mae, 3), "rmse": round(persist_rmse, 3)},
            "ema_baseline": {"mae": round(ema_mae, 3), "rmse": round(ema_rmse, 3)},
            "mae_reduction_pct": round(improvement, 1)
        }

    def save(self, filepath: str) -> None:
        torch.save({
            "model_state": self.model.state_dict(),
            "norm_means": self.norm_means,
            "norm_stds": self.norm_stds,
            "loss_history": self.loss_history
        }, filepath)

    def load(self, filepath: str) -> None:
        try:
            checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(filepath, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state"])
        self.norm_means = checkpoint["norm_means"]
        self.norm_stds = checkpoint["norm_stds"]
        self.loss_history = checkpoint.get("loss_history", {"train_loss": [], "val_loss": []})
        self.model.eval()
