"""
Preprocessing Pipeline
========================
- Imputes missing values via linear interpolation (temporal consistency)
- Encodes categorical values
- Z-score normalisation per feature (formula 4 in paper)
- Sliding-window feature engineering for LSTM input
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

FEATURE_COLS = [
    "build_duration",
    "test_execution_time",
    "deployment_frequency",
    "resource_utilization",
    "error_count",
]
SCALER_PATH = "results/scaler.pkl"


class PipelinePreprocessor:
    def __init__(self, window_size: int = 10, scaler_path: str = SCALER_PATH):
        self.window_size  = window_size
        self.scaler_path  = scaler_path
        self.scaler       = StandardScaler()
        self._fitted      = False

    # ──────────────────────────────────────────────────────────────────────
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Clean, normalise, and return scaled feature matrix."""
        X = self._clean(df)
        X_scaled = self.scaler.fit_transform(X)
        self._fitted = True
        os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)
        joblib.dump(self.scaler, self.scaler_path)
        return X_scaled

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted scaler."""
        if not self._fitted:
            self.scaler = joblib.load(self.scaler_path)
            self._fitted = True
        X = self._clean(df)
        return self.scaler.transform(X)

    # ──────────────────────────────────────────────────────────────────────
    def _clean(self, df: pd.DataFrame) -> np.ndarray:
        """Sort by timestamp, interpolate missing values, extract features."""
        df = df.copy()
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp").reset_index(drop=True)
        # Linear interpolation for missing values (paper methodology)
        for col in FEATURE_COLS:
            if col in df.columns:
                df[col] = df[col].interpolate(method="linear").ffill().bfill()
        return df[FEATURE_COLS].values.astype(np.float64)

    # ──────────────────────────────────────────────────────────────────────
    def make_sequences(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Build overlapping sliding-window sequences for LSTM input.
        Shape: (n_samples - window_size + 1, window_size, n_features)
        """
        seqs = []
        for i in range(len(X_scaled) - self.window_size + 1):
            seqs.append(X_scaled[i : i + self.window_size])
        return np.array(seqs)

    # ──────────────────────────────────────────────────────────────────────
    def zscore(self, series: np.ndarray) -> np.ndarray:
        """Manual Z-score normalisation (paper formula 4)."""
        mu    = np.mean(series)
        sigma = np.std(series) + 1e-9
        return (series - mu) / sigma
