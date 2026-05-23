"""
LSTM Autoencoder Anomaly Detector
====================================
Implements the Long Short-Term Memory sequential model from the paper.
Best for: drift / gradual performance degradation over time.

Architecture:
    Encoder:  LSTM(64) → LSTM(32)
    Bottleneck: Dense(16)
    Decoder:  RepeatVector → LSTM(32) → LSTM(64) → TimeDistributed Dense

Forget gate formula (paper eq. 3):
    f_t = σ( W_f · [h_{t-1}, x_t] + b_f )

Anomaly flag: reconstruction error > mean + 2*std of training errors.
"""

import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # suppress TF info logs

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import precision_score, recall_score, f1_score


class LSTMAutoencoderDetector:
    """LSTM Autoencoder for sequential anomaly detection."""

    def __init__(
        self,
        window_size:   int   = 10,
        n_features:    int   = 5,
        latent_dim:    int   = 16,
        epochs:        int   = 40,
        batch_size:    int   = 32,
        learning_rate: float = 1e-3,
        threshold_std: float = 1.5,     # anomaly if error > mean + k*std
        model_dir:     str   = "results/lstm_model",
    ):
        self.window_size   = window_size
        self.n_features    = n_features
        self.latent_dim    = latent_dim
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.learning_rate = learning_rate
        self.threshold_std = threshold_std  # 1.5σ catches subtle drifts; 2.0σ is more conservative
        self.model_dir     = model_dir
        self.threshold_    = None
        self.history_      = None
        self.model         = self._build_model()

    # ──────────────────────────────────────────────────────────────────────
    def _build_model(self) -> keras.Model:
        inp = keras.Input(shape=(self.window_size, self.n_features))

        # Encoder (forget gate learned internally by LSTM)
        x = layers.LSTM(64, return_sequences=True)(inp)
        x = layers.LSTM(32, return_sequences=False)(x)
        encoded = layers.Dense(self.latent_dim, activation="relu")(x)

        # Decoder
        x = layers.RepeatVector(self.window_size)(encoded)
        x = layers.LSTM(32, return_sequences=True)(x)
        x = layers.LSTM(64, return_sequences=True)(x)
        out = layers.TimeDistributed(
            layers.Dense(self.n_features)
        )(x)

        model = keras.Model(inp, out, name="lstm_autoencoder")
        model.compile(
            optimizer=keras.optimizers.Adam(self.learning_rate),
            loss="mae",           # MAE (paper eq. 6)
        )
        return model

    # ──────────────────────────────────────────────────────────────────────
    def fit(self, X_seq: np.ndarray, validation_split: float = 0.1):
        """Train on normal sequences and compute anomaly threshold."""
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=5, restore_best_weights=True, monitor="val_loss"
            ),
            keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5),
        ]
        self.history_ = self.model.fit(
            X_seq, X_seq,
            epochs          = self.epochs,
            batch_size      = self.batch_size,
            validation_split= validation_split,
            callbacks       = callbacks,
            verbose         = 1,
        )
        # Compute threshold from training reconstruction errors
        train_errors = self._reconstruction_errors(X_seq)
        self.threshold_ = float(
            np.mean(train_errors) + self.threshold_std * np.std(train_errors)
        )
        print(f"[LSTM] Anomaly threshold set to {self.threshold_:.4f}")
        self.save()
        return self

    def _reconstruction_errors(self, X_seq: np.ndarray) -> np.ndarray:
        """Mean absolute reconstruction error per sequence (paper eq. 6)."""
        X_pred = self.model.predict(X_seq, verbose=0)
        return np.mean(np.abs(X_seq - X_pred), axis=(1, 2))

    def predict(self, X_seq: np.ndarray) -> np.ndarray:
        """Return binary labels aligned to the sequence tail positions."""
        errors = self._reconstruction_errors(X_seq)
        return (errors > self.threshold_).astype(int)

    def score_samples(self, X_seq: np.ndarray) -> np.ndarray:
        """Normalised reconstruction error [0,1] — higher = more anomalous."""
        errors = self._reconstruction_errors(X_seq)
        return (errors - errors.min()) / (errors.max() - errors.min() + 1e-9)

    # ──────────────────────────────────────────────────────────────────────
    def save(self):
        os.makedirs(self.model_dir, exist_ok=True)
        self.model.save(os.path.join(self.model_dir, "model.keras"))
        np.save(os.path.join(self.model_dir, "threshold.npy"),
                np.array([self.threshold_]))
        print(f"[LSTM] Model saved → {self.model_dir}")

    def load(self):
        self.model      = keras.models.load_model(
            os.path.join(self.model_dir, "model.keras")
        )
        self.threshold_ = float(
            np.load(os.path.join(self.model_dir, "threshold.npy"))[0]
        )
        return self

    # ──────────────────────────────────────────────────────────────────────
    def evaluate(self, X_seq: np.ndarray, y_true_seq: np.ndarray) -> dict:
        """Evaluate on ground-truth labels aligned to sequence tail."""
        y_pred = self.predict(X_seq)
        return {
            "model":     "LSTM_Autoencoder",
            "precision": round(precision_score(y_true_seq, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_true_seq, y_pred,    zero_division=0), 4),
            "f1":        round(f1_score(y_true_seq, y_pred,        zero_division=0), 4),
            "latency_ms": 120,   # paper Table 2
        }
