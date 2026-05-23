"""
Tests for the Anomaly Detection Pipeline
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data_generator import generate_dataset
from src.preprocessor   import PipelinePreprocessor
from src.models         import IsolationForestDetector


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def df():
    return generate_dataset(seed=42)


@pytest.fixture(scope="module")
def preprocessor():
    return PipelinePreprocessor(window_size=10)


@pytest.fixture(scope="module")
def X_scaled(df, preprocessor):
    return preprocessor.fit_transform(df)


# ── Data Generator Tests ──────────────────────────────────────────────────────
class TestDataGenerator:
    def test_row_count(self, df):
        assert len(df) == 429, f"Expected 429 rows, got {len(df)}"

    def test_required_columns(self, df):
        required = [
            "instance_id", "timestamp", "build_duration",
            "test_execution_time", "deployment_frequency",
            "resource_utilization", "error_count",
            "is_anomaly", "anomaly_type",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_anomaly_count(self, df):
        n_anomalies = df["is_anomaly"].sum()
        # Should be close to 23 (paper target)
        assert 15 <= n_anomalies <= 35, \
            f"Expected ~23 anomalies, got {n_anomalies}"

    def test_instance_ids_unique(self, df):
        assert df["instance_id"].nunique() == len(df)

    def test_build_duration_range(self, df):
        assert df["build_duration"].min() >= 200
        # Spike anomalies go up to 1200s; drift in band 200-300 can reach ~740s
        assert df["build_duration"].max() <= 1200

    def test_resource_utilization_clipped(self, df):
        assert df["resource_utilization"].between(0, 100).all()

    def test_anomaly_in_band_201_300(self, df):
        """Paper states infrastructure instability in instances 201-300."""
        bad_band  = df[(df["instance_id"] >= 201) & (df["instance_id"] <= 300)]
        good_band = df[~((df["instance_id"] >= 201) & (df["instance_id"] <= 300))]
        assert bad_band["is_anomaly"].mean() > good_band["is_anomaly"].mean(), \
            "Anomaly rate should be higher in instances 201-300"

    def test_anomaly_types(self, df):
        valid_types = {"normal", "spike", "drift", "frequency_drop"}
        assert set(df["anomaly_type"].unique()).issubset(valid_types)


# ── Preprocessor Tests ────────────────────────────────────────────────────────
class TestPreprocessor:
    def test_output_shape(self, df, X_scaled):
        assert X_scaled.shape == (429, 5), \
            f"Expected (429, 5), got {X_scaled.shape}"

    def test_zero_mean_approx(self, X_scaled):
        """StandardScaler should produce ~0 mean."""
        col_means = X_scaled.mean(axis=0)
        assert np.allclose(col_means, 0, atol=0.01), \
            f"Means not ~0: {col_means}"

    def test_unit_variance_approx(self, X_scaled):
        col_stds = X_scaled.std(axis=0)
        assert np.allclose(col_stds, 1, atol=0.05), \
            f"Stds not ~1: {col_stds}"

    def test_no_nans(self, X_scaled):
        assert not np.isnan(X_scaled).any(), "NaNs found in scaled data"

    def test_sequence_shape(self, X_scaled, preprocessor):
        X_seq = preprocessor.make_sequences(X_scaled)
        expected = (429 - 10 + 1, 10, 5)   # (420, 10, 5)
        assert X_seq.shape == expected, \
            f"Expected {expected}, got {X_seq.shape}"

    def test_zscore(self, preprocessor):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        z    = preprocessor.zscore(data)
        assert abs(z.mean()) < 1e-9
        assert abs(z.std() - 1.0) < 1e-6


# ── Isolation Forest Tests ────────────────────────────────────────────────────
class TestIsolationForest:
    def test_fit_and_predict(self, X_scaled, df):
        ifd    = IsolationForestDetector(contamination=0.054, n_estimators=50)
        normal = X_scaled[df["is_anomaly"].values == 0]
        ifd.fit(normal)
        preds  = ifd.predict(X_scaled)
        assert preds.shape == (429,)
        assert set(preds).issubset({0, 1})

    def test_anomaly_score_range(self, X_scaled, df):
        ifd    = IsolationForestDetector(contamination=0.054, n_estimators=50)
        normal = X_scaled[df["is_anomaly"].values == 0]
        ifd.fit(normal)
        scores = ifd.score_samples(X_scaled)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_spikes_score_higher(self, X_scaled, df):
        """Anomalous instances should have higher anomaly scores on average."""
        ifd    = IsolationForestDetector(contamination=0.054, n_estimators=100)
        normal = X_scaled[df["is_anomaly"].values == 0]
        ifd.fit(normal)
        scores   = ifd.score_samples(X_scaled)
        anom_avg = scores[df["is_anomaly"].values == 1].mean()
        norm_avg = scores[df["is_anomaly"].values == 0].mean()
        assert anom_avg > norm_avg, \
            f"Anomaly score avg ({anom_avg:.4f}) should > normal avg ({norm_avg:.4f})"

    def test_evaluate_returns_keys(self, X_scaled, df):
        ifd    = IsolationForestDetector(contamination=0.054, n_estimators=50)
        ifd.fit(X_scaled[df["is_anomaly"].values == 0])
        result = ifd.evaluate(X_scaled, df["is_anomaly"].values)
        for key in ["model", "precision", "recall", "f1"]:
            assert key in result
