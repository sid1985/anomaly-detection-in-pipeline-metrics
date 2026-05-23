"""
Synthetic Pipeline Metrics Data Generator
==========================================
Generates a dataset of 429 pipeline execution records matching the paper:
  "Anomaly Detection in Pipeline Metrics: Employing AI to Detect Anomalies
   in CI/CD Pipeline Metrics"

Dataset schema:
  - instance_id          : unique execution trace ID
  - timestamp            : execution timestamp (3-month window)
  - build_duration       : build+package time in seconds
  - test_execution_time  : automated test suite time in seconds
  - deployment_frequency : deployments per day in rolling window
  - resource_utilization : CPU/memory utilisation % per run
  - is_anomaly           : ground-truth label (1 = anomaly)
  - anomaly_type         : category of anomaly (or 'normal')

Anomaly categories (matching paper findings):
  1. spike      - sudden spike in test_duration (flaky tests)       → instances ~200-300
  2. drift      - gradual build slowdown (dependency bloat)         → instances ~200-300
  3. frequency  - deployment frequency drop (misconfigured creds)   → scattered
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

SEED = 42
N_INSTANCES = 429
N_ANOMALIES_TARGET = 23          # matches paper: 23 real anomalies
ANOMALY_BAND_START = 200         # infrastructure instability phase in paper
ANOMALY_BAND_END   = 300


def generate_dataset(seed: int = SEED, n: int = N_INSTANCES) -> pd.DataFrame:
    """Generate synthetic CI/CD pipeline execution metrics dataset."""
    rng = np.random.default_rng(seed)
    random.seed(seed)

    base_start = datetime(2024, 1, 1, 8, 0, 0)
    records = []

    # Pre-assign anomaly positions to match paper (23 anomalies, clustered 201-300)
    anomaly_pool = list(range(ANOMALY_BAND_START, ANOMALY_BAND_END))  # 100 slots
    spike_indices     = set(sorted(random.sample(anomaly_pool, 8)))   # 8 flaky-test spikes
    drift_indices     = set(sorted(random.sample(anomaly_pool, 9)))   # 9 drift instances
    freq_drop_indices = set(random.sample(range(n), 6))               # 6 frequency drops (anywhere)
    all_anomaly_idx   = spike_indices | drift_indices | freq_drop_indices

    # Rolling baseline for drift simulation (instances 200-300 have elevated baseline)
    def _build_duration_normal(idx):
        """Normal build duration: 300-450s with ±30s daily noise."""
        base = 360.0
        daily_noise = rng.normal(0, 30)
        return float(np.clip(base + daily_noise, 280, 460))

    def _drift_factor(idx):
        """Gradual slowdown: ramps up over 200-300, peaks at instance 250."""
        if ANOMALY_BAND_START <= idx < ANOMALY_BAND_END:
            progress = (idx - ANOMALY_BAND_START) / 50  # 0→2
            return 1.0 + min(progress, 2.0) * 0.3       # up to +60%
        return 1.0

    deployment_freq_base = 4.0   # deployments per day baseline

    for i in range(n):
        ts = base_start + timedelta(hours=i * 0.7)  # ~30-min cadence

        # ── Build Duration ──────────────────────────────────────────────────
        bd = _build_duration_normal(i) * _drift_factor(i)
        is_spike = i in spike_indices
        is_drift = i in drift_indices and i not in spike_indices
        is_freq  = i in freq_drop_indices

        if is_spike:
            bd = rng.uniform(850, 1200)        # spike: strong outlier (paper max 780, boosted)
        elif is_drift:
            bd = bd * rng.uniform(1.5, 1.9)    # drift: +50-90% on top of existing drift factor

        # ── Test Execution Time  (correlated with build duration) ───────────
        tet = bd * rng.uniform(0.55, 0.72) + rng.normal(0, 20)
        if is_spike:
            tet = bd * rng.uniform(0.80, 0.95)  # flaky tests inflate test time

        # ── Deployment Frequency ────────────────────────────────────────────
        df = deployment_freq_base + rng.normal(0, 0.5)
        if is_freq:
            df = rng.uniform(0.0, 0.25)         # near-zero (misconfigured credentials)
        elif ANOMALY_BAND_START <= i < ANOMALY_BAND_END:
            df = max(0.5, df - (bd - 360) / 200)  # slight suppression during drift

        # ── Resource Utilization (%) ────────────────────────────────────────
        ru = 62 + (bd - 360) / 20 + rng.normal(0, 5)
        ru = float(np.clip(ru, 30, 99))

        # ── Error Count  ────────────────────────────────────────────────────
        error_count = int(max(0, rng.poisson(0.5)))
        if is_spike:
            error_count = int(max(error_count, rng.poisson(8)))   # spikes cause many errors
        elif is_drift:
            error_count = int(max(error_count, rng.poisson(4)))   # drift causes moderate errors
        elif ANOMALY_BAND_START <= i < ANOMALY_BAND_END:
            error_count = int(max(error_count, rng.poisson(2)))   # elevated in instability band

        # ── Labels ──────────────────────────────────────────────────────────
        is_anomaly = 1 if i in all_anomaly_idx else 0
        if is_spike:
            atype = "spike"
        elif is_drift:
            atype = "drift"
        elif is_freq:
            atype = "frequency_drop"
        else:
            atype = "normal"

        records.append({
            "instance_id":          i + 1,
            "timestamp":            ts,
            "build_duration":       round(bd, 2),
            "test_execution_time":  round(max(50, tet), 2),
            "deployment_frequency": round(max(0, df), 3),
            "resource_utilization": round(ru, 2),
            "error_count":          error_count,
            "is_anomaly":           is_anomaly,
            "anomaly_type":         atype,
        })

    df_out = pd.DataFrame(records)
    return df_out


def save_dataset(df: pd.DataFrame, path: str = "data/raw/pipeline_metrics.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[data_generator] Saved {len(df)} records → {path}")
    print(f"  Anomalies : {df['is_anomaly'].sum()} / {len(df)}")
    print(f"  Types     : {df['anomaly_type'].value_counts().to_dict()}")


if __name__ == "__main__":
    df = generate_dataset()
    save_dataset(df)
    print(df.describe())
