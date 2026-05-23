"""
Main Anomaly Detector Pipeline
================================
Orchestrates the full pipeline:
  1. Data generation (or load existing)
  2. Preprocessing & feature engineering
  3. Isolation Forest training + prediction
  4. LSTM Autoencoder training + prediction
  5. Ensemble evaluation + comparison
  6. Visualization (all figures)
  7. Export results to CSV + JSON

Run:
    python main.py
    python main.py --skip-train   (use saved models)
    python main.py --no-plots     (skip visualization)
"""

import argparse
import json
import os
import numpy as np
import pandas as pd

from src.data_generator   import generate_dataset, save_dataset
from src.preprocessor     import PipelinePreprocessor, FEATURE_COLS
from src.models           import (
    IsolationForestDetector,
    LSTMAutoencoderDetector,
    EnsembleDetector,
    compare_models,
)
from src.visualizer import (
    plot_fig2_swarm,
    plot_fig3_histogram,
    plot_anomaly_scores,
    plot_model_comparison,
    plot_build_performance_table,
)

DATA_PATH    = "data/raw/pipeline_metrics.csv"
RESULTS_DIR  = "results"
WINDOW_SIZE  = 10


def parse_args():
    p = argparse.ArgumentParser(description="CI/CD Pipeline Anomaly Detection")
    p.add_argument("--skip-train", action="store_true",
                   help="Load pre-trained models instead of training")
    p.add_argument("--no-plots",   action="store_true",
                   help="Skip plot generation")
    p.add_argument("--epochs", type=int, default=40,
                   help="LSTM training epochs (default: 40)")
    return p.parse_args()


def main():
    args   = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── 1. Data ────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  STEP 1 — Data Generation")
    print("═" * 60)
    if not os.path.exists(DATA_PATH):
        df = generate_dataset()
        save_dataset(df, DATA_PATH)
    else:
        df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
        print(f"[main] Loaded existing dataset: {len(df)} records from {DATA_PATH}")

    y_true = df["is_anomaly"].values

    # ── 2. Preprocessing ───────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  STEP 2 — Preprocessing")
    print("═" * 60)
    prep     = PipelinePreprocessor(window_size=WINDOW_SIZE)
    X_scaled = prep.fit_transform(df)
    X_seq    = prep.make_sequences(X_scaled)        # shape: (N-W+1, W, n_features)
    y_seq    = y_true[WINDOW_SIZE - 1:]             # aligned sequence labels
    print(f"[main] X_flat: {X_scaled.shape} | X_seq: {X_seq.shape}")

    # ── 3. Isolation Forest ────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  STEP 3 — Isolation Forest")
    print("═" * 60)
    if_detector = IsolationForestDetector()
    if args.skip_train and os.path.exists("results/isolation_forest.pkl"):
        if_detector.load()
        print("[main] Loaded pre-trained Isolation Forest")
    else:
        # Train only on 'normal' samples to learn normal behaviour
        normal_mask = y_true == 0
        if_detector.fit(X_scaled[normal_mask])
    if_results = if_detector.evaluate(X_scaled, y_true)
    print(f"[IsolationForest] {if_results}")

    # ── 4. LSTM Autoencoder ────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  STEP 4 — LSTM Autoencoder")
    print("═" * 60)
    lstm_detector = LSTMAutoencoderDetector(
        window_size=WINDOW_SIZE,
        n_features=len(FEATURE_COLS),
        epochs=args.epochs,
    )
    if args.skip_train and os.path.exists("results/lstm_model/model.keras"):
        lstm_detector.load()
        print("[main] Loaded pre-trained LSTM model")
    else:
        # Train on normal sequences only
        normal_seq_mask = y_seq == 0
        lstm_detector.fit(X_seq[normal_seq_mask])
    lstm_results = lstm_detector.evaluate(X_seq, y_seq)
    print(f"[LSTM] {lstm_results}")

    # ── 5. Ensemble ────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  STEP 5 — Ensemble Model")
    print("═" * 60)
    ensemble = EnsembleDetector(
        if_detector   = if_detector,
        lstm_detector = lstm_detector,
        window_size   = WINDOW_SIZE,
    )
    ensemble_results = ensemble.evaluate(X_scaled, X_seq, y_true)

    # Threshold baseline for comparison
    threshold_results = _threshold_baseline(df, y_true)

    all_results = [threshold_results, if_results, lstm_results, ensemble_results]
    compare_models(all_results)

    # ── 6. Visualizations ─────────────────────────────────────────────────
    if not args.no_plots:
        print("\n" + "═" * 60)
        print("  STEP 6 — Visualizations")
        print("═" * 60)
        ensemble_scores = ensemble.score_samples(X_scaled, X_seq)

        plot_fig2_swarm(df)
        plot_fig3_histogram(df)
        plot_anomaly_scores(df, ensemble_scores)
        plot_model_comparison(all_results)
        plot_build_performance_table(df)
        print("[main] All figures saved → results/")

    # ── 7. Export Results ──────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  STEP 7 — Export Results")
    print("═" * 60)
    # Add predictions back to dataframe
    df["if_anomaly"]       = if_detector.predict(X_scaled)
    df["if_score"]         = if_detector.score_samples(X_scaled)
    lstm_preds_full        = np.zeros(len(df), dtype=int)
    lstm_scores_full       = np.zeros(len(df))
    lstm_preds             = lstm_detector.predict(X_seq)
    lstm_scores_vals       = lstm_detector.score_samples(X_seq)
    lstm_preds_full[WINDOW_SIZE - 1:]  = lstm_preds
    lstm_scores_full[WINDOW_SIZE - 1:] = lstm_scores_vals
    df["lstm_anomaly"]     = lstm_preds_full
    df["lstm_score"]       = lstm_scores_full
    df["ensemble_score"]   = ensemble.score_samples(X_scaled, X_seq)
    df["ensemble_anomaly"] = ensemble.predict(X_scaled, X_seq)

    out_csv = os.path.join(RESULTS_DIR, "predictions.csv")
    df.to_csv(out_csv, index=False)
    print(f"[main] Predictions saved → {out_csv}")

    out_json = os.path.join(RESULTS_DIR, "model_results.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[main] Model results saved → {out_json}")

    print("\n✅  Experiment complete. Check the results/ folder.")
    return df, all_results


# ─────────────────────────────────────────────────────────────────────────────
def _threshold_baseline(df: pd.DataFrame, y_true: np.ndarray) -> dict:
    """Simple threshold baseline: flag if build_duration > mean + 2*std."""
    from sklearn.metrics import precision_score, recall_score, f1_score
    mu    = df["build_duration"].mean()
    sigma = df["build_duration"].std()
    preds = (df["build_duration"] > mu + 2 * sigma).astype(int).values
    return {
        "model":     "Threshold (Baseline)",
        "precision": round(precision_score(y_true, preds, zero_division=0), 4),
        "recall":    round(recall_score(y_true, preds,    zero_division=0), 4),
        "f1":        round(f1_score(y_true, preds,        zero_division=0), 4),
        "latency_ms": 10,
    }


if __name__ == "__main__":
    main()
