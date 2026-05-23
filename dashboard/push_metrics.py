"""
Prometheus Metrics Pusher
==========================
After running main.py, push model results and pipeline metrics
to the Prometheus Pushgateway for Grafana visualization.

Run:  python dashboard/push_metrics.py
Requires: dashboard/docker-compose.yml stack running
"""

import json
import os
import pandas as pd
from urllib import request, error


PUSHGATEWAY_URL = "http://localhost:9091"
JOB_NAME        = "pipeline_anomaly"

# Resolve paths relative to the project root (one level up from this script)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def push_metric(metric_name: str, value: float, labels: dict = None) -> None:
    """Push a single metric to Prometheus Pushgateway."""
    label_str = ",".join(f'{k}="{v}"' for k, v in (labels or {}).items())
    if label_str:
        full_name = f'{metric_name}{{{label_str}}}'
    else:
        full_name = metric_name

    body = f"# TYPE {metric_name} gauge\n{full_name} {value}\n"
    url  = f"{PUSHGATEWAY_URL}/metrics/job/{JOB_NAME}"
    req  = request.Request(url, data=body.encode(), method="POST",
                           headers={"Content-Type": "text/plain"})
    try:
        with request.urlopen(req, timeout=5) as resp:
            pass
    except error.URLError as e:
        print(f"[push_metrics] Could not reach Pushgateway: {e}")
        print("  Is the Docker stack running? cd dashboard && docker-compose up -d")


def push_all():
    """Push model results and per-run metrics to Pushgateway."""

    # ── Model comparison metrics ─────────────────────────────────────────
    results_path = os.path.join(_PROJECT_ROOT, "results", "model_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)

        model_label_map = {
            "Threshold (Baseline)": "threshold",
            "IsolationForest":      "isolation_forest",
            "LSTM_Autoencoder":     "lstm",
            "Ensemble(IF+LSTM)":    "ensemble",
        }

        for r in results:
            label = model_label_map.get(r["model"], r["model"].lower().replace(" ", "_"))
            push_metric("pipeline_model_f1_score",      r["f1"],       {"model": label})
            push_metric("pipeline_model_precision",     r["precision"], {"model": label})
            push_metric("pipeline_model_recall",        r["recall"],    {"model": label})
            push_metric("pipeline_model_latency_ms",    r["latency_ms"],{"model": label})

            if "fpr" in r:
                push_metric("pipeline_false_positive_rate", r["fpr"])

        print(f"[push_metrics] Pushed model comparison metrics for {len(results)} models")

    # ── Per-run pipeline metrics ─────────────────────────────────────────
    preds_path = os.path.join(_PROJECT_ROOT, "results", "predictions.csv")
    if os.path.exists(preds_path):
        df = pd.read_csv(preds_path)
        # Push last 50 runs as individual metrics
        for _, row in df.tail(50).iterrows():
            labels = {"instance": str(int(row["instance_id"]))}
            push_metric("pipeline_build_duration_seconds",
                        row["build_duration"], labels)
            push_metric("pipeline_anomaly_score",
                        row["ensemble_score"],  {**labels, "model": "ensemble"})
            push_metric("pipeline_anomaly_detected_total",
                        row["ensemble_anomaly"], labels)
            push_metric("pipeline_resource_utilization",
                        row["resource_utilization"], labels)

        push_metric("pipeline_anomaly_threshold_seconds", 600.0)
        print(f"[push_metrics] Pushed {min(50, len(df))} pipeline run metrics")

    print("\n[push_metrics] Done. Open Grafana at http://localhost:3000")


if __name__ == "__main__":
    push_all()
