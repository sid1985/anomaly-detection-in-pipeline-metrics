"""
ACA Pipeline Runner
====================
Simulates 50 real-style CI/CD pipeline runs against the deployed ACA API,
compares results to the synthetic baseline, and pushes metrics to Pushgateway.

Usage:
    python dashboard/run_aca_pipelines.py --url https://<aca-fqdn>
    python dashboard/run_aca_pipelines.py --url http://localhost:8000  # local

    # CI / GitHub Actions
    python dashboard/run_aca_pipelines.py \
        --url https://<aca-fqdn> \
        --output-json results/aca_test_results.json \
        --fail-below-f1 0.70

Env var alternative: ACA_API_URL=https://...
"""

import argparse
import datetime
import json
import os
import sys
import time
import random
import urllib.request
import urllib.error
from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np

PUSHGATEWAY_URL  = "http://localhost:9091"
JOB_NAME         = "aca_pipelines"
RANDOM_SEED      = 99

# Paper-reported targets for Isolation Forest (Table 2)
PAPER_IF_TARGETS = {"precision": 0.88, "recall": 0.82, "f1": 0.85, "fpr": 0.04}

# ── Pipeline run definitions ─────────────────────────────────────────────────
@dataclass
class PipelineRun:
    build_duration:       float
    test_execution_time:  float
    deployment_frequency: float
    resource_utilization: float
    error_count:          int
    true_label:           int        # 1 = anomaly, 0 = normal
    anomaly_kind:         str = "normal"


def generate_runs(seed: int = RANDOM_SEED) -> List[PipelineRun]:
    """Generate 50 labelled pipeline runs matching the paper's distribution."""
    rng = random.Random(seed)
    runs = []

    # 30 normal runs
    for _ in range(30):
        runs.append(PipelineRun(
            build_duration       = rng.uniform(180, 450),
            test_execution_time  = rng.uniform(40,  110),
            deployment_frequency = rng.uniform(2.0, 6.0),
            resource_utilization = rng.uniform(35,  72),
            error_count          = rng.randint(0, 1),
            true_label           = 0,
            anomaly_kind         = "normal",
        ))

    # 8 spike anomalies (build duration blows out)
    for _ in range(8):
        runs.append(PipelineRun(
            build_duration       = rng.uniform(850, 1250),
            test_execution_time  = rng.uniform(180, 350),
            deployment_frequency = rng.uniform(1.5, 3.5),
            resource_utilization = rng.uniform(75,  95),
            error_count          = rng.randint(6,  15),
            true_label           = 1,
            anomaly_kind         = "spike",
        ))

    # 7 drift anomalies (resource utilization creeping up)
    for i in range(7):
        drift = 1.5 + (i * 0.06)
        runs.append(PipelineRun(
            build_duration       = rng.uniform(350, 600) * drift,
            test_execution_time  = rng.uniform(90,  160) * drift,
            deployment_frequency = rng.uniform(1.0, 2.5),
            resource_utilization = min(98, rng.uniform(55, 75) * drift),
            error_count          = rng.randint(3,  8),
            true_label           = 1,
            anomaly_kind         = "drift",
        ))

    # 5 frequency-drop anomalies
    for _ in range(5):
        runs.append(PipelineRun(
            build_duration       = rng.uniform(180, 420),
            test_execution_time  = rng.uniform(40,  100),
            deployment_frequency = rng.uniform(0.05, 0.25),
            resource_utilization = rng.uniform(35,  65),
            error_count          = rng.randint(0,  2),
            true_label           = 1,
            anomaly_kind         = "frequency_drop",
        ))

    rng.shuffle(runs)
    return runs


# ── API call ─────────────────────────────────────────────────────────────────
def call_api(url: str, run: PipelineRun) -> dict:
    payload = {
        "build_duration":       run.build_duration,
        "test_execution_time":  run.test_execution_time,
        "deployment_frequency": run.deployment_frequency,
        "resource_utilization": run.resource_utilization,
        "error_count":          run.error_count,
    }
    body = json.dumps(payload).encode()
    req  = urllib.request.Request(
        f"{url}/predict",
        data    = body,
        method  = "POST",
        headers = {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


# ── Pushgateway helper ────────────────────────────────────────────────────────
def push_metric(name: str, value: float, labels: dict = None) -> None:
    label_str = ",".join(f'{k}="{v}"' for k, v in (labels or {}).items())
    full_name = f"{name}{{{label_str}}}" if label_str else name
    body = f"# TYPE {name} gauge\n{full_name} {value}\n".encode()
    url  = f"{PUSHGATEWAY_URL}/metrics/job/{JOB_NAME}"
    req  = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "text/plain"},
    )
    try:
        urllib.request.urlopen(req, timeout=5)
    except urllib.error.URLError:
        pass  # Pushgateway optional


# ── Metrics calculation ───────────────────────────────────────────────────────
def compute_metrics(true_labels: list, pred_labels: list) -> dict:
    tp = sum(t == 1 and p == 1 for t, p in zip(true_labels, pred_labels))
    fp = sum(t == 0 and p == 1 for t, p in zip(true_labels, pred_labels))
    fn = sum(t == 1 and p == 0 for t, p in zip(true_labels, pred_labels))
    tn = sum(t == 0 and p == 0 for t, p in zip(true_labels, pred_labels))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return dict(precision=precision, recall=recall, f1=f1, fpr=fpr,
                tp=tp, fp=fp, fn=fn, tn=tn)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Run ACA pipeline experiments")
    parser.add_argument("--url", default=os.environ.get("ACA_API_URL", "http://localhost:8000"),
                        help="ACA API base URL (no trailing slash)")
    parser.add_argument("--output-json", metavar="PATH", default=None,
                        help="Write full results JSON to this file (optional)")
    parser.add_argument("--fail-below-f1", type=float, default=None,
                        metavar="THRESHOLD",
                        help="Exit with code 1 if F1 < THRESHOLD (for CI gate)")
    args = parser.parse_args()
    api_url = args.url.rstrip("/")

    # Health check
    print(f"\n[ACA Runner] Connecting to: {api_url}")
    try:
        hc = urllib.request.urlopen(f"{api_url}/health", timeout=10)
        print(f"[ACA Runner] Health: {json.loads(hc.read())}")
    except Exception as e:
        print(f"[ACA Runner] ERROR: cannot reach API → {e}")
        sys.exit(1)

    runs = generate_runs()
    print(f"[ACA Runner] Sending {len(runs)} pipeline runs "
          f"({sum(r.true_label for r in runs)} anomalies) ...\n")

    true_labels, pred_labels, latencies = [], [], []
    per_kind: dict = {"normal": [], "spike": [], "drift": [], "frequency_drop": []}

    for i, run in enumerate(runs, 1):
        t0 = time.time()
        try:
            result = call_api(api_url, run)
        except Exception as e:
            print(f"  run {i:02d}: ERROR {e}")
            continue
        elapsed_ms = (time.time() - t0) * 1000

        pred = int(result["is_anomaly"])
        true_labels.append(run.true_label)
        pred_labels.append(pred)
        latencies.append(elapsed_ms)
        per_kind[run.anomaly_kind].append((run.true_label, pred))

        status = "✓ TP" if run.true_label==1 and pred==1 else \
                 "✗ FN" if run.true_label==1 and pred==0 else \
                 "✗ FP" if run.true_label==0 and pred==1 else "  TN"
        print(f"  run {i:02d} [{run.anomaly_kind:<14s}] "
              f"score={result['anomaly_score']:.4f}  pred={'ANOM' if pred else 'norm'}  {status}  {elapsed_ms:.0f}ms")

    print()

    # ── Overall metrics ───────────────────────────────────────────────────────
    m = compute_metrics(true_labels, pred_labels)
    avg_lat = sum(latencies) / len(latencies) if latencies else 0

    print("=" * 60)
    print("  ACA LIVE RESULTS (Isolation Forest, real API calls)")
    print("=" * 60)
    print(f"  Precision : {m['precision']:.4f}")
    print(f"  Recall    : {m['recall']:.4f}")
    print(f"  F1 Score  : {m['f1']:.4f}")
    print(f"  FPR       : {m['fpr']:.4f}")
    print(f"  TP/FP/FN/TN: {m['tp']}/{m['fp']}/{m['fn']}/{m['tn']}")
    print(f"  Avg Latency: {avg_lat:.1f} ms")

    # ── Per anomaly kind breakdown ────────────────────────────────────────────
    print("\n  Per-kind detection rate:")
    for kind, pairs in per_kind.items():
        positives = [p for t, p in pairs if t == 1]
        if positives:
            det = sum(positives) / len(positives)
            print(f"    {kind:<16s}: {sum(positives)}/{len(positives)} detected ({det:.0%})")

    # ── Load synthetic baseline for comparison ────────────────────────────────
    results_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "results", "model_results.json")
    synthetic = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            for r in json.load(f):
                synthetic[r["model"]] = r

    if_syn  = synthetic.get("IsolationForest", {})
    ens_syn = synthetic.get("Ensemble(IF+LSTM)", {})
    print("\n" + "=" * 60)
    print("  COMPARISON: Synthetic dataset  vs  ACA live")
    print("=" * 60)
    print(f"  {'Metric':<12}  {'Synth IF':>9}  {'Synth Ens':>10}  {'ACA IF':>9}  {'Paper IF':>9}")
    print(f"  {'-'*60}")
    print(f"  {'Precision':<12}  {if_syn.get('precision',0):>9.4f}  {ens_syn.get('precision',0):>10.4f}  {m['precision']:>9.4f}  {'0.89':>9}")
    print(f"  {'Recall':<12}  {if_syn.get('recall',0):>9.4f}  {ens_syn.get('recall',0):>10.4f}  {m['recall']:>9.4f}  {'0.82':>9}")
    print(f"  {'F1':<12}  {if_syn.get('f1',0):>9.4f}  {ens_syn.get('f1',0):>10.4f}  {m['f1']:>9.4f}  {'0.85':>9}")
    print(f"  {'FPR':<12}  {'—':>9}  {ens_syn.get('fpr',0):>10.4f}  {m['fpr']:>9.4f}  {'0.04':>9}")
    print("=" * 60)

    # ── Push to Prometheus Pushgateway ────────────────────────────────────────
    print("\n[ACA Runner] Pushing ACA metrics to Pushgateway...")
    push_metric("aca_model_f1_score",       m["f1"],        {"model": "isolation_forest_aca"})
    push_metric("aca_model_precision",      m["precision"],  {"model": "isolation_forest_aca"})
    push_metric("aca_model_recall",         m["recall"],     {"model": "isolation_forest_aca"})
    push_metric("aca_model_fpr",            m["fpr"],        {"model": "isolation_forest_aca"})
    push_metric("aca_model_latency_ms",     avg_lat,         {"model": "isolation_forest_aca"})
    push_metric("aca_run_count_total",      len(runs))
    push_metric("aca_anomaly_detected",     sum(pred_labels))
    push_metric("aca_anomaly_groundtruth",  sum(true_labels))
    push_metric("aca_true_positives",       m["tp"])
    push_metric("aca_false_positives",      m["fp"])
    push_metric("aca_false_negatives",      m["fn"])

    # Push per-kind detection
    for kind, pairs in per_kind.items():
        positives = [p for t, p in pairs if t == 1]
        if positives:
            push_metric("aca_detection_rate",
                        sum(positives) / len(positives),
                        {"anomaly_kind": kind})

    # Also push synthetic IF for side-by-side panel in Grafana
    if if_syn:
        push_metric("aca_model_f1_score",   if_syn["f1"],       {"model": "isolation_forest_synthetic"})
        push_metric("aca_model_precision",  if_syn["precision"], {"model": "isolation_forest_synthetic"})
        push_metric("aca_model_recall",     if_syn["recall"],    {"model": "isolation_forest_synthetic"})

    print("[ACA Runner] Done. Check Grafana at http://localhost:3000")
    print("[ACA Runner] Pushgateway: http://localhost:9091")

    # ── JSON output ───────────────────────────────────────────────────────────
    per_kind_stats: dict = {}
    for kind, pairs in per_kind.items():
        total_pos = [p for t, p in pairs if t == 1]
        per_kind_stats[kind] = {
            "detected": sum(total_pos),
            "total":    len(total_pos),
            "rate":     (sum(total_pos) / len(total_pos)) if total_pos else None,
        }

    passed = (m["f1"] >= args.fail_below_f1) if args.fail_below_f1 is not None else True

    results_payload = {
        "run_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "api_url":       api_url,
        "commit_sha":    os.environ.get("GITHUB_SHA", "local"),
        "metric_note":   os.environ.get("METRIC_RUN_NOTE", ""),
        "metrics": {
            "precision": round(m["precision"], 6),
            "recall":    round(m["recall"],    6),
            "f1":        round(m["f1"],        6),
            "fpr":       round(m["fpr"],       6),
            "tp": m["tp"], "fp": m["fp"],
            "fn": m["fn"], "tn": m["tn"],
        },
        "latency": {
            "avg_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
            "min_ms": round(min(latencies), 2)                  if latencies else 0,
            "max_ms": round(max(latencies), 2)                  if latencies else 0,
        },
        "per_kind":      per_kind_stats,
        "paper_targets": PAPER_IF_TARGETS,
        "fail_threshold_f1": args.fail_below_f1,
        "passed":        passed,
        "total_runs":    len(true_labels),
        "total_anomalies_ground_truth": sum(true_labels),
        "total_anomalies_predicted":    sum(pred_labels),
    }

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w") as jf:
            json.dump(results_payload, jf, indent=2)
        print(f"\n[ACA Runner] Results written to: {args.output_json}")

    # ── GitHub Actions step summary ───────────────────────────────────────────
    gha_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if gha_summary:
        _write_gha_summary(gha_summary, results_payload)

    # ── CI gate: fail if F1 below threshold ───────────────────────────────────
    if args.fail_below_f1 is not None and m["f1"] < args.fail_below_f1:
        print(f"\n[ACA Runner] FAIL: F1={m['f1']:.4f} < threshold {args.fail_below_f1}")
        sys.exit(1)


def _write_gha_summary(summary_path: str, r: dict) -> None:
    """Append a Markdown table to the GitHub Actions step summary file."""
    m   = r["metrics"]
    lat = r["latency"]
    pk  = r["per_kind"]

    lines = [
        "## ACA Integration Test Results\n",
        f"**API:** {r['api_url']}  \n",
        f"**Commit:** `{r['commit_sha'][:7] if len(r['commit_sha']) >= 7 else r['commit_sha']}`\n\n",
        "### Performance vs Paper (Isolation Forest, Table 2)\n\n",
        "| Metric | This Run | Paper Target | Status |\n",
        "|--------|----------|:------------:|:------:|\n",
    ]
    targets = [("Precision", "precision", 0.88),
               ("Recall",    "recall",    0.82),
               ("F1",        "f1",        0.85),
               ("FPR",       "fpr",       0.04)]
    for label, key, target in targets:
        val = m.get(key, 0)
        # FPR: lower is better; others: higher is better
        ok  = ("✅" if val <= target * 1.5 else "⚠️") if key == "fpr" \
              else ("✅" if val >= target * 0.85 else "⚠️")
        lines.append(f"| {label} | {val:.4f} | {target:.2f} | {ok} |\n")

    lines += [
        f"\n**Avg latency:** {lat['avg_ms']:.1f} ms"
        f" · min {lat['min_ms']:.1f} ms · max {lat['max_ms']:.1f} ms\n\n",
        "### Detection Rate by Anomaly Type\n\n",
        "| Type | Detected | Total | Rate |\n",
        "|------|:--------:|:-----:|:----:|\n",
    ]
    for kind, stats in pk.items():
        total = stats.get("total", 0)
        if total:
            rate = stats.get("rate") or 0
            icon = "✅" if rate >= 0.75 else "⚠️"
            lines.append(
                f"| {kind} | {stats['detected']} | {total} | {rate:.0%} {icon} |\n"
            )

    lines += [
        "\n### Confusion Matrix\n\n",
        "| | Predicted Anomaly | Predicted Normal |\n",
        "|--|:----------------:|:----------------:|\n",
        f"| **Actual Anomaly** | TP = {m['tp']} | FN = {m['fn']} |\n",
        f"| **Actual Normal**  | FP = {m['fp']} | TN = {m['tn']} |\n",
        f"\n**Overall:** {'✅ PASSED' if r['passed'] else '❌ FAILED'}\n",
    ]

    with open(summary_path, "a") as f:
        f.writelines(lines)


if __name__ == "__main__":
    main()
