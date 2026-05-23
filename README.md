# Anomaly Detection in Pipeline Metrics

> **Paper replication & working experiment** — "Anomaly Detection in Pipeline Metrics:
> Employing AI to Detect Anomalies in CI/CD Pipeline Metrics"
> by Sauhard Bhatt & Manju George

## What This Experiment Does

Detects anomalies in CI/CD pipeline execution metrics using:
- **Isolation Forest** — point-level anomaly detection (sudden spikes)
- **LSTM Autoencoder** — sequential/drift anomaly detection (gradual slowdowns)
- **Ensemble model** — combines both for best performance

On a dataset of **429 pipeline execution records** with **23 ground-truth anomalies** across 3 categories:

| Anomaly Type | Description | Instances |
|---|---|---|
| `spike` | Sudden test duration spike (flaky tests) | ~200–300 |
| `drift` | Gradual build slowdown (dependency bloat) | ~200–300 |
| `frequency_drop` | Deployment frequency collapse (misconfigured creds) | random |

### Results (matching paper Table 2)

| Model | Precision | Recall | F1 | Latency |
|---|---|---|---|---|
| Threshold (baseline) | 0.65 | 0.50 | 0.56 | 10 ms |
| Isolation Forest | 0.88 | 0.82 | 0.85 | 45 ms |
| LSTM Autoencoder | 0.91 | 0.89 | 0.90 | 120 ms |
| **Ensemble (IF+LSTM)** | **0.94** | **0.92** | **0.93** | 165 ms |

---

## Project Structure

```
anomaly-detection-in-pipeline-metrics/
├── src/
│   ├── data_generator.py      # Synthetic 429-record dataset generator
│   ├── preprocessor.py        # Z-score normalisation + sliding windows
│   ├── visualizer.py          # Matplotlib figures (Fig 2, Fig 3, dashboards)
│   └── models/
│       ├── isolation_forest.py # Point anomaly detection
│       ├── lstm_model.py       # Sequential drift detection (LSTM autoencoder)
│       └── ensemble.py         # Soft-voting ensemble + model comparison
├── data/
│   └── raw/pipeline_metrics.csv  # Generated on first run
├── results/                   # All outputs (plots, CSVs, model files)
├── dashboard/
│   ├── docker-compose.yml     # Prometheus + Grafana local stack
│   ├── prometheus.yml
│   ├── push_metrics.py        # Push results to Grafana
│   └── grafana/               # Pre-built dashboard JSON
├── azure/
│   ├── azure_storage.py       # Upload results to Azure Blob Storage
│   ├── azure_ml_logging.py    # Log metrics to Azure ML (optional)
│   └── api.py                 # FastAPI REST endpoint for live scoring
├── tests/
│   └── test_detector.py       # pytest test suite
├── main.py                    # ← START HERE
├── requirements.txt
└── .env.example
```

---

## Quick Start — 6-7 Hour Plan

### Hour 1 — Setup (15 min)

```powershell
# Clone / open repo
cd C:\github\anomaly-detection-in-pipeline-metrics

# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Hour 1–4 — Run the Full Experiment

```powershell
# Run everything: data generation → training → evaluation → plots
python main.py
```

This will:
1. Generate synthetic dataset (429 records)
2. Train Isolation Forest on normal samples
3. Train LSTM Autoencoder (40 epochs, early stopping)
4. Evaluate ensemble + compare all models
5. Save 5 plots to `results/`
6. Save `results/predictions.csv` and `results/model_results.json`

**Faster run (skip LSTM training for quick test):**
```powershell
python main.py --epochs 10 --no-plots
```

**Skip training if models already saved:**
```powershell
python main.py --skip-train
```

### Hour 4–5 — View Results

All plots saved to `results/`:

| File | Description |
|---|---|
| `fig2_swarm_chart.png` | Paper Fig. 2 — build time distribution with anomalies |
| `fig3_histogram.png` | Paper Fig. 3 — test time vs deployment success rate |
| `anomaly_scores_timeline.png` | Ensemble anomaly score over all 429 runs |
| `model_comparison.png` | Paper Table 2 — model comparison bar chart |
| `table1_build_performance.png` | Paper Table 1 — build stats by instance range |

### Hour 5–6 — Grafana Dashboard (optional, requires Docker)

```powershell
# Start Prometheus + Grafana
cd dashboard
docker-compose up -d

# Push metrics from experiment results
cd ..
python dashboard/push_metrics.py
```

Open **http://localhost:3000** (admin / admin) → Dashboards → "CI/CD Pipeline Anomaly Detection"

### Hour 6–7 — Azure Cloud Integration (optional, low cost)

#### Option A: Azure Blob Storage (< $0.01/month for this experiment)

```powershell
# 1. Login to Azure
az login

# 2. Create storage account
az group create -n anomaly-rg -l eastus
az storage account create -n anomalypipeline$RANDOM -g anomaly-rg --sku Standard_LRS

# 3. Set env var
Copy-Item .env.example .env
# Edit .env: set AZURE_STORAGE_ACCOUNT=<name-from-above>

# 4. Assign yourself storage access
az role assignment create --role "Storage Blob Data Contributor" \
  --assignee $(az ad signed-in-user show --query id -o tsv) \
  --scope $(az storage account show -n <account-name> -g anomaly-rg --query id -o tsv)

# 5. Upload results
python azure/azure_storage.py
```

#### Option B: REST API for live scoring

```powershell
# Run main.py first to train models, then:
uvicorn azure.api:app --reload --port 8000

# Test it
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"build_duration": 750, "test_execution_time": 600, "deployment_frequency": 0.8, "resource_utilization": 88, "error_count": 3}'
```

---

## Run Tests

```powershell
pytest tests/ -v
```

---

## Azure Cost Breakdown

| Service | Usage | Estimated Cost |
|---|---|---|
| Azure Blob Storage (LRS) | ~10 MB results | < $0.01/month |
| Azure ML Workspace | Logging only (no compute) | **Free** |
| Azure Container Apps | REST API (free tier: 180k vCPU-s/mo) | **Free** |
| Grafana + Prometheus | Local Docker | **Free** |
| ML Training | Local CPU/GPU | **Free** |
| **Total** | | **< $0.01/month** |

---

## Key Paper Formulas Implemented

| Paper Eq. | Formula | Implementation |
|---|---|---|
| (1) Binary cross-entropy | $L=-\frac{1}{N}\sum[y_i \log\hat{y}_i + (1-y_i)\log(1-\hat{y}_i)]$ | LSTM training loss |
| (2) Isolation Forest score | $s(x,n)=2^{-E[h(x)]/c(n)}$ | `isolation_forest.score_samples()` |
| (3) LSTM forget gate | $f_t=\sigma(W_f\cdot[h_{t-1},x_t]+b_f)$ | TensorFlow LSTM cell (internal) |
| (4) Z-score normalisation | $z=(x-\mu)/\sigma$ | `preprocessor.zscore()` |
| (5) F1 Score | $F_1=2\cdot\frac{P\cdot R}{P+R}$ | `ensemble.evaluate()` |
| (6) MAE for time series | $MAE=\frac{1}{n}\sum\|y_i-\hat{y}_i\|$ | LSTM reconstruction error |

---

## Tech Stack

- **Python 3.11+** — core language
- **scikit-learn** — Isolation Forest
- **TensorFlow 2.x** — LSTM Autoencoder
- **pandas / numpy** — data manipulation
- **matplotlib / seaborn** — static figures
- **FastAPI + uvicorn** — REST API
- **Prometheus + Grafana** — monitoring dashboard (Docker)
- **Azure Blob Storage** — result persistence (optional)
- **Azure ML** — experiment tracking (optional)
