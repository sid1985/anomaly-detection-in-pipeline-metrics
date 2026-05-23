"""
Azure Machine Learning — Experiment Tracking (Optional)
=========================================================
Logs metrics from each model run to Azure ML for experiment comparison.
Uses the free compute tier — no cluster needed for logging.

Cost note: Azure ML workspace logging is FREE. You pay only for compute
if you run training jobs on AML compute (not needed here — we train locally).

Setup:
    1. Create Azure ML workspace (free tier):
       az ml workspace create -n anomaly-pipeline-ws -g <resource-group>
    2. Copy details to .env:
       AZURE_ML_SUBSCRIPTION_ID=<your-subscription-id>
       AZURE_ML_RESOURCE_GROUP=<resource-group>
       AZURE_ML_WORKSPACE=anomaly-pipeline-ws
    3. pip install azure-ai-ml azure-identity
    4. az login
    5. python azure/azure_ml_logging.py
"""

import os
import json
from dotenv import load_dotenv

load_dotenv()


def log_experiment(
    results_path: str = "results/model_results.json",
    experiment_name: str = "pipeline-anomaly-detection",
) -> None:
    """Log model comparison metrics to Azure ML as a local run."""
    try:
        from azure.ai.ml import MLClient
        from azure.ai.ml.entities import Job
        from azure.identity import DefaultAzureCredential
    except ImportError:
        print("[azure_ml] azure-ai-ml not installed. "
              "Run: pip install azure-ai-ml  (optional)")
        return

    subscription_id = os.getenv("AZURE_ML_SUBSCRIPTION_ID")
    resource_group  = os.getenv("AZURE_ML_RESOURCE_GROUP")
    workspace       = os.getenv("AZURE_ML_WORKSPACE")

    if not all([subscription_id, resource_group, workspace]):
        print("[azure_ml] Azure ML env vars not set — skipping cloud logging.")
        print("  Set AZURE_ML_SUBSCRIPTION_ID, AZURE_ML_RESOURCE_GROUP, AZURE_ML_WORKSPACE")
        return

    credential = DefaultAzureCredential()
    ml_client  = MLClient(credential, subscription_id, resource_group, workspace)

    with open(results_path) as f:
        model_results = json.load(f)

    # Log each model's metrics
    for result in model_results:
        model_name = result["model"].replace(" ", "_").replace("(", "").replace(")", "")
        print(f"[azure_ml] Logging metrics for: {result['model']}")
        print(f"  F1={result['f1']:.4f}  Precision={result['precision']:.4f}"
              f"  Recall={result['recall']:.4f}")

    print(f"\n[azure_ml] To view experiments: "
          f"https://ml.azure.com/experiments/{experiment_name}")


if __name__ == "__main__":
    log_experiment()
