"""
Visualizer
===========
Reproduces the paper's three core figures plus additional diagnostics.

Fig 2 – Swarm/strip chart: build duration distribution with anomaly overlay
Fig 3 – Histogram + deployment success rate line
Extra – Anomaly score heatmap over time
Extra – Model comparison bar chart (Table 2)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (works in all environments)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Paper-aligned colour palette
NORMAL_COLOR  = "#4C72B0"
ANOMALY_COLOR = "#DD4444"
ACCENT_COLOR  = "#2CA02C"


# ─────────────────────────────────────────────────────────────────────────────
def plot_fig2_swarm(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """
    Fig. 2 – Distribution of pipeline deviations and build times.
    Strip chart with jitter; anomalies highlighted in red.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    normal   = df[df["is_anomaly"] == 0]
    anomalous = df[df["is_anomaly"] == 1]

    jitter   = np.random.default_rng(42)
    x_normal  = jitter.uniform(-0.3, 0.3, len(normal))
    x_anomaly = jitter.uniform(-0.3, 0.3, len(anomalous))

    ax.scatter(x_normal,  normal["build_duration"],
               c=NORMAL_COLOR, alpha=0.5, s=18, label="Normal")
    ax.scatter(x_anomaly, anomalous["build_duration"],
               c=ANOMALY_COLOR, alpha=0.85, s=35, label="Anomaly",
               edgecolors="darkred", linewidths=0.5)

    ax.axhline(600, color=ANOMALY_COLOR, ls="--", lw=1.2,
               label="Anomaly boundary (~600 s)")
    ax.set_xticks([])
    ax.set_xlabel("Execution Instances (jittered)", fontsize=11)
    ax.set_ylabel("Build Duration (seconds)", fontsize=11)
    ax.set_title(
        "Fig. 2 — Distribution of Pipeline Deviations and Build Times\n"
        "(Paper replication – 429 instances)", fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=10)
    fig.tight_layout()

    if save:
        path = os.path.join(RESULTS_DIR, "fig2_swarm_chart.png")
        fig.savefig(path, dpi=150)
        print(f"[Visualizer] Saved {path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
def plot_fig3_histogram(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """
    Fig. 3 – Histogram of test execution time + deployment success rate line.
    """
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Histogram of test execution time
    bins = np.linspace(df["test_execution_time"].min(),
                       df["test_execution_time"].max(), 25)
    ax1.hist(df["test_execution_time"], bins=bins,
             color=NORMAL_COLOR, alpha=0.7, label="Test Execution Time (s)")
    ax1.set_xlabel("Test Execution Time (seconds)", fontsize=11)
    ax1.set_ylabel("Build Count", fontsize=11, color=NORMAL_COLOR)
    ax1.tick_params(axis="y", labelcolor=NORMAL_COLOR)

    # Deployment success rate overlay
    df_sorted = df.sort_values("test_execution_time")
    bucket_labels = pd.cut(df_sorted["test_execution_time"], bins=bins)
    bucket_df     = df_sorted.groupby(bucket_labels, observed=True).agg(
        success_rate=("is_anomaly", lambda x: 1 - x.mean()),
        count=("is_anomaly", "count"),
    ).reset_index()
    bucket_df["mid"] = bucket_df["test_execution_time"].apply(
        lambda iv: (iv.left + iv.right) / 2
    )

    ax2 = ax1.twinx()
    ax2.plot(bucket_df["mid"], bucket_df["success_rate"],
             color=ACCENT_COLOR, lw=2.2, marker="o", markersize=4,
             label="Deployment Success Rate")
    ax2.set_ylabel("Deployment Success Rate", fontsize=11, color=ACCENT_COLOR)
    ax2.tick_params(axis="y", labelcolor=ACCENT_COLOR)
    ax2.set_ylim(0, 1.05)

    ax1.set_title(
        "Fig. 3 — Test Execution Frequency vs Deployment Success Rate\n"
        "(Paper replication)", fontsize=12, fontweight="bold"
    )
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="upper right")

    fig.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, "fig3_histogram.png")
        fig.savefig(path, dpi=150)
        print(f"[Visualizer] Saved {path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
def plot_anomaly_scores(
    df: pd.DataFrame,
    ensemble_scores: np.ndarray,
    threshold: float = 0.5,
    save: bool = True,
) -> plt.Figure:
    """Anomaly score timeline with threshold line and ground-truth markers."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    x = df["instance_id"].values

    # Panel 1 – Build Duration
    c = [ANOMALY_COLOR if v else NORMAL_COLOR for v in df["is_anomaly"]]
    axes[0].bar(x, df["build_duration"], color=c, alpha=0.7, width=1.0)
    axes[0].set_ylabel("Build Duration (s)")
    axes[0].set_title("Build Duration per Pipeline Run", fontweight="bold")
    normal_p  = mpatches.Patch(color=NORMAL_COLOR, label="Normal")
    anomaly_p = mpatches.Patch(color=ANOMALY_COLOR, label="Anomaly (ground truth)")
    axes[0].legend(handles=[normal_p, anomaly_p], fontsize=9)

    # Panel 2 – Ensemble Anomaly Score
    axes[1].plot(x, ensemble_scores, color="#8B4513", lw=1.2, label="Ensemble Score")
    axes[1].axhline(threshold, color=ANOMALY_COLOR, ls="--", lw=1.5,
                    label=f"Threshold ({threshold})")
    # Mark ground-truth anomalies
    gt_mask = df["is_anomaly"].values.astype(bool)
    axes[1].scatter(x[gt_mask], ensemble_scores[gt_mask],
                    c=ANOMALY_COLOR, zorder=5, s=40, label="GT Anomaly")
    axes[1].set_ylabel("Anomaly Score")
    axes[1].set_title("Ensemble Anomaly Score Timeline", fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].set_ylim(-0.05, 1.05)

    # Panel 3 – Resource Utilization
    axes[2].fill_between(x, df["resource_utilization"],
                         alpha=0.5, color="#9467BD", label="Resource Util %")
    axes[2].plot(x, df["resource_utilization"], color="#7B2F8E", lw=0.8)
    axes[2].set_ylabel("Resource Util (%)")
    axes[2].set_xlabel("Instance ID")
    axes[2].set_title("Resource Utilization", fontweight="bold")
    axes[2].legend(fontsize=9)

    fig.suptitle(
        "AI-Based Anomaly Detection — Pipeline Metrics Dashboard\n"
        "(429 CI/CD Execution Records)",
        fontsize=13, fontweight="bold", y=1.01
    )
    fig.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, "anomaly_scores_timeline.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Visualizer] Saved {path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
def plot_model_comparison(results: list[dict], save: bool = True) -> plt.Figure:
    """
    Grouped bar chart: actual run results vs paper Table 2 targets.
    """
    # Paper reference F1 targets (Table 2)
    paper_f1 = {
        "Threshold (Baseline)": 0.56,
        "IsolationForest":      0.85,
        "LSTM_Autoencoder":     0.90,
        "Ensemble(IF+LSTM)":    0.93,
    }
    # Display labels (human-readable)
    display_label = {
        "Threshold (Baseline)": "Threshold\n(Baseline)",
        "IsolationForest":      "Isolation\nForest",
        "LSTM_Autoencoder":     "LSTM\nAutoencoder",
        "Ensemble(IF+LSTM)":    "Ensemble\n(IF+LSTM)",
    }

    # Only plot models that have actual results
    ordered = [r for r in results if r["model"] in paper_f1]
    models        = [r["model"] for r in ordered]
    actual_f1     = [r["f1"] for r in ordered]
    actual_prec   = [r["precision"] for r in ordered]
    actual_rec    = [r["recall"] for r in ordered]
    paper_targets = [paper_f1[m] for m in models]
    labels        = [display_label[m] for m in models]

    x = np.arange(len(models))
    w = 0.20
    fig, ax = plt.subplots(figsize=(12, 6))

    b1 = ax.bar(x - w,     actual_prec,   w, label="Actual Precision", color="#4C72B0", alpha=0.9)
    b2 = ax.bar(x,         actual_rec,    w, label="Actual Recall",    color="#DD4444", alpha=0.9)
    b3 = ax.bar(x + w,     actual_f1,     w, label="Actual F1",        color="#2CA02C", alpha=0.9)
    b4 = ax.bar(x + 2 * w, paper_targets, w, label="Paper F1 Target",  color="#FF7F0E",
                alpha=0.6, hatch="///", edgecolor="darkorange")

    for bars in [b1, b2, b3, b4]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        "Model Comparison — Actual Results vs Paper Table 2\n"
        "(Synthetic data: local run)", fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.axhline(1.0, color="gray", ls=":", lw=0.8)
    fig.tight_layout()

    if save:
        path = os.path.join(RESULTS_DIR, "model_comparison.png")
        fig.savefig(path, dpi=150)
        print(f"[Visualizer] Saved {path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
def plot_build_performance_table(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """
    Reproduces paper Table 1 — Build performance by instance range.
    """
    ranges = [(1, 100), (101, 200), (201, 300), (301, 400), (401, 429)]
    rows   = []
    for (lo, hi) in ranges:
        sub = df[(df["instance_id"] >= lo) & (df["instance_id"] <= hi)]
        rows.append({
            "Range":          f"{lo}-{hi}",
            "Avg Duration":   round(sub["build_duration"].mean(), 0),
            "Max Duration":   round(sub["build_duration"].max(),  0),
            "Resource Load":  round(sub["resource_utilization"].mean(), 0),
            "Error Count":    int(sub["error_count"].sum()),
        })
    tbl = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.axis("off")
    t = ax.table(
        cellText   = tbl.values,
        colLabels  = tbl.columns,
        cellLoc    = "center",
        loc        = "center",
    )
    t.auto_set_font_size(False)
    t.set_fontsize(10)
    t.scale(1.2, 1.6)
    # Highlight 201-300 row (infrastructure instability)
    for col in range(len(tbl.columns)):
        t[(3, col)].set_facecolor("#FFCCCC")
    ax.set_title("Table 1 — Build Performance Range (Paper Replication)",
                 fontweight="bold", fontsize=11, pad=10)
    fig.tight_layout()

    if save:
        path = os.path.join(RESULTS_DIR, "table1_build_performance.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Visualizer] Saved {path}")
    return fig
