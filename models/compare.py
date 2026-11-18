"""
Model Comparison and Visualization Generator (Owner: Person B)
Generates consolidated markdown table and publication-quality plots from real experiment logs.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from models.harness import (
    load_feature_cols,
    load_candidate_data,
    RESULTS_DIR
)
import models.m1_logistic as m1
import models.m2_knn as m2
import models.m3_random_forest as m3
import models.m4_boosting as m4
import models.m5_mlp as m5


def generate_comparison_plots(df: pd.DataFrame):
    plots_dir = RESULTS_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    models = df["model"].tolist()
    x = np.arange(len(models))

    # 1. CV Macro F1 with error bars (+/- std)
    plt.figure(figsize=(10, 5))
    bars = plt.bar(
        x,
        df["cv_macro_f1_mean"],
        yerr=df["cv_macro_f1_std"],
        capsize=5,
        color=["#4A90E2", "#50E3C2", "#F5A623", "#7ED321", "#9013FE", "#E91E63"][:len(models)],
        edgecolor="black",
        alpha=0.85
    )
    plt.xticks(x, models, fontsize=10, fontweight="bold", rotation=15)
    plt.ylabel("CV Macro F1", fontsize=12)
    plt.ylim([max(0.5, min(df["cv_macro_f1_mean"]) - 0.05), 1.005])
    plt.title("5-Fold GroupKFold CV Macro F1 on Real Data (with error bars)", fontsize=13, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.002, f"{yval:.4f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(plots_dir / "cv_macro_f1_comparison.png", dpi=300)
    plt.close()

    # 2. Student Test Top-1 Accuracy and Macro F1
    plt.figure(figsize=(11, 5))
    width = 0.35
    plt.bar(x - width/2, df["test_top1_acc"], width, label="Student-Test Top-1 Acc", color="#2C3E50", alpha=0.85)
    plt.bar(x + width/2, df["test_macro_f1"], width, label="Student-Test Macro F1", color="#16A085", alpha=0.85)
    plt.xticks(x, models, fontsize=10, fontweight="bold", rotation=15)
    plt.ylabel("Score", fontsize=12)
    plt.ylim([max(0.5, min(df["test_top1_acc"]) - 0.05), 1.005])
    plt.title("Student-Test Performance: Top-1 Accuracy vs Macro F1 (Real Trips)", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plots_dir / "test_top1_and_f1.png", dpi=300)
    plt.close()

    # 3. Log Loss and Inference Latency
    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax2 = ax1.twinx()
    ax1.plot(x, df["test_log_loss"], "o-", color="#D32F2F", lw=2.5, markersize=8, label="Test Log Loss (lower is better)")
    ax2.bar(x, df["test_infer_ms_per_trip"], width=0.4, alpha=0.3, color="#1976D2", label="Inference Latency (ms/trip)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=10, fontweight="bold", rotation=15)
    ax1.set_ylabel("Binary Log Loss", color="#D32F2F", fontsize=12)
    ax2.set_ylabel("Inference Time (ms / trip)", color="#1976D2", fontsize=12)
    plt.title("Model Probability Quality (Log Loss) vs Inference Latency", fontsize=13, fontweight="bold")
    fig.tight_layout()
    plt.savefig(plots_dir / "test_log_loss_and_latency.png", dpi=300)
    plt.close()

    print("Generated all comparison plots in results/plots/.")


def generate_markdown_table(df: pd.DataFrame):
    md_path = RESULTS_DIR / "model_comparison_table.md"
    table_cols = [
        "model", "family", "cv_macro_f1_mean", "cv_macro_f1_std",
        "cv_top1_acc_mean", "test_macro_f1", "test_top1_acc",
        "test_log_loss", "test_infer_ms_per_trip"
    ]
    renamed = {
        "model": "Model",
        "family": "Family",
        "cv_macro_f1_mean": "CV Macro F1 (Mean)",
        "cv_macro_f1_std": "CV F1 (Std)",
        "cv_top1_acc_mean": "CV Top-1 Acc",
        "test_macro_f1": "Student-Test F1",
        "test_top1_acc": "Student-Test Top-1",
        "test_log_loss": "Test Log Loss",
        "test_infer_ms_per_trip": "Latency (ms/trip)"
    }
    sub_df = df[table_cols].rename(columns=renamed)
    header = "| " + " | ".join(sub_df.columns) + " |\n"
    sep = "| " + " | ".join(["---"] * len(sub_df.columns)) + " |\n"
    body = ""
    for _, row in sub_df.iterrows():
        body += "| " + " | ".join(str(val) for val in row.values) + " |\n"
    md_content = "# PathSnap Model Comparison (Candidate Scoring Level)\n\n"
    md_content += header + sep + body
    md_content += "\n*Note: Route continuity will be populated when integrated with Person C's HMM decode module.*\n"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Saved comparison table to {md_path}")


def main():
    exp_file = RESULTS_DIR / "experiments.csv"
    if not exp_file.exists():
        print(f"No experiments found in {exp_file}")
        return
    df = pd.read_csv(exp_file)
    df = df.drop_duplicates(subset=["model"], keep="last")

    generate_comparison_plots(df)
    generate_markdown_table(df)


if __name__ == "__main__":
    main()
