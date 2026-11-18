"""Report figures for the HMM section, generated from the CSV/JSON outputs of
hmm/evaluate.py (no numbers are typed in by hand).

    results/hmm/plots/top1_per_trip.png          classifier-only vs HMM, student_test
    results/hmm/plots/continuity_per_trip.png    same, route continuity
    results/hmm/plots/overall_metrics.png        all route metrics, student_test totals
    results/hmm/plots/beta_sensitivity.png       tuning trials on train (OOF scores)

Usage:
    python -m hmm.plots
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RES = os.path.join("results", "hmm")
OUT = os.path.join(RES, "plots")
CLF, HMM = "#eb6834", "#2a78d6"            # categorical slots 2 and 1
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # slots 1-3, fixed order
INK, INK2, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"


def _style(ax, ylabel):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def _pct(ax):
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))


def grouped(df, col, title, fname, ylabel):
    x = np.arange(len(df))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 3.8), dpi=150)
    ax.bar(x - w / 2 - 0.01, df[f"clf_{col}"], w, color=CLF, label="Classifier only")
    ax.bar(x + w / 2 + 0.01, df[f"hmm_{col}"], w, color=HMM, label="Classifier + HMM")
    ax.set_xticks(x, df["trip_id"].str[-3:], rotation=0)
    ax.set_xlabel("Student-test trip (last 3 digits of id)", color=INK2, fontsize=9)
    lo = max(0.0, min(df[f"clf_{col}"].min(), df[f"hmm_{col}"].min()) - 0.1)
    ax.set_ylim(lo, 1.0)
    _style(ax, ylabel)
    _pct(ax)
    ax.set_title(title, loc="left", color=INK, fontsize=11, pad=22)
    ax.legend(frameon=False, fontsize=9, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)


def overall(summary, fname):
    keys = [("top1", "Top-1 accuracy"), ("continuity", "Route continuity"),
            ("route_precision", "Route precision"), ("route_recall", "Route recall"),
            ("route_jaccard", "Route Jaccard")]
    x = np.arange(len(keys))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150)
    c = [summary[f"clf_{k}"] for k, _ in keys]
    h = [summary[f"hmm_{k}"] for k, _ in keys]
    ax.bar(x - w / 2 - 0.01, c, w, color=CLF, label="Classifier only")
    ax.bar(x + w / 2 + 0.01, h, w, color=HMM, label="Classifier + HMM")
    for xi, v in zip(x, h):   # label only the HMM bars (the headline)
        ax.text(xi + w / 2 + 0.01, v + 0.01, f"{v:.1%}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, [n for _, n in keys])
    ax.set_ylim(0, 1.08)
    _style(ax, "Point-weighted mean over trips")
    _pct(ax)
    ax.set_title("Student-test: classifier-only vs classifier + HMM", loc="left", color=INK, fontsize=11, pad=22)
    ax.legend(frameon=False, fontsize=9, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)


def sensitivity(trials, fname):
    best_sigma = trials.sort_values("hmm_top1", ascending=False).iloc[0]["sigma"]
    t = trials[trials["sigma"] == best_sigma]
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=150)
    for colour, (alpha, g) in zip(SERIES, sorted(t.groupby("alpha"), key=lambda z: -z[0])):
        g = g.sort_values("beta")
        ax.plot(g["beta"], g["hmm_top1"], color=colour, linewidth=2, marker="o",
                markersize=5, label=f"alpha = {alpha:g}")
    base = trials["clf_top1"].iloc[0]
    ax.axhline(base, color=INK2, linewidth=1, linestyle="--")
    ax.text(trials["beta"].max(), base + 0.002, "classifier only", ha="right",
            va="bottom", fontsize=8, color=INK2)
    ax.set_xlabel("beta (m), transition scale", color=INK2, fontsize=9)
    _style(ax, "Top-1 accuracy (train, out-of-fold)")
    _pct(ax)
    ax.set_title(f"HMM sensitivity to beta and alpha (sigma = {best_sigma:g} m)",
                 loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    pt = os.path.join(RES, "per_trip_student_test_clf.csv")
    sm = os.path.join(RES, "summary_student_test_clf.json")
    tr = os.path.join(RES, "trials_train_clf.csv")
    if os.path.exists(pt):
        df = pd.read_csv(pt, dtype={"trip_id": str})
        grouped(df, "top1", "Top-1 accuracy per student-test trip", "top1_per_trip.png", "Top-1 accuracy")
        grouped(df, "continuity", "Route continuity per student-test trip",
                "continuity_per_trip.png", "Route continuity")
        print("saved per-trip plots")
    if os.path.exists(sm):
        overall(json.load(open(sm)), "overall_metrics.png")
        print("saved overall_metrics.png")
    if os.path.exists(tr):
        sensitivity(pd.read_csv(tr), "beta_sensitivity.png")
        print("saved beta_sensitivity.png")
    print("plots in", OUT)


if __name__ == "__main__":
    main()
