"""Feature-importance and feature-distribution plots for the report (Figures 5a, 5b).

Usage (from repo root, after features/build_dataset.py):
    python eda/feature_plots.py

5a: split gain per feature, read from the trees stored in artifacts/model.joblib
    (parsed from the LightGBM model text, so lightgbm need not be importable).
5b: distributions of four key features for true vs wrong candidates (training set).
"""
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = os.path.join("results", "plots")
BLUE, ORANGE, INK, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#fcfcfb"


def importance():
    text = open(os.path.join("artifacts", "model.joblib"), "rb").read().decode("latin1")
    cols = json.load(open(os.path.join("artifacts", "feature_cols.json")))
    gain = np.zeros(len(cols))
    splits = np.zeros(len(cols))
    for sf, sg in re.findall(r"split_feature=([^\n]*)\n.*?split_gain=([^\n]*)\n", text, re.S):
        f = np.array(sf.split(), int)
        np.add.at(gain, f, np.array(sg.split(), float))
        np.add.at(splits, f, 1)
    return np.array(cols), gain, splits


def main():
    os.makedirs(OUT, exist_ok=True)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    f, g, s = importance()
    share = 100 * g / g.sum()
    pd.DataFrame({"feature": f, "gain": g, "gain_share_pct": share, "splits": s}) \
        .sort_values("gain", ascending=False).to_csv(os.path.join(OUT, "feature_importance.csv"), index=False)
    o = np.argsort(share)
    fig, ax = plt.subplots(figsize=(8, 6.4), facecolor=SURF)
    ax.set_facecolor(SURF)
    ax.barh(range(len(o)), share[o], color=BLUE, height=0.7)
    ax.set_yticks(range(len(o)), f[o], fontsize=9)
    for i, k in enumerate(o):
        ax.text(share[k] + 0.4, i, f"{share[k]:.1f}%", va="center", fontsize=8, color="#52514e")
    ax.set_xlabel("Share of total split gain in the final tuned LightGBM (%)")
    ax.set_title("Which features the final model relies on", loc="left", color=INK, fontsize=12)
    ax.xaxis.grid(True, color="#e6e5e0", lw=0.8)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "feature_importance.png"), dpi=150, facecolor=SURF)

    df = pd.read_pickle(os.path.join("artifacts", "features", "train_cands.pkl"))
    df = df[df.point % 3 == 0]
    spec = [("dist_minus_min", "Distance minus nearest candidate (m)", (0, 40)),
            ("head_diff", "Heading difference (degrees)", (0, 180)),
            ("dist_ctx_mean", "Mean distance of neighbouring points to arc (m)", (0, 60)),
            ("end_proximity", "Distance from projection to nearest arc end (m)", (0, 40))]
    fig, axs = plt.subplots(2, 2, figsize=(10, 6.6), facecolor=SURF)
    for a, (c, lab, (lo, hi)) in zip(axs.ravel(), spec):
        a.set_facecolor(SURF)
        bins = np.linspace(lo, hi, 41)
        for y, col, name in [(0, ORANGE, "Wrong candidate"), (1, BLUE, "True road segment")]:
            a.hist(np.clip(df.loc[df.y == y, c].values, lo, hi), bins=bins, density=True,
                   histtype="step", lw=2, color=col, label=name)
        a.set_xlabel(lab)
        a.set_yticks([])
        a.set_xlim(lo, hi)
    axs[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("Key features separate true from wrong candidates (training set, values clipped at axis limit)",
                 x=0.02, ha="left", color=INK, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "feature_distributions.png"), dpi=150, facecolor=SURF)
    print("saved feature_importance.png, feature_distributions.png, feature_importance.csv")


if __name__ == "__main__":
    main()
