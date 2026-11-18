"""Exploratory data analysis for the train and student_test partitions.

Usage:  python eda/eda.py --data data --out results/eda
Produces inventory CSV, summary stats, data-quality log and plots.
Evaluator-testing data is never read here.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_processing.io_utils import list_trips, load_trip, point_segment  # noqa: E402


def trip_stats(trip_dir, partition):
    tid = os.path.basename(trip_dir)
    raw = np.loadtxt(os.path.join(trip_dir, f"{tid}.track"), ndmin=2)
    tr = load_trip(trip_dir)
    t = tr.track[:, 2]
    dt = np.diff(t)
    V, E = len(tr.nodes), len(tr.arcs)
    g = coo_matrix((np.ones(E), (tr.arcs[:, 0], tr.arcs[:, 1])), shape=(V, V))
    n_weak, lab_w = connected_components(g, directed=True, connection="weak")
    n_strong, lab_s = connected_components(g, directed=True, connection="strong")
    deg = np.bincount(tr.arcs.ravel(), minlength=V)
    arcset = set(map(tuple, tr.arcs.tolist()))
    two_way = np.mean([(b, a) in arcset for a, b in tr.arcs.tolist()])
    xy = tr.gps_xy
    step = np.sqrt((np.diff(xy, axis=0) ** 2).sum(1))
    speed = step / np.maximum(dt, 1e-9)
    nxy = tr.node_xy
    arc_len = np.sqrt(((nxy[tr.arcs[:, 1]] - nxy[tr.arcs[:, 0]]) ** 2).sum(1))
    r = tr.route
    route_cont = float(np.mean(tr.arcs[r[:-1], 1] == tr.arcs[r[1:], 0])) if r is not None and len(r) > 1 else np.nan
    # distance from every GPS point to nearest route arc = GPS-noise proxy
    if r is not None:
        a, b = nxy[tr.arcs[r, 0]], nxy[tr.arcs[r, 1]]
        dmin = np.full(len(xy), np.inf)
        for k in range(0, len(r), 200):
            d, _ = point_segment(xy[:, None, :], a[None, k:k + 200], b[None, k:k + 200])
            dmin = np.minimum(dmin, d.min(1))
    else:
        dmin = np.array([np.nan])
    return dict(
        partition=partition, trip_id=tid,
        n_gps_raw=len(raw), n_gps_clean=len(tr.track),
        n_dropped=len(raw) - len(tr.track),
        n_nan=int(np.isnan(raw).sum()),
        duration_s=float(t[-1] - t[0]),
        dt_median=float(np.median(dt)), dt_max=float(dt.max()),
        pct_dt_1s=float(np.mean(dt == 1) * 100),
        n_gaps_gt10s=int((dt > 10).sum()), n_gaps_gt60s=int((dt > 60).sum()),
        path_len_km=float(step.sum() / 1000),
        speed_p95_ms=float(np.percentile(speed, 95)),
        n_jumps_gt50ms=int((speed > 50).sum()),
        lon_min=tr.track[:, 0].min(), lon_max=tr.track[:, 0].max(),
        lat_min=tr.track[:, 1].min(), lat_max=tr.track[:, 1].max(),
        n_nodes=V, n_arcs=E, isolated_nodes=int((deg == 0).sum()),
        weak_components=n_weak, strong_components=n_strong,
        largest_scc_frac=float(np.bincount(lab_s).max() / V),
        two_way_frac=float(two_way), arc_len_median_m=float(np.median(arc_len)),
        route_len=0 if r is None else len(r),
        route_continuity=route_cont,
        gps_to_route_median_m=float(np.nanmedian(dmin)),
        gps_to_route_p95_m=float(np.nanpercentile(dmin, 95)),
        gps_to_route_max_m=float(np.nanmax(dmin)),
        file_kb=sum(os.path.getsize(os.path.join(trip_dir, f)) for f in os.listdir(trip_dir)) / 1024,
    ), dt, dmin, tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/eda")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows, all_dt, all_d = [], {}, {}
    examples = {}
    for part in ["train", "student_test"]:
        all_dt[part], all_d[part] = [], []
        for d in list_trips(os.path.join(args.data, part)):
            s, dt, dmin, tr = trip_stats(d, part)
            rows.append(s)
            all_dt[part].append(dt)
            all_d[part].append(dmin)
            if len(examples) < 4 and part == "train":
                examples[s["trip_id"]] = tr
            print(part, s["trip_id"], s["n_gps_clean"], s["n_arcs"], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "inventory.csv"), index=False)

    summ = df.groupby("partition")[[
        "n_gps_clean", "duration_s", "path_len_km", "dt_max", "n_gaps_gt10s",
        "n_nodes", "n_arcs", "isolated_nodes", "weak_components", "route_len",
        "gps_to_route_median_m", "gps_to_route_p95_m"]].describe().T
    summ.to_csv(os.path.join(args.out, "summary_stats.csv"))

    # ---------------- data-quality log ----------------
    with open(os.path.join(args.out, "data_quality.md"), "w") as f:
        f.write("# Data-quality findings (train + student_test)\n\n")
        for part in ["train", "student_test"]:
            p = df[df.partition == part]
            dt = np.concatenate(all_dt[part])
            dd = np.concatenate(all_d[part])
            f.write(f"## {part} ({len(p)} trips)\n")
            f.write(f"- GPS points: total {p.n_gps_clean.sum()}, per trip min {p.n_gps_clean.min()}, median {int(p.n_gps_clean.median())}, max {p.n_gps_clean.max()}\n")
            f.write(f"- NaN values in raw tracks: {p.n_nan.sum()}; rows dropped (dup/invalid ts): {p.n_dropped.sum()}\n")
            f.write(f"- Sampling: {np.mean(dt == 1)*100:.2f}% of steps exactly 1 s; steps > 10 s: {(dt > 10).sum()}; > 60 s: {(dt > 60).sum()}; max gap {dt.max():.0f} s\n")
            f.write(f"- Trips with a gap > 60 s: {list(p[p.n_gaps_gt60s > 0].trip_id)}\n")
            f.write(f"- Road network: nodes {p.n_nodes.min()}-{p.n_nodes.max()}, arcs {p.n_arcs.min()}-{p.n_arcs.max()}, isolated nodes total {p.isolated_nodes.sum()}, weak components median {p.weak_components.median()}\n")
            f.write(f"- Arcs that have a reverse arc: {p.two_way_frac.mean()*100:.1f}% (directed network, mostly two-way)\n")
            f.write(f"- Route continuity (arc[i].to == arc[i+1].from): mean {p.route_continuity.mean():.4f}; trips < 1: {list(p[p.route_continuity < 1].trip_id)}\n")
            f.write(f"- GPS distance to nearest reference-route arc: median {np.median(dd):.1f} m, p95 {np.percentile(dd,95):.1f} m, p99 {np.percentile(dd,99):.1f} m, max {dd.max():.0f} m\n\n")

    # ---------------- plots ----------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for part, c in [("train", "tab:blue"), ("student_test", "tab:orange")]:
        p = df[df.partition == part]
        ax[0].hist(p.n_gps_clean, bins=30, alpha=.6, label=part, color=c)
        dt = np.concatenate(all_dt[part])
        ax[1].hist(np.clip(dt, 0, 20), bins=40, alpha=.6, label=part, color=c, log=True)
        dd = np.concatenate(all_d[part])
        ax[2].hist(np.clip(dd, 0, 60), bins=60, alpha=.6, label=part, color=c, density=True)
    ax[0].set(title="GPS points per trip", xlabel="points")
    ax[1].set(title="Sampling interval (clipped at 20 s)", xlabel="seconds")
    ax[2].set(title="GPS distance to reference route (clipped 60 m)", xlabel="metres")
    for a in ax:
        a.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "distributions.png"), dpi=120)
    plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].scatter(df.n_arcs, df.n_gps_clean, c=(df.partition == "train").map({True: "tab:blue", False: "tab:orange"}))
    ax[0].set(xscale="log", xlabel="arcs in road network", ylabel="GPS points", title="Network size vs trajectory length")
    ax[1].bar(range(len(df)), df.weak_components, color=(df.partition == "train").map({True: "tab:blue", False: "tab:orange"}))
    ax[1].set(yscale="log", title="Weakly connected components per trip", xlabel="trip index")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "network_stats.png"), dpi=120)
    plt.close()

    fig, axs = plt.subplots(1, len(examples), figsize=(5 * len(examples), 5))
    for a, (tid, tr) in zip(np.atleast_1d(axs), examples.items()):
        n = tr.nodes
        seg = np.stack([n[tr.arcs[:, 0]], n[tr.arcs[:, 1]]], 1)
        from matplotlib.collections import LineCollection
        a.add_collection(LineCollection(seg, colors="lightgrey", linewidths=.4))
        rs = np.stack([n[tr.arcs[tr.route, 0]], n[tr.arcs[tr.route, 1]]], 1)
        a.add_collection(LineCollection(rs, colors="green", linewidths=2, label="reference route"))
        a.scatter(tr.track[:, 0], tr.track[:, 1], s=2, c="tab:blue", label="GPS")
        pad = .005
        a.set_xlim(tr.track[:, 0].min() - pad, tr.track[:, 0].max() + pad)
        a.set_ylim(tr.track[:, 1].min() - pad, tr.track[:, 1].max() + pad)
        a.set_title(f"train trip {tid}")
        a.set_aspect(1 / np.cos(np.radians(tr.origin[1])))
        a.legend(loc="lower right", fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "example_trajectories.png"), dpi=120)
    plt.close()
    print(open(os.path.join(args.out, "data_quality.md")).read())


if __name__ == "__main__":
    main()
