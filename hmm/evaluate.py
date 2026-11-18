"""Evaluate / tune the HMM on real trips: classifier-only vs classifier+HMM.

Usage examples
--------------
# 1) Data-driven starting values for sigma and beta (training labels only)
python -m hmm.evaluate --part train --estimate

# 2) Distance-only HMM on student_test (works before Rishab Naveen's model exists)
python -m hmm.evaluate --part student_test

# 3) With Rishab Naveen's scores (Saanvi Samayam's candidate table + a `prob` column)
python -m hmm.evaluate --part student_test --scored artifacts/scored/student_test_scored.pkl

# 4) Grid-search HMM params on training out-of-fold scores, save best config
python -m hmm.evaluate --part train --scored artifacts/scored/train_oof_scored.pkl --tune

Leakage rules: only `train` and `student_test` are ever loaded here.
Reference labels (from data_processing.labels.align_route) are used ONLY to
score predictions, never as decoder input.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_processing.io_utils import list_trips, load_trip  # noqa: E402
from data_processing.labels import align_route  # noqa: E402
from features.candidates import build_frame  # noqa: E402
from hmm.graph import RoadGraph  # noqa: E402
from hmm.metrics import route_continuity, sequence_metrics, top1_accuracy  # noqa: E402
from hmm.viterbi import DEFAULT_PARAMS, argmax_only, build_lattice, decode_lattice  # noqa: E402

ALLOWED_PARTS = {"train", "student_test"}
EXCLUDED_TRAIN = {"00000033", "00000054"}   # Saanvi Samayam: reference route misaligned


def load_scored(path):
    if path is None:
        return None
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_pickle(path)
    df["trip_id"] = df["trip_id"].astype(str)
    return df


def prepare(data_dir, part, scored=None, max_trips=None, verbose=True):
    """Load trips, build candidates (or take B's scored rows) and lattices."""
    assert part in ALLOWED_PARTS, "evaluator-testing data must never be used here"
    part_dir = os.path.join(data_dir, part)
    if not os.path.isdir(part_dir):
        sys.exit(f"Dataset not found: {part_dir}\n"
                 f"Copy the provided Student folder into data/ so you have data/{part}/<trip_id>/<trip_id>.track")
    items = []
    for d in list_trips(part_dir)[:max_trips]:
        t0 = time.time()
        trip = load_trip(d)
        if part == "train" and trip.trip_id in EXCLUDED_TRAIN:
            continue
        lab, _, dlab = align_route(trip.gps_xy, trip.node_xy, trip.arcs, trip.route)
        if scored is not None:
            cand = scored[scored["trip_id"] == trip.trip_id].reset_index(drop=True)
            if cand.empty:
                print("  no scored rows for", trip.trip_id, "- skipped")
                continue
            prob = cand["prob"].to_numpy()
        else:
            cand = build_frame(trip)
            prob = None
        graph = RoadGraph(trip)
        lat = build_lattice(cand, trip, graph, gap_s=DEFAULT_PARAMS["gap_s"])
        items.append(dict(trip=trip, lab=lab, dlab=dlab, graph=graph, lat=lat, prob=prob, cand=cand))
        if verbose:
            print(f"  {part} {trip.trip_id}: {len(lab)} pts, {len(cand)} cands, "
                  f"{time.time() - t0:.1f}s", flush=True)
    return items


def evaluate(items, params, cont_gap_m=30.0):
    rows = []
    for it in items:
        t0 = time.time()
        hmm, info = decode_lattice(it["lat"], it["prob"], params)
        dec_s = time.time() - t0
        base = argmax_only(it["lat"], it["prob"])
        r = {"trip_id": it["trip"].trip_id, "n_points": len(it["lab"]), "decode_s": dec_s, **info}
        for name, pred in [("clf", base), ("hmm", hmm)]:
            r[f"{name}_top1"] = top1_accuracy(pred, it["lab"])
            r[f"{name}_continuity"] = route_continuity(pred, it["graph"], cont_gap_m)
            for k, v in sequence_metrics(pred, it["trip"].route, it["graph"].arc_len).items():
                r[f"{name}_{k}"] = v
        rows.append(r)
    df = pd.DataFrame(rows)
    w = df["n_points"]
    summary = {c: float(np.average(df[c], weights=w)) for c in df.columns
               if c.startswith(("clf_", "hmm_"))}
    summary["n_trips"] = len(df)
    summary["n_breaks"] = int(df["n_breaks"].sum())
    return df, summary


def estimate_params(items):
    """Newson & Krumm style estimates from TRAINING labels:
       sigma = 1.4826 * median(GPS-to-label-arc distance)   (robust std)
       beta  = median(|d_gc - d_route(label_t, label_t+1)|) / ln 2
    """
    dl, diffs = [], []
    for it in items:
        dl.append(it["dlab"])
        trip, lab, g = it["trip"], it["lab"], it["graph"]
        gps, t = trip.gps_xy, trip.track[:, 2]
        nxy = trip.node_xy
        from data_processing.io_utils import point_segment
        _, frac = point_segment(gps, nxy[trip.arcs[lab, 0]], nxy[trip.arcs[lab, 1]])
        for k in range(1, len(lab)):
            if t[k] - t[k - 1] > 2:        # only regular 1 Hz steps
                continue
            dr = g.route_distance(lab[k - 1], frac[k - 1], lab[k], frac[k])
            if np.isfinite(dr):
                diffs.append(abs(np.hypot(*(gps[k] - gps[k - 1])) - dr))
    sigma = 1.4826 * float(np.median(np.concatenate(dl)))
    beta = float(np.median(diffs) / np.log(2))
    return {"sigma": round(sigma, 2), "beta": round(beta, 2), "n_transitions": len(diffs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--part", default="student_test", choices=sorted(ALLOWED_PARTS))
    ap.add_argument("--scored", default=None, help="B's scored candidate table (.pkl/.parquet)")
    ap.add_argument("--params", default="artifacts/hmm_config.json")
    ap.add_argument("--out", default="results/hmm")
    ap.add_argument("--max-trips", type=int, default=None)
    ap.add_argument("--estimate", action="store_true", help="estimate sigma/beta from labels")
    ap.add_argument("--tune", action="store_true", help="grid-search params, save best")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    params = dict(DEFAULT_PARAMS)
    if os.path.exists(args.params):
        params.update(json.load(open(args.params)))
    print("params:", params)
    scored = load_scored(args.scored)
    tag = f"{args.part}_{'clf' if scored is not None else 'dist'}"
    items = prepare(args.data, args.part, scored, args.max_trips)

    if args.estimate:
        est = estimate_params(items)
        print("estimated:", est)
        json.dump(est, open(os.path.join(args.out, f"estimate_{args.part}.json"), "w"), indent=2)
        return

    if args.tune:
        grid = {
            "beta": [2, 3, 5, 8, 12, 20],
            "sigma": [4, 6, 8],
            "alpha": [1.0, 0.7, 0.4] if scored is not None else [0.0],
        }
        trials = []
        for vals in itertools.product(*grid.values()):
            p = {**params, **dict(zip(grid, vals))}
            t0 = time.time()
            _, s = evaluate(items, p)
            trials.append({**dict(zip(grid, vals)), **s, "seconds": time.time() - t0})
            print(trials[-1], flush=True)
        tr = pd.DataFrame(trials).sort_values("hmm_top1", ascending=False)
        tr.to_csv(os.path.join(args.out, f"trials_{tag}.csv"), index=False)
        best = tr.iloc[0]
        params.update({k: float(best[k]) for k in grid})
        os.makedirs(os.path.dirname(args.params) or ".", exist_ok=True)
        json.dump(params, open(args.params, "w"), indent=2)
        print("best params saved to", args.params, params)

    per_trip, summary = evaluate(items, params)
    per_trip.to_csv(os.path.join(args.out, f"per_trip_{tag}.csv"), index=False)
    json.dump({"params": params, **summary},
              open(os.path.join(args.out, f"summary_{tag}.json"), "w"), indent=2)
    print(per_trip[["trip_id", "n_points", "clf_top1", "hmm_top1",
                    "clf_continuity", "hmm_continuity", "n_breaks"]].to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
