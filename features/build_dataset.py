"""Build candidate feature tables for train and student_test.

Usage: python features/build_dataset.py --data data --out artifacts/features

Outputs per partition:
  <part>_cands.pkl   one row per (point, candidate) with features + y
  <part>_points.pkl  one row per GPS point: label arc, label distance,
                     whether label is in the candidate set (candidate miss)
Train trips whose reference route is grossly misaligned with the GPS
(median label distance > 30 m) are excluded from training and logged.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_processing.io_utils import list_trips, load_trip  # noqa: E402
from data_processing.labels import align_route  # noqa: E402
from features.candidates import DEFAULT_K, DEFAULT_RADIUS, build_frame  # noqa: E402

BAD_LABEL_MEDIAN_M = 30.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="artifacts/features")
    ap.add_argument("--radius", type=float, default=DEFAULT_RADIUS)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    log = {"radius": args.radius, "k": args.k, "excluded_trips": {}}

    for part in ["train", "student_test"]:
        frames, points = [], []
        for d in list_trips(os.path.join(args.data, part)):
            t0 = time.time()
            tr = load_trip(d)
            lab, j, dlab = align_route(tr.gps_xy, tr.node_xy, tr.arcs, tr.route)
            med = float(np.median(dlab))
            if part == "train" and med > BAD_LABEL_MEDIAN_M:
                log["excluded_trips"][tr.trip_id] = f"median GPS-to-label distance {med:.1f} m"
                print("EXCLUDE", tr.trip_id, med)
                continue
            df = build_frame(tr, lab, args.radius, args.k)
            hit = df.groupby("point")["y"].max().reindex(range(len(tr.track)), fill_value=0)
            ncand = df.groupby("point").size().reindex(range(len(tr.track)), fill_value=0)
            points.append(pd.DataFrame({
                "trip_id": tr.trip_id, "point": np.arange(len(tr.track)),
                "t": tr.track[:, 2], "label_arc": lab, "label_dist": dlab,
                "label_in_cands": hit.values, "n_cand": ncand.values,
            }))
            frames.append(df)
            print(part, tr.trip_id, len(tr.track), len(df),
                  f"miss={1 - hit.mean():.3f}", f"{time.time() - t0:.1f}s", flush=True)
        cands = pd.concat(frames, ignore_index=True)
        pts = pd.concat(points, ignore_index=True)
        cands.to_pickle(os.path.join(args.out, f"{part}_cands.pkl"))
        pts.to_pickle(os.path.join(args.out, f"{part}_points.pkl"))
        log[part] = {
            "trips": int(pts.trip_id.nunique()), "points": int(len(pts)),
            "candidate_rows": int(len(cands)),
            "candidate_miss_rate": float(1 - pts.label_in_cands.mean()),
            "points_without_candidates": int((pts.n_cand == 0).sum()),
            "positive_rate": float(cands.y.mean()),
            "label_dist_median_m": float(pts.label_dist.median()),
        }
    with open(os.path.join(args.out, "dataset_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
