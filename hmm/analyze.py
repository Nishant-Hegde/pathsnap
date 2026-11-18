"""Error analysis: where does the HMM help, where does it hurt, and why?

For every GPS point on student_test it compares the reference arc, the
classifier-only arc and the classifier+HMM arc, then puts each HMM error in
one bucket (checked in this order):
  label_not_in_candidates  reference arc was not generated -> nobody can be right
  unreliable_reference     GPS is > 30 m from its reference arc (label doubtful)
  near_time_gap            within 5 points of a gap > gap_s (trip split there)
  reverse_direction        predicted the same street in the opposite direction
  adjacent_arc             predicted an arc directly connected to the reference
                           (boundary timing: switched one arc early / late)
  other                    a different road (parallel street, wrong branch...)

Outputs (results/hmm/):
  analysis_points_student_test.csv   one row per GPS point
  analysis_errors_student_test.csv   per trip: fixed / broken / error buckets
and prints the trips to use for the "HMM fixed it" / "HMM struggled" maps.

Usage:
    python -m hmm.analyze --scored artifacts/scored/student_test_scored.pkl
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hmm.evaluate import load_scored, prepare  # noqa: E402
from hmm.viterbi import DEFAULT_PARAMS, argmax_only, decode_lattice  # noqa: E402

BUCKETS = ["label_not_in_candidates", "unreliable_reference", "near_time_gap",
           "reverse_direction", "adjacent_arc", "other"]


def classify_errors(trip, lab, pred, dlab, in_cands, near_gap):
    arcs = trip.arcs
    rev = (arcs[pred, 0] == arcs[lab, 1]) & (arcs[pred, 1] == arcs[lab, 0])
    adj = (arcs[pred, 1] == arcs[lab, 0]) | (arcs[pred, 0] == arcs[lab, 1])
    out = np.full(len(lab), "", dtype=object)
    wrong = pred != lab
    rules = [(~in_cands, BUCKETS[0]), (dlab > 30, BUCKETS[1]), (near_gap, BUCKETS[2]),
             (rev, BUCKETS[3]), (adj, BUCKETS[4]), (np.ones_like(wrong), BUCKETS[5])]
    for mask, name in rules:
        sel = wrong & (out == "") & mask
        out[sel] = name
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--scored", default="artifacts/scored/student_test_scored.pkl")
    ap.add_argument("--params", default="artifacts/hmm_config.json")
    ap.add_argument("--out", default="results/hmm")
    a = ap.parse_args()
    params = dict(DEFAULT_PARAMS)
    if os.path.exists(a.params):
        params.update(json.load(open(a.params)))
    print("params:", params)
    items = prepare(a.data, "student_test", load_scored(a.scored), verbose=False)

    pts, rows = [], []
    for it in items:
        trip, lab, dlab, lat, prob = it["trip"], it["lab"], it["dlab"], it["lat"], it["prob"]
        hmm, info = decode_lattice(lat, prob, params)
        clf = argmax_only(lat, prob)
        n = len(lab)
        cand = it["cand"]
        hit = cand.assign(lab=lab[cand["point"].to_numpy()])
        in_cands = np.zeros(n, bool)
        in_cands[hit.loc[hit["arc"] == hit["lab"], "point"].to_numpy()] = True
        t = trip.track[:, 2]
        gap_idx = np.flatnonzero(np.diff(t) > params["gap_s"]) + 1
        near_gap = np.zeros(n, bool)
        for g in gap_idx:
            near_gap[max(0, g - 5):g + 5] = True
        err = classify_errors(trip, lab, hmm, dlab, in_cands, near_gap)
        pts.append(pd.DataFrame({
            "trip_id": trip.trip_id, "point": np.arange(n), "t": t,
            "label_arc": lab, "clf_arc": clf, "hmm_arc": hmm, "label_dist_m": dlab,
            "label_in_cands": in_cands, "clf_ok": clf == lab, "hmm_ok": hmm == lab,
            "hmm_error_type": err}))
        fixed = int(((clf != lab) & (hmm == lab)).sum())
        broken = int(((clf == lab) & (hmm != lab)).sum())
        r = {"trip_id": trip.trip_id, "n_points": n,
             "clf_top1": float((clf == lab).mean()), "hmm_top1": float((hmm == lab).mean()),
             "gain_pts": float((hmm == lab).mean() - (clf == lab).mean()),
             "fixed_by_hmm": fixed, "broken_by_hmm": broken,
             "n_breaks": info["n_breaks"], "n_segments": info["n_segments"]}
        for b in BUCKETS:
            r[f"err_{b}"] = int((err == b).sum())
        rows.append(r)

    P = pd.concat(pts, ignore_index=True)
    R = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    P.to_csv(os.path.join(a.out, "analysis_points_student_test.csv"), index=False)
    R.to_csv(os.path.join(a.out, "analysis_errors_student_test.csv"), index=False)

    tot = R.sum(numeric_only=True)
    n_err = int((P["hmm_error_type"] != "").sum())
    print(R[["trip_id", "n_points", "clf_top1", "hmm_top1", "fixed_by_hmm", "broken_by_hmm"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nTotal points {int(tot['n_points'])}: fixed by HMM {int(tot['fixed_by_hmm'])}, "
          f"broken by HMM {int(tot['broken_by_hmm'])}, remaining HMM errors {n_err}")
    for b in BUCKETS:
        c = int(tot[f"err_{b}"])
        print(f"  {b:25s} {c:6d}  ({c / max(n_err, 1):.1%} of HMM errors)")
    best = R.sort_values("gain_pts", ascending=False).iloc[0]
    worst = R.sort_values("hmm_top1").iloc[0]
    print(f"\nMap suggestions:\n  HMM fixed it : trip {best.trip_id} "
          f"({best.clf_top1:.1%} -> {best.hmm_top1:.1%})\n  HMM struggled: trip {worst.trip_id} "
          f"({worst.clf_top1:.1%} -> {worst.hmm_top1:.1%})")


if __name__ == "__main__":
    main()
