"""End-to-end prediction pipeline - this is what the API calls.

    trip  ->  candidates + features   (Person A: features/candidates.build_frame)
          ->  classifier probability  (Person B: models/predict.score_candidates)
          ->  HMM / Viterbi decoding  (hmm/viterbi)
          ->  one road arc per GPS point

No reference route / labels are used anywhere in this file.

CLI (development only - train / student_test):
    python pipeline.py --trip data/student_test/00000065
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from data_processing.io_utils import Trip, load_trip  # noqa: E402
from features.candidates import build_frame  # noqa: E402
from hmm.graph import RoadGraph  # noqa: E402
from hmm.viterbi import DEFAULT_PARAMS, decode  # noqa: E402

HMM_CONFIG = os.path.join(ROOT, "artifacts", "hmm_config.json")


def load_hmm_params(path: str = HMM_CONFIG) -> dict:
    p = dict(DEFAULT_PARAMS)
    if os.path.exists(path):
        p.update(json.load(open(path)))
    return p


def _scorer():
    """Person B's scorer if available, else None (distance-only HMM)."""
    try:
        from models.predict import score_candidates  # type: ignore
        return score_candidates
    except Exception as e:  # model not trained / not merged yet
        print(f"[pipeline] WARNING: classifier unavailable ({e}); using distance-only HMM")
        return None


_SCORE = None
_PARAMS = None


def predict_route(trip: Trip, params: dict | None = None) -> pd.DataFrame:
    """Return DataFrame[obs_idx, arc_id] - exactly one arc per GPS point.
    df.attrs['info'] holds timings and decoder diagnostics."""
    global _SCORE, _PARAMS
    if _PARAMS is None:
        _PARAMS = load_hmm_params()
        _SCORE = _scorer()
    params = {**_PARAMS, **(params or {})}
    t0 = time.time()
    cand = build_frame(trip)                      # no labels
    t1 = time.time()
    if _SCORE is not None:
        cand = _SCORE(cand)                       # adds column 'prob'
    else:
        params = {**params, "alpha": 0.0}        # distance-only emissions
    t2 = time.time()
    out = decode(cand, trip, params, graph=RoadGraph(trip))
    info = out.attrs.get("info", {})
    out = out.rename(columns={"point": "obs_idx", "arc": "arc_id"})  # API contract names
    out.attrs["info"] = info
    t3 = time.time()
    out.attrs["info"] = {**out.attrs.get("info", {}),
                         "candidates_ms": 1000 * (t1 - t0), "scoring_ms": 1000 * (t2 - t1),
                         "hmm_ms": 1000 * (t3 - t2), "total_ms": 1000 * (t3 - t0),
                         "classifier_used": _SCORE is not None}
    return out


def route_geometry(trip: Trip, arcs_per_point: np.ndarray) -> list:
    """Ordered, de-duplicated list of predicted arcs as [[lat, lon], [lat, lon]]
    segments - the format the API returns to the UI."""
    seq = arcs_per_point[arcs_per_point >= 0]
    seq = seq[np.r_[True, seq[1:] != seq[:-1]]] if len(seq) else seq
    nodes = trip.nodes
    return [[[float(nodes[trip.arcs[a, 0], 1]), float(nodes[trip.arcs[a, 0], 0])],
             [float(nodes[trip.arcs[a, 1], 1]), float(nodes[trip.arcs[a, 1], 0])]] for a in seq]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trip", required=True, help="path to a trip folder (train/student_test)")
    a = ap.parse_args()
    assert "eval" not in a.trip.lower(), "do not run development code on evaluator data"
    trip = load_trip(a.trip, with_route=False)
    res = predict_route(trip)
    print(res.head())
    print(json.dumps(res.attrs["info"], indent=2))
