"""Sanity test for the HMM on a hand-made road network (no real data needed).

Scenario (same idea as the problem-statement video):
  * a 7 x 5 street grid, 100 m blocks, every street two-way
  * a long diagonal road cutting across the grid
  * the car drives a staircase route along the grid
  * GPS = true position + Gaussian noise (sigma = 6 m), 1 Hz, ~10 m/s
  * a fake "classifier" that is usually right but, for a stretch of points,
    confidently prefers the diagonal road (the classic map-matching error)

Expected: classifier-only (argmax) jumps onto the diagonal and back;
the HMM keeps a connected route on the grid and has higher accuracy and
route continuity.

Run:  python -m hmm.test_synthetic          (main scenario)
      python -m hmm.test_synthetic --edge   (+ edge cases)
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_processing.io_utils import EARTH_R, Trip  # noqa: E402
from features.candidates import build_frame  # noqa: E402
from hmm.graph import RoadGraph  # noqa: E402
from hmm.metrics import route_continuity, sequence_metrics, top1_accuracy  # noqa: E402
from hmm.viterbi import argmax_only, build_lattice, decode_lattice  # noqa: E402

LON0, LAT0 = 8.6, 50.0


def xy_to_lonlat(xy):
    lon = LON0 + np.degrees(xy[:, 0] / (EARTH_R * np.cos(np.radians(LAT0))))
    lat = LAT0 + np.degrees(xy[:, 1] / EARTH_R)
    return np.column_stack([lon, lat])


def make_network(nx=7, ny=5, block=100.0):
    nodes, idx = [], {}
    for i in range(nx):
        for j in range(ny):
            idx[(i, j)] = len(nodes)
            nodes.append((i * block, j * block))
    arcs = []
    for i in range(nx):
        for j in range(ny):
            if i + 1 < nx:
                arcs += [(idx[i, j], idx[i + 1, j]), (idx[i + 1, j], idx[i, j])]
            if j + 1 < ny:
                arcs += [(idx[i, j], idx[i, j + 1]), (idx[i, j + 1], idx[i, j])]
    # diagonal road: separate nodes, NOT connected to the grid
    diag = [(150 + k * 50, -20 + k * 45) for k in range(10)]
    d0 = len(nodes)
    nodes += diag
    for k in range(len(diag) - 1):
        arcs += [(d0 + k, d0 + k + 1), (d0 + k + 1, d0 + k)]
    return np.array(nodes, float), np.array(arcs, np.int64), idx


def arc_id(arcs, u, v):
    return int(np.flatnonzero((arcs[:, 0] == u) & (arcs[:, 1] == v))[0])


def main(seed=42):
    rng = np.random.default_rng(seed)
    node_xy, arcs, idx = make_network()
    # staircase route over grid nodes
    path = [(0, 1), (1, 1), (2, 1), (2, 2), (3, 2), (4, 2), (4, 3), (5, 3), (6, 3)]
    route = np.array([arc_id(arcs, idx[a], idx[b]) for a, b in zip(path, path[1:])])
    # drive at 10 m/s, 1 Hz
    pos, lab = [], []
    for r in route:
        a, b = node_xy[arcs[r, 0]], node_xy[arcs[r, 1]]
        for f in np.arange(0, 1, 0.1):
            pos.append(a + f * (b - a))
            lab.append(r)
    pos, lab = np.array(pos), np.array(lab)
    gps_xy = pos + rng.normal(0, 6.0, pos.shape)
    t = np.arange(len(pos), dtype=float)
    track = np.column_stack([xy_to_lonlat(gps_xy), t])
    nodes_ll = xy_to_lonlat(node_xy)
    origin = (float(np.median(track[:, 0])), float(np.median(track[:, 1])))
    trip = Trip("synthetic", track, nodes_ll, arcs, route, origin)

    cand = build_frame(trip, labels=lab)
    # fake classifier: prob decreasing with distance, plus noise ...
    score = -cand["dist"].to_numpy() / 8 + rng.normal(0, 0.5, len(cand))
    # ... and a systematic error: near the diagonal, prefer the diagonal road
    diag_arc = cand["arc"].to_numpy() >= arcs.shape[0] - 18
    bad_pts = cand["point"].between(15, 45)
    score[diag_arc & bad_pts] += 3.0
    cand["prob"] = 1 / (1 + np.exp(-score))

    graph = RoadGraph(trip)
    lat = build_lattice(cand, trip, graph)
    prob = cand["prob"].to_numpy()
    base = argmax_only(lat, prob)
    hmm, info = decode_lattice(lat, prob, {"beta": 5.0})

    print(f"points={len(lab)} candidate rows={len(cand)} info={info}")
    for name, pred in [("classifier-only", base), ("classifier+HMM", hmm)]:
        print(f"{name:16s} top1={top1_accuracy(pred, lab):.3f} "
              f"continuity={route_continuity(pred, graph):.3f} "
              f"{ {k: round(v, 3) for k, v in sequence_metrics(pred, route, graph.arc_len).items()} }")
    assert top1_accuracy(hmm, lab) > top1_accuracy(base, lab)
    assert route_continuity(hmm, graph) > route_continuity(base, graph)
    print("OK: HMM beats classifier-only on the synthetic network")
    return trip, cand, base, hmm, lab




def edge_cases():
    """Points with no candidates, a large time gap, and a disconnected road."""
    trip, cand, base, hmm, lab = main()
    graph = RoadGraph(trip)
    prob_col = "prob"
    # 1) remove all candidates of points 30-34 -> must still get an arc
    c1 = cand[~cand["point"].between(30, 34)].reset_index(drop=True)
    lat = build_lattice(c1, trip, graph)
    out, info = decode_lattice(lat, c1[prob_col].to_numpy())
    assert (out >= 0).all() and len(out) == len(lab), "every point needs an arc"
    print("no-candidate points ->", info, "top1", round(top1_accuracy(out, lab), 3))
    # 2) a 300 s time gap after point 40 -> trip split into 2 segments
    trip.track[41:, 2] += 300
    lat = build_lattice(cand, trip, graph)
    out, info = decode_lattice(lat, cand[prob_col].to_numpy())
    assert info["n_segments"] == 2
    print("time gap ->", info, "top1", round(top1_accuracy(out, lab), 3))
    trip.track[41:, 2] -= 300
    # 3) only the unreachable diagonal road available for points 50-52
    c3 = cand[~(cand["point"].between(50, 52) & (cand["arc"] < trip.arcs.shape[0] - 18))]
    c3 = c3.reset_index(drop=True)
    lat = build_lattice(c3, trip, graph)
    out, info = decode_lattice(lat, c3[prob_col].to_numpy())
    print("disconnected road ->", info, "top1", round(top1_accuracy(out, lab), 3))
    print("OK: edge cases handled")


if __name__ == "__main__":
    edge_cases() if "--edge" in sys.argv else main()
