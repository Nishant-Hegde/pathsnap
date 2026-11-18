"""Candidate generation and feature engineering.

Because each trip ships with its own road network, arc ids are not shared
between trips. We therefore frame the problem as a *candidate-scoring* task:
for each GPS point we generate up to K nearby arcs (candidates) and build a
feature vector for each (point, candidate) pair. A classifier predicts
P(candidate is the true arc | features). These probabilities become HMM
emission scores.

The same function is used for training, student-testing and the API, so the
preprocessing is identical everywhere (no labels needed at inference time).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from data_processing.io_utils import Trip, angle_diff, bearing_xy, point_segment

DEFAULT_RADIUS = 60.0   # m; student_test p99 GPS->route distance is ~28 m
DEFAULT_K = 10
DENSIFY_STEP = 5.0      # m

FEATURES = [
    "dist", "dist_rank", "dist_minus_min", "dist_ratio",
    "head_diff", "head_diff_x_speed", "speed", "arc_len", "t_proj", "n_cand",
    "dist_prev", "dist_next", "dist_ctx_mean", "head_diff_next",
    "dt_prev", "dt_next", "jitter", "from_indeg", "to_outdeg", "two_way",
    "end_proximity",
]


def _gps_kinematics(xy: np.ndarray, t: np.ndarray, win: int = 2):
    n = len(xy)
    i0 = np.clip(np.arange(n) - win, 0, n - 1)
    i1 = np.clip(np.arange(n) + win, 0, n - 1)
    d = xy[i1] - xy[i0]
    span = np.maximum(t[i1] - t[i0], 1e-6)
    speed = np.sqrt((d ** 2).sum(1)) / span
    heading = bearing_xy(d[:, 0], d[:, 1])
    # forward-only heading (helps at the start of a turn)
    j1 = np.clip(np.arange(n) + 1, 0, n - 1)
    dn = xy[j1] - xy
    head_next = bearing_xy(dn[:, 0], dn[:, 1])
    dt = np.diff(t)
    dt_prev = np.r_[0.0, dt]
    dt_next = np.r_[dt, 0.0]
    # jitter: distance from point to the midpoint of its neighbours
    ip = np.clip(np.arange(n) - 1, 0, n - 1)
    jitter = np.sqrt(((xy - (xy[ip] + xy[j1]) / 2) ** 2).sum(1))
    return speed, heading, head_next, dt_prev, dt_next, jitter


def relevant_arcs(gps_xy, node_xy, arcs, radius):
    """Arcs that could be within `radius` of any GPS point."""
    a, b = node_xy[arcs[:, 0]], node_xy[arcs[:, 1]]
    mid = (a + b) / 2
    half = np.sqrt(((b - a) ** 2).sum(1)) / 2
    tree = cKDTree(gps_xy)
    d, _ = tree.query(mid, k=1)
    return np.where(d <= radius + half)[0]


def generate_candidates(trip: Trip, radius=DEFAULT_RADIUS, k=DEFAULT_K):
    """Return arrays (point_idx, arc_id, dist, t_proj) for all candidates."""
    gps = trip.gps_xy
    nxy = trip.node_xy
    arcs = trip.arcs
    keep = relevant_arcs(gps, nxy, arcs, radius)
    a, b = nxy[arcs[keep, 0]], nxy[arcs[keep, 1]]
    L = np.sqrt(((b - a) ** 2).sum(1))
    nseg = np.maximum(1, np.ceil(L / DENSIFY_STEP).astype(int))
    owner = np.repeat(np.arange(len(keep)), nseg + 1)
    frac = np.concatenate([np.linspace(0, 1, s + 1) for s in nseg]) if len(keep) else np.zeros(0)
    samples = a[owner] + frac[:, None] * (b - a)[owner]
    tree = cKDTree(samples)
    hits = tree.query_ball_point(gps, r=radius + DENSIFY_STEP / 2)
    P, A = [], []
    for i, h in enumerate(hits):
        if h:
            u = np.unique(owner[h])
            P.append(np.full(len(u), i))
            A.append(u)
    if not P:
        return (np.zeros(0, int),) * 2 + (np.zeros(0),) * 2
    P = np.concatenate(P)
    A = np.concatenate(A)
    d, tp = point_segment(gps[P], a[A], b[A])
    m = d <= radius
    P, A, d, tp = P[m], A[m], d[m], tp[m]
    # top-k per point by distance
    order = np.lexsort((d, P))
    P, A, d, tp = P[order], A[order], d[order], tp[order]
    first = np.r_[0, np.flatnonzero(np.diff(P)) + 1]
    rank = np.arange(len(P)) - np.repeat(first, np.diff(np.r_[first, len(P)]))
    m = rank < k
    return P[m], keep[A[m]], d[m], tp[m]


def build_frame(trip: Trip, labels: np.ndarray | None = None,
                radius=DEFAULT_RADIUS, k=DEFAULT_K) -> pd.DataFrame:
    """One row per (GPS point, candidate arc). If labels given adds `y`."""
    gps = trip.gps_xy
    t = trip.track[:, 2]
    nxy = trip.node_xy
    arcs = trip.arcs
    P, A, d, tp = generate_candidates(trip, radius, k)
    speed, heading, head_next, dt_prev, dt_next, jitter = _gps_kinematics(gps, t)

    a, b = nxy[arcs[A, 0]], nxy[arcs[A, 1]]
    ab = b - a
    arc_len = np.sqrt((ab ** 2).sum(1))
    arc_bear = bearing_xy(ab[:, 0], ab[:, 1])

    df = pd.DataFrame({"point": P, "arc": A, "dist": d, "t_proj": tp})
    g = df.groupby("point")["dist"]
    df["dist_rank"] = g.rank(method="first") - 1
    mn = g.transform("min")
    df["dist_minus_min"] = d - mn
    df["dist_ratio"] = d / (mn + 1.0)
    df["n_cand"] = g.transform("size")
    hd = angle_diff(heading[P], arc_bear)
    df["head_diff"] = hd
    # heading only meaningful when moving; interaction lets models learn that
    df["speed"] = speed[P]
    df["head_diff_x_speed"] = hd * np.minimum(speed[P], 15) / 15
    df["head_diff_next"] = angle_diff(head_next[P], arc_bear)
    df["arc_len"] = arc_len
    # context: distance of the previous / next GPS point to the same arc
    n = len(gps)
    ip = np.clip(P - 1, 0, n - 1)
    inx = np.clip(P + 1, 0, n - 1)
    df["dist_prev"], _ = point_segment(gps[ip], a, b)
    df["dist_next"], _ = point_segment(gps[inx], a, b)
    ip2 = np.clip(P - 3, 0, n - 1)
    in2 = np.clip(P + 3, 0, n - 1)
    d2p, _ = point_segment(gps[ip2], a, b)
    d2n, _ = point_segment(gps[in2], a, b)
    df["dist_ctx_mean"] = (df.dist_prev + df.dist_next + d2p + d2n + d) / 5
    df["dt_prev"] = dt_prev[P]
    df["dt_next"] = dt_next[P]
    df["jitter"] = jitter[P]
    V = len(trip.nodes)
    indeg = np.bincount(arcs[:, 1], minlength=V)
    outdeg = np.bincount(arcs[:, 0], minlength=V)
    df["from_indeg"] = indeg[arcs[A, 0]]
    df["to_outdeg"] = outdeg[arcs[A, 1]]
    key = arcs[:, 0].astype(np.int64) * (V + 1) + arcs[:, 1]
    rkey = arcs[A, 1].astype(np.int64) * (V + 1) + arcs[A, 0]
    df["two_way"] = np.isin(rkey, key).astype(np.int8)
    df["end_proximity"] = np.minimum(tp, 1 - tp) * arc_len
    df["trip_id"] = trip.trip_id
    if labels is not None:
        df["y"] = (A == labels[P]).astype(np.int8)
    return df
