"""Derive per-GPS-point reference labels from the .route arc sequence.

The .route file is a chronological list of arcs, NOT one label per GPS row.
We align GPS points to route arcs with a monotone dynamic programme:

    j(0) <= j(1) <= ... <= j(N-1)        (vehicle never goes back in the route)
    minimise  sum_i  min(dist(gps_i, route_arc_{j(i)}), CAP)

This is a DTW-style alignment; it lets a short arc receive zero points and
handles routes that revisit the same street (monotonicity disambiguates).
The label of point i is route[j(i)], plus the distance to that arc so we can
flag unreliable labels.
"""
import numpy as np

from .io_utils import point_segment

CAP = 100.0  # metres; caps the influence of outliers / gaps


def align_route(gps_xy: np.ndarray, node_xy: np.ndarray, arcs: np.ndarray, route: np.ndarray):
    a = node_xy[arcs[route, 0]]
    b = node_xy[arcs[route, 1]]
    N, M = len(gps_xy), len(route)
    back = np.empty((N, M), dtype=np.int32)
    prev = None
    for i in range(N):
        d, _ = point_segment(np.broadcast_to(gps_xy[i], (M, 2)), a, b)
        d = np.minimum(d, CAP)
        if prev is None:
            # first point may start at any arc but we prefer the start of route
            cost = d + np.arange(M) * 1e-3
            back[i] = np.arange(M)
        else:
            # prefix minimum of prev -> best predecessor with index <= j
            idx = np.arange(M)
            better = np.r_[True, prev[1:] < np.minimum.accumulate(prev)[:-1]]
            arg = np.maximum.accumulate(np.where(better, idx, 0))
            cost = d + prev[arg]
            back[i] = arg
        prev = cost
    j = np.empty(N, dtype=np.int64)
    j[-1] = int(np.argmin(prev))
    for i in range(N - 1, 0, -1):
        j[i - 1] = back[i, j[i]]
    lab = route[j]
    d_lab, _ = point_segment(gps_xy, node_xy[arcs[lab, 0]], node_xy[arcs[lab, 1]])
    return lab, j, d_lab
