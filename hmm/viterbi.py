"""HMM + Viterbi route reconstruction.

HMM formulation
---------------
* Observations  o_t : the GPS fixes of a trip, in time order.
* Hidden states     : at time t, the candidate arcs generated for GPS point t
                      by Person A (features/candidates.py, radius 60 m, K<=10).
                      Each state keeps its (arc id, distance, projection
                      fraction) so the candidate -> road-segment mapping is
                      preserved exactly.
* Emission  log B_j(o_t) =   alpha       * log(p_j + eps)
                           + (1 - alpha) * (-0.5 * (d_j / sigma)^2)
      p_j   = classifier probability that candidate j is the true arc
      d_j   = GPS-to-arc distance in metres (Gaussian GPS-noise model)
      alpha = 1 -> trust only the classifier; alpha = 0 -> pure distance HMM.
* Transition log A_ij  = -| d_gc - d_route(i, j) | / beta        (Newson & Krumm 2009)
      d_gc    = straight-line distance between GPS points t and t+1
      d_route = driving distance along the network from candidate i to j
      A pair is impossible (-inf) if the arc is unreachable or the implied
      speed d_route / dt exceeds vmax.
* Viterbi recurrence (log space):
      delta_1(j)   = log B_j(o_1)
      delta_t(j)   = max_i [ delta_{t-1}(i) + log A_ij ] + log B_j(o_t)
      psi_t(j)     = argmax_i [ ... ]
      path_T       = argmax_j delta_T(j); path_{t-1} = psi_t(path_t)

Edge cases
----------
* GPS point with no candidates  -> skipped by the HMM (the next transition
  is computed between the surrounding points); after decoding it gets the arc
  of the nearest decoded point in time.
* Time gap > gap_s seconds      -> the trip is split into independent
  segments, each decoded separately (no transition across the gap).
* All transitions -inf ("disconnected" road segments, e.g. the true road is
  missing from the map or outside the corridor) -> the chain is closed at
  t-1 and a new chain starts at t from its emissions. Breaks are counted.

Performance
-----------
`build_lattice` does all the expensive graph work (route distances) ONCE per
trip; `decode_lattice` is pure numpy, so HMM parameters can be tuned quickly
over many settings, and models can be swapped without recomputing paths.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from hmm.graph import RoadGraph

DEFAULT_PARAMS = {
    "alpha": 1.0,        # weight of classifier log-prob vs Gaussian distance term
    "sigma": 5.0,        # m, GPS noise std for the distance term
    "beta": 5.0,         # m, scale of the transition |d_gc - d_route| penalty
    "eps": 1e-4,         # floor added to probabilities before log
    "vmax": 60.0,        # m/s (216 km/h) max plausible speed between fixes
    "speed_slack": 50.0, # m, tolerance added to vmax * dt
    "gap_s": 60.0,       # s, split trip into segments at larger time gaps
}


@dataclass
class Lattice:
    trip_id: str
    n_points: int                      # all GPS points in the trip
    pts: np.ndarray                    # GPS point ids that have candidates (time order)
    seg_start: np.ndarray              # bool per pts: starts a new segment (gap)
    arcs: list = field(default_factory=list)   # per pts: candidate arc ids
    dist: list = field(default_factory=list)   # per pts: GPS-to-arc distance
    rows: list = field(default_factory=list)   # per pts: row index into scored df
    d_gc: np.ndarray = None            # per transition k (pts[k-1] -> pts[k])
    dt: np.ndarray = None
    d_route: list = field(default_factory=list)  # per transition: (n_prev, n_cur) metres


def build_lattice(cand_df: pd.DataFrame, trip, graph: RoadGraph | None = None,
                  gap_s: float = DEFAULT_PARAMS["gap_s"]) -> Lattice:
    """Pre-compute everything that does not depend on the HMM parameters
    or on the classifier probabilities. `cand_df` = Person A's build_frame
    output for ONE trip (columns point, arc, dist, t_proj)."""
    if graph is None:
        graph = RoadGraph(trip)
    df = cand_df.reset_index(drop=True)
    gps = trip.gps_xy
    t = trip.track[:, 2]
    order = np.lexsort((df["dist"].to_numpy(), df["point"].to_numpy()))
    P = df["point"].to_numpy()[order]
    starts = np.r_[0, np.flatnonzero(np.diff(P)) + 1]
    ends = np.r_[starts[1:], len(P)]
    pts = P[starts]
    lat = Lattice(trip.trip_id, len(t), pts, np.zeros(len(pts), bool))
    A = df["arc"].to_numpy()
    D = df["dist"].to_numpy()
    F = df["t_proj"].to_numpy()
    fracs = []
    for s, e in zip(starts, ends):
        idx = order[s:e]
        lat.rows.append(idx)
        lat.arcs.append(A[idx])
        lat.dist.append(D[idx])
        fracs.append(F[idx])
    n = len(pts)
    lat.d_gc = np.zeros(n)
    lat.dt = np.zeros(n)
    lat.d_route = [None] * n
    if n:
        lat.seg_start[0] = True
    for k in range(1, n):
        p0, p1 = pts[k - 1], pts[k]
        dt = t[p1] - t[p0]
        lat.dt[k] = dt
        lat.d_gc[k] = float(np.hypot(*(gps[p1] - gps[p0])))
        if dt > gap_s:
            lat.seg_start[k] = True
            continue
        a0, f0, a1, f1 = lat.arcs[k - 1], fracs[k - 1], lat.arcs[k], fracs[k]
        lat.d_route[k] = graph.route_matrix(a0, f0, a1, f1)
    return lat


def _emission(lat: Lattice, k: int, prob: np.ndarray | None, p: dict) -> np.ndarray:
    d = lat.dist[k]
    e_dist = -0.5 * (d / p["sigma"]) ** 2
    if prob is None:
        return e_dist
    pr = prob[lat.rows[k]]
    e_cls = np.log(pr + p["eps"])
    return p["alpha"] * e_cls + (1.0 - p["alpha"]) * e_dist


def decode_lattice(lat: Lattice, prob: np.ndarray | None, params: dict | None = None):
    """Run Viterbi. `prob` = classifier probability per row of the candidate
    df used to build the lattice (None -> distance-only HMM).
    Returns (arc per GPS point (n_points,), info dict)."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    n = len(lat.pts)
    chosen = np.full(n, -1, dtype=np.int64)  # index into candidates of pts[k]
    n_breaks = 0

    def backtrack(k_end, delta, psi, k_start):
        j = int(np.argmax(delta))
        for k in range(k_end, k_start - 1, -1):
            chosen[k] = j
            if k > k_start:
                j = int(psi[k][j])

    delta, psi, k0 = None, {}, 0
    for k in range(n):
        em = _emission(lat, k, prob, p)
        new_chain = lat.seg_start[k] or delta is None
        if not new_chain:
            M = lat.d_route[k]
            logA = -np.abs(lat.d_gc[k] - M) / p["beta"]
            too_fast = M > p["vmax"] * max(lat.dt[k], 1e-3) + p["speed_slack"]
            logA[~np.isfinite(M) | too_fast] = -np.inf
            score = delta[:, None] + logA
            best = score.max(0)
            if not np.isfinite(best).any():
                n_breaks += 1          # disconnected: close chain, restart here
                new_chain = True
            else:
                psi[k] = score.argmax(0)
                delta = best + em
        if new_chain:
            if delta is not None:
                backtrack(k - 1, delta, psi, k0)
            delta, psi, k0 = em.copy(), {}, k
    if delta is not None:
        backtrack(n - 1, delta, psi, k0)

    arc_at = np.array([lat.arcs[k][chosen[k]] for k in range(n)], dtype=np.int64)
    out = _fill_all_points(lat, arc_at)
    info = {"n_breaks": n_breaks, "n_segments": int(lat.seg_start.sum()),
            "n_points_without_candidates": int(lat.n_points - n)}
    return out, info


def argmax_only(lat: Lattice, prob: np.ndarray | None) -> np.ndarray:
    """Classifier-only baseline: highest-probability candidate per point
    (or nearest arc if prob is None), with the same gap filling."""
    arc_at = np.empty(len(lat.pts), dtype=np.int64)
    for k in range(len(lat.pts)):
        if prob is None:
            arc_at[k] = lat.arcs[k][np.argmin(lat.dist[k])]
        else:
            arc_at[k] = lat.arcs[k][np.argmax(prob[lat.rows[k]])]
    return _fill_all_points(lat, arc_at)


def _fill_all_points(lat: Lattice, arc_at: np.ndarray) -> np.ndarray:
    """Give every GPS point an arc: points without candidates take the arc of
    the nearest point (in index) that has one; -1 only if the trip has none."""
    out = np.full(lat.n_points, -1, dtype=np.int64)
    if len(lat.pts) == 0:
        return out
    out[lat.pts] = arc_at
    idx = np.arange(lat.n_points)
    pos = np.searchsorted(lat.pts, idx)
    left = lat.pts[np.clip(pos - 1, 0, len(lat.pts) - 1)]
    right = lat.pts[np.clip(pos, 0, len(lat.pts) - 1)]
    nearest = np.where(np.abs(idx - left) <= np.abs(right - idx), left, right)
    missing = out == -1
    out[missing] = out[nearest[missing]]
    return out


def decode(scored_df: pd.DataFrame, trip, params: dict | None = None,
           graph: RoadGraph | None = None, prob_col: str = "prob") -> pd.DataFrame:
    """Contract entry point: one arc per GPS point -> DataFrame[point, arc]."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    lat = build_lattice(scored_df, trip, graph, gap_s=p["gap_s"])
    prob = scored_df[prob_col].to_numpy() if prob_col in scored_df else None
    arcs, info = decode_lattice(lat, prob, p)
    out = pd.DataFrame({"point": np.arange(lat.n_points), "arc": arcs})
    out.attrs["info"] = info
    return out
