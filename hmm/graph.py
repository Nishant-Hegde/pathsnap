"""Road-network graph and network (driving) distances for the HMM.

Why this exists
---------------
The HMM transition probability compares two distances between consecutive
GPS points t and t+1:
  * the straight-line distance between the two GPS fixes, and
  * the distance a car would have to DRIVE along the road network to get from
    candidate arc i (at t) to candidate arc j (at t+1).
If those two numbers are very different, the pair (i, j) is implausible
(e.g. jumping onto a parallel street or a crossing diagonal road).

Design decisions
----------------
* Arcs are DIRECTED (from_node -> to_node). EDA showed ~92 % of arcs have a
  reverse arc, so two-way streets are two arcs; one-way streets are respected.
* Edge weight = arc length in metres, using Person A's per-trip local
  projection (data_processing.io_utils.project).
* Networks have up to ~300k arcs, but a trip only touches a corridor around
  its GPS track. We keep only arcs within `buffer_m` of the track
  (Person A's `relevant_arcs`) and run Dijkstra on that sub-graph.
* A position on the network is (arc, frac) where frac in [0, 1] is the
  projection fraction along the arc (Person A's `t_proj`).
* Dijkstra runs lazily from each needed source node with a distance cap
  (`max_route_m`) and results are cached, so each source node is expanded
  at most once per trip.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from features.candidates import relevant_arcs

INF = np.inf


class RoadGraph:
    def __init__(self, trip, buffer_m: float = 400.0, max_route_m: float = 3000.0):
        self.trip = trip
        self.max_route_m = float(max_route_m)
        nxy = trip.node_xy
        arcs = trip.arcs
        self.arcs = arcs
        a, b = nxy[arcs[:, 0]], nxy[arcs[:, 1]]
        # length of every arc in metres (tiny epsilon so Dijkstra never sees 0)
        self.arc_len = np.maximum(np.sqrt(((b - a) ** 2).sum(1)), 1e-3)

        # sub-graph: only arcs near the GPS track
        keep = relevant_arcs(trip.gps_xy, nxy, arcs, buffer_m)
        self.keep_arcs = keep
        nodes = np.unique(arcs[keep].ravel())
        self.node_to_sub = {int(n): i for i, n in enumerate(nodes)}
        self.n_sub = len(nodes)
        u = np.searchsorted(nodes, arcs[keep, 0])
        v = np.searchsorted(nodes, arcs[keep, 1])
        w = self.arc_len[keep]
        # if parallel arcs exist between the same node pair keep the shortest
        order = np.lexsort((w, v, u))
        u, v, w = u[order], v[order], w[order]
        first = np.r_[True, (np.diff(u) != 0) | (np.diff(v) != 0)]
        self.G = csr_matrix((w[first], (u[first], v[first])), shape=(self.n_sub, self.n_sub))
        self._cache: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------ #
    def node_dist(self, src_node: int, dst_node: int) -> float:
        """Shortest driving distance between two ORIGINAL node ids (metres).
        Returns inf if unreachable within max_route_m or outside the corridor."""
        if src_node == dst_node:
            return 0.0
        s = self.node_to_sub.get(int(src_node))
        d = self.node_to_sub.get(int(dst_node))
        if s is None or d is None:
            return INF
        return float(self._row(s)[d])

    def route_distance(self, arc_i: int, frac_i: float, arc_j: int, frac_j: float) -> float:
        """Driving distance from position (arc_i, frac_i) to (arc_j, frac_j).

        * same arc: distance along the arc. GPS jitter can make the projected
          position move slightly backwards, so we use |frac_j - frac_i| * L
          instead of forbidding it (a real backwards move on a one-way street
          is still penalised by the transition model because the GPS points
          would also have to move backwards).
        * different arcs: rest of arc_i + shortest path between
          to_node(arc_i) and from_node(arc_j) + the part of arc_j driven.
          A U-turn onto the reverse arc is covered automatically because
          to_node(arc_i) == from_node(reverse arc).
        """
        Li, Lj = self.arc_len[arc_i], self.arc_len[arc_j]
        if arc_i == arc_j:
            return abs(frac_j - frac_i) * Li
        mid = self.node_dist(self.arcs[arc_i, 1], self.arcs[arc_j, 0])
        if not np.isfinite(mid):
            return INF
        return (1.0 - frac_i) * Li + mid + frac_j * Lj

    def _row(self, sub_src: int) -> np.ndarray:
        row = self._cache.get(sub_src)
        if row is None:
            row = dijkstra(self.G, directed=True, indices=sub_src,
                           limit=self.max_route_m).astype(np.float32)
            self._cache[sub_src] = row
        return row

    def route_matrix(self, arcs_i, frac_i, arcs_j, frac_j) -> np.ndarray:
        """Vectorised route_distance for all pairs (len(arcs_i), len(arcs_j))."""
        arcs_i, arcs_j = np.asarray(arcs_i), np.asarray(arcs_j)
        frac_i, frac_j = np.asarray(frac_i), np.asarray(frac_j)
        Li, Lj = self.arc_len[arcs_i], self.arc_len[arcs_j]
        src = self.arcs[arcs_i, 1]
        dst = self.arcs[arcs_j, 0]
        dst_sub = np.array([self.node_to_sub.get(int(d), -1) for d in dst])
        mid = np.full((len(arcs_i), len(arcs_j)), np.inf)
        for a, s in enumerate(src):
            ss = self.node_to_sub.get(int(s))
            if ss is None:
                continue
            row = self._row(ss)
            ok = dst_sub >= 0
            mid[a, ok] = row[dst_sub[ok]]
            mid[a, dst == s] = 0.0
        M = (1.0 - frac_i)[:, None] * Li[:, None] + mid + (frac_j * Lj)[None, :]
        same = arcs_i[:, None] == arcs_j[None, :]
        M[same] = (np.abs(frac_j[None, :] - frac_i[:, None]) * Li[:, None])[same]
        return M

    def connected(self, arc_i: int, arc_j: int, max_gap_m: float = 0.0) -> bool:
        """True if arc_j can follow arc_i: same arc, head-to-tail adjacent,
        or reachable with at most `max_gap_m` metres of road in between
        (lets very short skipped arcs count as continuous at high speed)."""
        if arc_i == arc_j:
            return True
        a, b = self.arcs[arc_i, 1], self.arcs[arc_j, 0]
        if a == b:
            return True
        if max_gap_m <= 0:
            return False
        return self.node_dist(a, b) <= max_gap_m
