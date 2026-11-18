"""Route-level metrics for comparing classifier-only vs classifier+HMM.

All functions take per-GPS-point arc arrays of equal length
(pred[i], label[i] = arc id for GPS point i).
"""
from __future__ import annotations

import numpy as np

from hmm.graph import RoadGraph


def top1_accuracy(pred: np.ndarray, label: np.ndarray) -> float:
    """Observation-level Top-1 accuracy: fraction of GPS points whose
    predicted arc equals the reference arc."""
    return float(np.mean(pred == label))


def route_continuity(pred: np.ndarray, graph: RoadGraph, max_gap_m: float = 0.0) -> float:
    """Fraction of consecutive GPS-point pairs whose predicted arcs are the
    same arc or head-to-tail connected (to_node(a) == from_node(b)).
    With max_gap_m > 0, a pair also counts as continuous if b is reachable
    from a within that many metres of road (e.g. one very short skipped arc)."""
    if len(pred) < 2:
        return 1.0
    a, b = pred[:-1], pred[1:]
    ok = (a == b) | (graph.arcs[a, 1] == graph.arcs[b, 0])
    if max_gap_m > 0:
        for k in np.flatnonzero(~ok):
            ok[k] = graph.connected(int(a[k]), int(b[k]), max_gap_m)
    return float(ok.mean())


def collapse(seq: np.ndarray) -> np.ndarray:
    """Remove consecutive duplicates: per-point arcs -> traversed arc sequence."""
    seq = np.asarray(seq)
    if len(seq) == 0:
        return seq
    return seq[np.r_[True, seq[1:] != seq[:-1]]]


def sequence_metrics(pred: np.ndarray, route: np.ndarray, arc_len: np.ndarray) -> dict:
    """Sequence-level comparison of the predicted traversed arcs against the
    reference route arcs (as sets, weighted by arc length so that tiny arcs
    don't dominate):
      precision = length of predicted arcs that are on the reference route
                  / total predicted length
      recall    = length of reference arcs recovered / total reference length
      jaccard   = |pred ∩ ref| / |pred ∪ ref|  (length-weighted)
    """
    P = set(collapse(pred).tolist()) - {-1}
    R = set(np.asarray(route).tolist())
    L = lambda s: float(arc_len[list(s)].sum()) if s else 0.0  # noqa: E731
    inter = L(P & R)
    return {
        "route_precision": inter / L(P) if P else 0.0,
        "route_recall": inter / L(R) if R else 0.0,
        "route_jaccard": inter / L(P | R) if (P | R) else 0.0,
    }
