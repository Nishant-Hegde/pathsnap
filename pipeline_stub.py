"""Temporary end-to-end predictor used while the HMM pipeline is integrated.

It chooses the closest candidate arc for each GPS observation.  It uses no
reference route and is intentionally isolated so api/main.py can switch to
``from pipeline import predict_route`` when Person C's implementation arrives.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from data_processing.io_utils import Trip, point_segment
from features.candidates import generate_candidates


def _nearest_arc_for_missing_points(trip: Trip, missing: np.ndarray) -> np.ndarray:
    """Fallback for points outside the candidate radius: choose nearest endpoint arc."""
    arcs, nodes = trip.arcs, trip.node_xy
    tree = cKDTree(nodes)
    _, node_ids = tree.query(trip.gps_xy[missing], k=1)
    incident: dict[int, list[int]] = {}
    for arc_id, (start, end) in enumerate(arcs):
        incident.setdefault(int(start), []).append(arc_id)
        incident.setdefault(int(end), []).append(arc_id)
    chosen = []
    for point_index, node_id in zip(missing, node_ids):
        options = np.asarray(incident[int(node_id)], dtype=np.int64)
        a, b = nodes[arcs[options, 0]], nodes[arcs[options, 1]]
        p = np.broadcast_to(trip.gps_xy[point_index], a.shape)
        distances, _ = point_segment(p, a, b)
        chosen.append(int(options[np.argmin(distances)]))
    return np.asarray(chosen, dtype=np.int64)


def predict_route(trip: Trip) -> pd.DataFrame:
    """Return exactly one nearest candidate arc for each GPS observation."""
    point, arc, distance, _ = generate_candidates(trip)
    best = pd.DataFrame({"obs_idx": point, "arc_id": arc, "distance": distance})
    best = best.sort_values(["obs_idx", "distance", "arc_id"]).drop_duplicates("obs_idx")

    all_points = np.arange(len(trip.track), dtype=np.int64)
    missing = np.setdiff1d(all_points, best["obs_idx"].to_numpy(dtype=np.int64))
    if len(missing):
        fallback = pd.DataFrame({"obs_idx": missing, "arc_id": _nearest_arc_for_missing_points(trip, missing)})
        best = pd.concat([best[["obs_idx", "arc_id"]], fallback], ignore_index=True)
    else:
        best = best[["obs_idx", "arc_id"]]
    return best.sort_values("obs_idx", ignore_index=True)
