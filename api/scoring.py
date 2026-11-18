"""Server-only evaluation helpers.

This module is deliberately separate from the prediction path.  It may read a
reference route to calculate aggregate metrics, but never returns that route or
point-level labels to the client.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data_processing.io_utils import Trip
from data_processing.labels import align_route


def route_continuity(arc_ids: np.ndarray, arcs: np.ndarray) -> float | None:
    """Share of adjacent predictions that repeat or connect head-to-tail."""
    if len(arc_ids) < 2:
        return None
    left, right = arc_ids[:-1], arc_ids[1:]
    connected = (left == right) | (arcs[left, 1] == arcs[right, 0])
    return float(connected.mean())


def score_prediction(predicted: pd.DataFrame, trip: Trip) -> dict[str, float | None]:
    """Calculate evaluation metrics on the server when a reference is available."""
    chosen = predicted.sort_values("obs_idx")["arc_id"].to_numpy(dtype=np.int64)
    continuity = route_continuity(chosen, trip.arcs)
    if trip.route is None:
        return {"top1_accuracy": None, "route_continuity": continuity}

    labels, _, _ = align_route(trip.gps_xy, trip.node_xy, trip.arcs, trip.route)
    if len(labels) != len(chosen):
        raise ValueError("Prediction did not return one arc for every GPS observation.")
    return {
        "top1_accuracy": float((chosen == labels).mean()),
        "route_continuity": continuity,
    }
