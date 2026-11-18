"""Raw-data loading and geometry helpers for PathSnap.

File formats (verified during EDA, see eda/eda.py):
  .track : lon lat t      (whitespace, no header)
  .nodes : lon lat        (row index = node_id)
  .arcs  : from to        (row index = arc_id, directed)
  .route : arc_id         (chronological ground-truth arcs)

Distances: each trip covers a few km, so we project lon/lat to a local
equirectangular plane (metres) centred on the trip. At this scale the error
versus true geodesic distance is far below GPS noise (< 0.1 %).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

EARTH_R = 6_371_000.0


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray          # (N, 3) lon, lat, t
    nodes: np.ndarray          # (V, 2) lon, lat
    arcs: np.ndarray           # (E, 2) from, to
    route: np.ndarray | None   # (R,) arc ids, None if not supplied
    origin: tuple[float, float]  # lon0, lat0 used for projection

    # projected coordinates (metres)
    @property
    def gps_xy(self) -> np.ndarray:
        return project(self.track[:, :2], self.origin)

    @property
    def node_xy(self) -> np.ndarray:
        return project(self.nodes, self.origin)


def project(lonlat: np.ndarray, origin: tuple[float, float]) -> np.ndarray:
    lon0, lat0 = origin
    x = np.radians(lonlat[:, 0] - lon0) * EARTH_R * np.cos(np.radians(lat0))
    y = np.radians(lonlat[:, 1] - lat0) * EARTH_R
    return np.column_stack([x, y])


def _load(path, dtype=float):
    a = np.loadtxt(path, dtype=dtype, ndmin=2)
    return a


def load_trip(trip_dir: str, with_route: bool = True) -> Trip:
    tid = os.path.basename(os.path.normpath(trip_dir))
    track = _load(os.path.join(trip_dir, f"{tid}.track"))
    nodes = _load(os.path.join(trip_dir, f"{tid}.nodes"))
    arcs = _load(os.path.join(trip_dir, f"{tid}.arcs"), dtype=np.int64)
    rpath = os.path.join(trip_dir, f"{tid}.route")
    route = None
    if with_route and os.path.exists(rpath):
        route = np.loadtxt(rpath, dtype=np.int64, ndmin=1)
    # sort by timestamp, drop exact duplicate timestamps / invalid coords
    track = track[np.argsort(track[:, 2], kind="stable")]
    ok = np.isfinite(track).all(1) & (np.abs(track[:, 0]) <= 180) & (np.abs(track[:, 1]) <= 90)
    track = track[ok]
    keep = np.r_[True, np.diff(track[:, 2]) > 0]
    track = track[keep]
    origin = (float(np.median(track[:, 0])), float(np.median(track[:, 1])))
    return Trip(tid, track, nodes, arcs, route, origin)


def list_trips(partition_dir: str) -> list[str]:
    return sorted(
        os.path.join(partition_dir, d)
        for d in os.listdir(partition_dir)
        if os.path.isdir(os.path.join(partition_dir, d))
    )


def point_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray):
    """Distance from points p (n,2) to segments a->b (n,2).
    Returns distance (n,) and projection fraction t in [0,1] (n,)."""
    ab = b - a
    L2 = (ab ** 2).sum(-1)
    t = np.where(L2 > 0, ((p - a) * ab).sum(-1) / np.maximum(L2, 1e-12), 0.0)
    t = np.clip(t, 0.0, 1.0)
    proj = a + t[..., None] * ab
    return np.sqrt(((p - proj) ** 2).sum(-1)), t


def bearing_xy(dx, dy):
    """Bearing in degrees, 0 = north, clockwise."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def angle_diff(a, b):
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)
