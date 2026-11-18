"""Folium map of a trip: GPS observations, road network, reference route,
classifier-only route and HMM route as separate toggleable layers.

Used for the report (section 6) and reusable by the UI (call with
reference=None so ground truth is never shown to the evaluator UI).

CLI (train / student_test only):
    python -m viz.route_map --trip data/student_test/00000065 --out viz/maps/st_00000065.html
"""
from __future__ import annotations

import argparse
import os
import sys

import folium
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_processing.io_utils import Trip, load_trip  # noqa: E402

STYLE = {  # layer name -> (colour, weight, dash)
    "Reference route": ("#2ca02c", 8, None),
    "Classifier-only": ("#d62728", 4, "6 6"),
    "HMM / Viterbi": ("#1f77b4", 5, None),
}


def _arc_lines(trip: Trip, arcs_seq):
    nodes = trip.nodes
    a = np.asarray(arcs_seq)
    a = a[a >= 0]
    if len(a) == 0:
        return []
    a = a[np.r_[True, a[1:] != a[:-1]]]
    return [[(nodes[trip.arcs[x, 0], 1], nodes[trip.arcs[x, 0], 0]),
             (nodes[trip.arcs[x, 1], 1], nodes[trip.arcs[x, 1], 0])] for x in a]


def make_route_map(trip: Trip, routes: dict, reference=None, network_buffer_m=150.0,
                   max_network_arcs=6000) -> folium.Map:
    """routes: {layer name: per-point arc array}; reference: route arc list or None."""
    lat, lon = trip.track[:, 1], trip.track[:, 0]
    m = folium.Map(location=[float(np.mean(lat)), float(np.mean(lon))],
                   zoom_start=14, tiles="OpenStreetMap", control_scale=True)

    # road network near the track (light grey)
    from features.candidates import relevant_arcs
    keep = relevant_arcs(trip.gps_xy, trip.node_xy, trip.arcs, network_buffer_m)[:max_network_arcs]
    net = folium.FeatureGroup(name="Road network", show=True)
    for x in keep:
        u, v = trip.arcs[x]
        folium.PolyLine([(trip.nodes[u, 1], trip.nodes[u, 0]), (trip.nodes[v, 1], trip.nodes[v, 0])],
                        color="#9e9e9e", weight=2, opacity=0.6).add_to(net)
    net.add_to(m)

    if reference is not None:
        routes = {"Reference route": reference, **routes}
    for name, arcs_seq in routes.items():
        colour, weight, dash = STYLE.get(name, ("#ff7f0e", 4, None))
        fg = folium.FeatureGroup(name=name, show=True)
        for seg in _arc_lines(trip, arcs_seq):
            folium.PolyLine(seg, color=colour, weight=weight, opacity=0.85,
                            dash_array=dash, tooltip=name).add_to(fg)
        fg.add_to(m)

    gps = folium.FeatureGroup(name="GPS observations", show=True)
    step = max(1, len(lat) // 3000)   # keep the HTML light for long trips
    for y, x in zip(lat[::step], lon[::step]):
        folium.CircleMarker((y, x), radius=2, color="#000000", fill=True,
                            fill_opacity=0.7, weight=0).add_to(gps)
    gps.add_to(m)
    folium.Marker((lat[0], lon[0]), tooltip="Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker((lat[-1], lon[-1]), tooltip="End", icon=folium.Icon(color="red")).add_to(m)

    legend = "".join(f'<div><span style="display:inline-block;width:22px;height:4px;'
                     f'background:{c};margin-right:6px;vertical-align:middle"></span>{n}</div>'
                     for n, (c, _, _) in STYLE.items() if n in routes)
    legend += ('<div><span style="display:inline-block;width:8px;height:8px;border-radius:4px;'
               'background:#000;margin:0 13px 0 7px"></span>GPS observation</div>'
               '<div><span style="display:inline-block;width:22px;height:3px;background:#9e9e9e;'
               'margin-right:6px;vertical-align:middle"></span>Road network</div>')
    m.get_root().html.add_child(folium.Element(
        '<div style="position:fixed;bottom:25px;left:25px;z-index:9999;background:white;'
        'padding:8px 10px;border:1px solid #999;border-radius:6px;font:12px sans-serif">'
        f'<b>{trip.trip_id}</b>{legend}</div>'))
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[float(lat.min()), float(lon.min())], [float(lat.max()), float(lon.max())]])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trip", required=True)
    ap.add_argument("--scored", default=None, help="B's scored table, for the classifier-only layer")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    assert "eval" not in a.trip.lower(), "maps with reference routes: train/student_test only"
    from hmm.graph import RoadGraph
    from hmm.viterbi import argmax_only, build_lattice, decode_lattice
    from pipeline import load_hmm_params
    from features.candidates import build_frame
    import pandas as pd
    trip = load_trip(a.trip)
    params = load_hmm_params()
    if a.scored:
        s = pd.read_pickle(a.scored) if a.scored.endswith(".pkl") else pd.read_parquet(a.scored)
        cand = s[s["trip_id"].astype(str) == trip.trip_id].reset_index(drop=True)
        prob = cand["prob"].to_numpy()
    else:
        cand, prob = build_frame(trip), None
        params["alpha"] = 0.0
    lat = build_lattice(cand, trip, RoadGraph(trip), gap_s=params["gap_s"])
    hmm, _ = decode_lattice(lat, prob, params)
    m = make_route_map(trip, {"Classifier-only": argmax_only(lat, prob), "HMM / Viterbi": hmm},
                       reference=trip.route)
    out = a.out or os.path.join("viz", "maps", f"route_{trip.trip_id}.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    m.save(out)
    print("saved", out)


if __name__ == "__main__":
    main()
