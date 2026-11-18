"""Folium EDA map: GPS observations, nearby road network and reference route.

Usage (needs `pip install folium`):
  python eda/folium_eda_map.py --part train --trips 00000048 00000002
  python eda/folium_eda_map.py --part student_test            # all trips
Output: results/eda/maps/<part>_<trip>.html
Only train / student_test are mapped (evaluator data is never touched).
"""
import argparse
import os
import sys

import folium

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_processing.io_utils import load_trip  # noqa: E402
from features.candidates import relevant_arcs  # noqa: E402

LEGEND = """
<div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999; background: white;
 padding: 10px 14px; border: 1px solid #999; border-radius: 6px; font-size: 13px;">
<b>PathSnap EDA – {part} trip {tid}</b><br>
<span style="color:#9e9e9e">&#9644;</span> Road network (within 150 m)<br>
<span style="color:#1aa64b">&#9644;</span> Reference route<br>
<span style="color:#2b7bff">&#9679;</span> GPS observations<br>
<span style="color:#d62728">&#9679;</span> Points after a gap &gt; 10 s
</div>"""


def seg(tr, a):
    u, v = tr.arcs[a]
    return [[float(tr.nodes[u, 1]), float(tr.nodes[u, 0])], [float(tr.nodes[v, 1]), float(tr.nodes[v, 0])]]


def make_map(trip_dir, out_path, part):
    tr = load_trip(trip_dir)
    lat, lon, t = tr.track[:, 1], tr.track[:, 0], tr.track[:, 2]
    m = folium.Map(location=[float(lat.mean()), float(lon.mean())], zoom_start=14, tiles="cartodbpositron")

    net = folium.FeatureGroup(name="Road network", show=True)
    for a in relevant_arcs(tr.gps_xy, tr.node_xy, tr.arcs, 150):
        folium.PolyLine(seg(tr, a), color="#9e9e9e", weight=1.5, opacity=0.6).add_to(net)
    net.add_to(m)

    if tr.route is not None:
        ref = folium.FeatureGroup(name="Reference route", show=True)
        for a in tr.route:
            folium.PolyLine(seg(tr, a), color="#1aa64b", weight=6, opacity=0.65,
                            tooltip=f"arc {int(a)}").add_to(ref)
        ref.add_to(m)

    gps = folium.FeatureGroup(name="GPS observations", show=True)
    step = max(1, len(lat) // 3000)
    for i in range(0, len(lat), step):
        folium.CircleMarker([float(lat[i]), float(lon[i])], radius=2, color="#2b7bff", fill=True,
                            fill_opacity=0.8, weight=0, tooltip=f"t={t[i]:.0f}s").add_to(gps)
    gps.add_to(m)

    gaps = folium.FeatureGroup(name="GPS gaps > 10 s", show=True)
    for i in range(1, len(t)):
        if t[i] - t[i - 1] > 10:
            folium.CircleMarker([float(lat[i]), float(lon[i])], radius=5, color="#d62728", fill=True,
                                tooltip=f"gap {t[i] - t[i - 1]:.0f}s").add_to(gaps)
    gaps.add_to(m)

    folium.Marker([float(lat[0]), float(lon[0])], tooltip="start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker([float(lat[-1]), float(lon[-1])], tooltip="end", icon=folium.Icon(color="red")).add_to(m)
    m.get_root().html.add_child(folium.Element(LEGEND.format(part=part, tid=tr.trip_id)))
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[float(lat.min()), float(lon.min())], [float(lat.max()), float(lon.max())]])
    m.save(out_path)
    print("saved", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--part", default="train", choices=["train", "student_test"])
    ap.add_argument("--trips", nargs="*", default=None)
    args = ap.parse_args()
    out = os.path.join("results", "eda", "maps")
    os.makedirs(out, exist_ok=True)
    base = os.path.join(args.data, args.part)
    for tid in args.trips or sorted(os.listdir(base)):
        make_map(os.path.join(base, tid), os.path.join(out, f"{args.part}_{tid}.html"), args.part)


if __name__ == "__main__":
    main()
