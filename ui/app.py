"""Streamlit UI intended to run on Laptop 2."""
from __future__ import annotations

import os

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(page_title="PathSnap", page_icon="🛣️", layout="wide")
st.title("PathSnap route reconstruction")
st.caption("Select a trip, send it to Laptop 1, and inspect the returned route.")

api_url = st.sidebar.text_input("Laptop 1 API URL", value=os.getenv("PATHSNAP_API_URL", "http://127.0.0.1:8000")).rstrip("/")


def get_json(path: str):
    response = requests.get(f"{api_url}{path}", timeout=10)
    response.raise_for_status()
    return response.json()


try:
    available = get_json("/trips")["trips"]
except requests.RequestException as exc:
    st.error(f"Cannot reach Laptop 1 at {api_url}. Check the address, Wi-Fi/hotspot, server, and firewall. ({exc})")
    st.stop()

trip_id = st.selectbox("Trip", available)
if st.button("Run prediction", type="primary"):
    try:
        with st.spinner("Laptop 1 is reconstructing the route..."):
            response = requests.post(f"{api_url}/predict", json={"trip_id": trip_id}, timeout=120)
            response.raise_for_status()
            st.session_state["prediction"] = response.json()
    except requests.RequestException as exc:
        st.error(f"Prediction request failed: {exc}")

result = st.session_state.get("prediction")
if result:
    metrics = result["metrics"]
    columns = st.columns(4)
    columns[0].metric("GPS points", metrics["n_points"])
    columns[1].metric("Top-1 accuracy", "N/A" if metrics["top1_accuracy"] is None else f"{metrics['top1_accuracy']:.1%}")
    columns[2].metric("Route continuity", "N/A" if metrics["route_continuity"] is None else f"{metrics['route_continuity']:.1%}")
    columns[3].metric("Server inference", f"{metrics['inference_ms']:.1f} ms")

    gps = result["gps"]
    m = folium.Map(location=gps[len(gps) // 2], zoom_start=14, control_scale=True)
    gps_layer = folium.FeatureGroup(name="Raw GPS observations", show=True)
    for point in gps:
        folium.CircleMarker(point, radius=2, color="#2563eb", fill=True, fill_opacity=0.65).add_to(gps_layer)
    gps_layer.add_to(m)
    route_layer = folium.FeatureGroup(name="Predicted route", show=True)
    for segment in result["predicted_route"]:
        folium.PolyLine(segment, color="#16a34a", weight=4, opacity=0.85).add_to(route_layer)
    route_layer.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds(gps)
    st_folium(m, height=600, use_container_width=True)
