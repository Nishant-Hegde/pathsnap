"""Evaluator-facing FastAPI server, intended to run on Laptop 1."""
from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.schemas import Metrics, PredictRequest, PredictResponse
from api.scoring import score_prediction
from data_processing.io_utils import Trip, list_trips, load_trip
from pipeline_stub import predict_route

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("pathsnap.api")

DATA_ROOT = Path(os.getenv("PATHSNAP_DATA", ROOT / "data"))
SERVING_PARTITION = os.getenv("PATHSNAP_SERVING_PARTITION", "student_test")
TRIPS: dict[str, Trip] = {}


def _load_serving_trips() -> dict[str, Trip]:
    partition_dir = DATA_ROOT / SERVING_PARTITION
    if not partition_dir.is_dir():
        raise RuntimeError(f"Data partition is missing: {partition_dir}")
    return {trip.trip_id: trip for path in list_trips(str(partition_dir)) for trip in [load_trip(path)]}


@asynccontextmanager
async def lifespan(_: FastAPI):
    global TRIPS
    TRIPS = _load_serving_trips()
    LOGGER.info("Loaded %d trips from %s", len(TRIPS), DATA_ROOT / SERVING_PARTITION)
    yield
    TRIPS.clear()


app = FastAPI(title="PathSnap API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # laptops share a trusted local network during the hackathon
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "partition": SERVING_PARTITION, "loaded_trips": len(TRIPS)}


@app.get("/trips")
def trips() -> dict[str, list[str]]:
    return {"trips": sorted(TRIPS)}


def _prediction_trip(trip: Trip) -> Trip:
    """Return an inference-only copy. The predictor has no reference route."""
    return replace(trip, route=None)


def _route_geometry(trip: Trip, arc_ids) -> list[list[list[float]]]:
    result: list[list[list[float]]] = []
    last_arc: int | None = None
    for arc_id in arc_ids:
        arc_id = int(arc_id)
        if arc_id == last_arc:
            continue
        start, end = trip.arcs[arc_id]
        start_lon, start_lat = trip.nodes[start]
        end_lon, end_lat = trip.nodes[end]
        result.append([[float(start_lat), float(start_lon)], [float(end_lat), float(end_lon)]])
        last_arc = arc_id
    return result


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    trip = TRIPS.get(payload.trip_id)
    if trip is None:
        raise HTTPException(status_code=404, detail=f"Unknown trip_id: {payload.trip_id}")

    started = time.perf_counter()
    try:
        predicted = predict_route(_prediction_trip(trip))
        metrics = score_prediction(predicted, trip)
    except Exception as exc:
        LOGGER.exception("Prediction failed for trip_id=%s", payload.trip_id)
        raise HTTPException(status_code=500, detail="Prediction failed on the server.") from exc

    elapsed_ms = (time.perf_counter() - started) * 1_000
    client = request.client.host if request.client else "unknown"
    LOGGER.info(
        "predict client=%s trip_id=%s points=%d inference_ms=%.1f top1=%s continuity=%s",
        client, payload.trip_id, len(trip.track), elapsed_ms,
        metrics["top1_accuracy"], metrics["route_continuity"],
    )
    gps = [[float(lat), float(lon)] for lon, lat in trip.track[:, :2]]
    arc_ids = predicted.sort_values("obs_idx")["arc_id"].to_numpy()
    return PredictResponse(
        trip_id=payload.trip_id,
        gps=gps,
        predicted_route=_route_geometry(trip, arc_ids),
        metrics=Metrics(n_points=len(gps), inference_ms=round(elapsed_ms, 2), **metrics),
    )
