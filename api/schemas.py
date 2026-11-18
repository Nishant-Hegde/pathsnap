"""Request and response models exposed by the PathSnap API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    trip_id: str = Field(min_length=1, description="Identifier returned by GET /trips")


class Metrics(BaseModel):
    top1_accuracy: float | None = None
    route_continuity: float | None = None
    n_points: int
    inference_ms: float


class PredictResponse(BaseModel):
    trip_id: str
    gps: list[list[float]]
    predicted_route: list[list[list[float]]]
    metrics: Metrics
