"""
Thin proxy sitting between the web-app frontend and Valhalla.

For every incoming /route request, it merges in the precomputed
`linear_cost_factors` (built by build_safety_factors.py from your ML
model's output) and forwards the request to the real Valhalla server.
Valhalla's own pathfinding then does the safety-aware routing - this proxy
does no scoring or reranking itself, it just injects the penalties.

Run standalone for local testing:
    uvicorn app:app --reload --port 5000
"""
import json
import os
from pathlib import Path
from typing import List, Tuple

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

import safety_score
import lighting_score
import recommend
from polyline6 import decode

VALHALLA_URL = os.environ.get("VALHALLA_URL", "http://localhost:8002")
FACTORS_PATH = Path(__file__).parent / "safety_factors.json"

# How much to pad the request's own bounding box before filtering factors,
# to allow for realistic detours rather than just the straight line between
# origin and destination. Two components:
#  - a fixed minimum pad (degrees) so short trips still get a sensible margin
#  - a proportional pad, relative to the request's own bounding box size, so
#    long trips get a wider margin too
MIN_PAD_DEG = 0.01       # ~1.1 km at London's latitude
PAD_FACTOR = 0.5         # + 50% of the request's own bbox span, each side

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local dev only - tighten this if you ever deploy this publicly
    allow_methods=["*"],
    allow_headers=["*"],
)

safety_score.build_grid()
print(f"Safety scoring grid built: {len(safety_score._grid)} cells")

lighting_score.build_grid()
print(f"Lighting scoring grid built: {len(lighting_score._grid)} cells")

# Each entry: (min_lat, max_lat, min_lon, max_lon, original_factor_dict).
# Decoding every shape's bounding box happens once, here, at startup - not
# on every request - so per-request filtering below is just cheap
# arithmetic over precomputed numbers.
_factor_index: List[Tuple[float, float, float, float, dict]] = []

if FACTORS_PATH.exists():
    with open(FACTORS_PATH) as f:
        raw_factors = json.load(f)
    for entry in raw_factors:
        points = decode(entry["shape"])
        if not points:
            continue
        lats = [p[0] for p in points]
        lons = [p[1] for p in points]
        _factor_index.append((min(lats), max(lats), min(lons), max(lons), entry))
    print(f"Indexed {len(_factor_index)} safety factor entries for bounding-box filtering")
else:
    print(
        f"WARNING: {FACTORS_PATH} not found - routes will NOT be safety-weighted. "
        "Run build_safety_factors.py first."
    )


def _request_bbox(locations: list) -> Tuple[float, float, float, float]:
    """Padded bounding box around a request's origin/destination(s)."""
    lats = [loc["lat"] for loc in locations]
    lons = [loc["lon"] for loc in locations]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    lat_pad = max(MIN_PAD_DEG, (max_lat - min_lat) * PAD_FACTOR)
    lon_pad = max(MIN_PAD_DEG, (max_lon - min_lon) * PAD_FACTOR)

    return (min_lat - lat_pad, max_lat + lat_pad, min_lon - lon_pad, max_lon + lon_pad)


def _bboxes_overlap(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    a_min_lat, a_max_lat, a_min_lon, a_max_lon = a
    b_min_lat, b_max_lat, b_min_lon, b_max_lon = b
    return not (
        a_max_lat < b_min_lat or a_min_lat > b_max_lat or
        a_max_lon < b_min_lon or a_min_lon > b_max_lon
    )


def _inject_safety_factors(payload: dict) -> dict:
    if not _factor_index:
        return payload

    locations = payload.get("locations", [])
    if not locations:
        return payload

    req_bbox = _request_bbox(locations)
    relevant = [
        entry for (min_lat, max_lat, min_lon, max_lon, entry) in _factor_index
        if _bboxes_overlap(req_bbox, (min_lat, max_lat, min_lon, max_lon))
    ]

    if relevant:
        # If the incoming request already specifies its own
        # linear_cost_factors, don't silently clobber them - extend instead.
        existing = payload.get("linear_cost_factors", [])
        payload["linear_cost_factors"] = existing + relevant

    return payload


@app.api_route("/route", methods=["GET", "POST"])
async def route(request: Request):
    if request.method == "POST":
        payload = await request.json()
    else:
        # Valhalla's GET form is /route?json={...}
        raw = request.query_params.get("json")
        payload = json.loads(raw) if raw else {}

    payload = _inject_safety_factors(payload)

    async with httpx.AsyncClient(timeout=30.0) as client:
        upstream = await client.post(f"{VALHALLA_URL}/route", json=payload)

    if upstream.status_code == 200:
        try:
            body = upstream.json()
            if "trip" in body:
                safety_score.annotate_trip_summary(body["trip"])
                lighting_score.annotate_trip_lighting(body["trip"])
            for alt in body.get("alternates", []):
                if "trip" in alt:
                    safety_score.annotate_trip_summary(alt["trip"])
                    lighting_score.annotate_trip_lighting(alt["trip"])

            recommend.apply_recommendation(body)

            return Response(
                content=json.dumps(body),
                status_code=upstream.status_code,
                media_type="application/json",
            )
        except (ValueError, KeyError) as e:
            print(f"WARNING: could not annotate safety scores ({e}), returning unmodified response")

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
    )


@app.api_route("/{full_path:path}", methods=["GET", "POST"])
async def passthrough(request: Request, full_path: str):
    """Anything else (locate, status, etc) goes straight through unmodified."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        if request.method == "POST":
            body = await request.body()
            upstream = await client.post(
                f"{VALHALLA_URL}/{full_path}", content=body,
                params=request.query_params,
            )
        else:
            upstream = await client.get(
                f"{VALHALLA_URL}/{full_path}", params=request.query_params,
            )

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
    )
