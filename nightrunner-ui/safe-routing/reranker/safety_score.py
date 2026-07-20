"""
Computes an interpretable safety metric for a route, by checking how much
of its length passes close to edges your ML model flagged as risky.

Approach: bucket scored-edge midpoints into a coarse grid (cheap spatial
index, no extra dependencies), then walk each route's decoded shape in
consecutive segments, checking each segment's midpoint against nearby grid
cells. A segment counts as "near a risky edge" if within RISKY_RADIUS_M of
any scored-edge midpoint.

Output per route:
  - safety_score: 0-100, 100 = route never goes near a flagged edge
  - pct_length_near_risky: % of route length spent near flagged edges
  - worst_tag_encountered: highest safety_tag_value (1-11) the route
    passes near, or None if it avoids all flagged edges entirely
"""
import math
from typing import List, Optional, Tuple

from build_safety_factors import load_scored_edges
from polyline6 import decode

RISKY_RADIUS_M = 20.0  # how close a route must pass to count as "near" a flagged edge
GRID_SIZE_DEG = 0.001  # ~110m grid cells at London's latitude

_grid: dict = {}


def _cell(lat: float, lon: float) -> Tuple[int, int]:
    return (round(lat / GRID_SIZE_DEG), round(lon / GRID_SIZE_DEG))


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def build_grid() -> None:
    """Call once at startup. Populates the module-level spatial index."""
    global _grid
    _grid = {}
    for coords, tag, _osmid in load_scored_edges():
        if tag <= 1:
            continue
        mid_lat = sum(c[0] for c in coords) / len(coords)
        mid_lon = sum(c[1] for c in coords) / len(coords)
        _grid.setdefault(_cell(mid_lat, mid_lon), []).append((mid_lat, mid_lon, tag))


def _nearest_tag(lat: float, lon: float) -> Optional[int]:
    """Returns the tag of the nearest scored edge within RISKY_RADIUS_M, or None."""
    cy, cx = _cell(lat, lon)
    best_tag = None
    best_dist = RISKY_RADIUS_M
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            for (elat, elon, tag) in _grid.get((cy + dy, cx + dx), []):
                d = _haversine_m(lat, lon, elat, elon)
                if d <= best_dist:
                    best_dist = d
                    best_tag = tag
    return best_tag


def score_shape(encoded_shape: str) -> dict:
    """Computes the safety metric for one route's encoded polyline6 shape."""
    points = decode(encoded_shape)
    if len(points) < 2:
        return {"safety_score": 100.0, "pct_length_near_risky": 0.0, "worst_tag_encountered": None}

    total_length = 0.0
    risky_length = 0.0
    worst_tag = None

    for i in range(len(points) - 1):
        lat1, lon1 = points[i]
        lat2, lon2 = points[i + 1]
        seg_len = _haversine_m(lat1, lon1, lat2, lon2)
        total_length += seg_len

        mid_lat, mid_lon = (lat1 + lat2) / 2, (lon1 + lon2) / 2
        tag = _nearest_tag(mid_lat, mid_lon)
        if tag is not None:
            risky_length += seg_len
            if worst_tag is None or tag > worst_tag:
                worst_tag = tag

    if total_length == 0:
        pct = 0.0
    else:
        pct = 100.0 * risky_length / total_length

    return {
        "safety_score": round(max(0.0, 100.0 - pct), 1),
        "pct_length_near_risky": round(pct, 1),
        "worst_tag_encountered": worst_tag,
    }


def annotate_trip_summary(trip: dict) -> None:
    """Mutates a Valhalla trip object in place, adding safety fields to its summary."""
    legs = trip.get("legs", [])
    full_shape_points: List[Tuple[float, float]] = []
    for leg in legs:
        full_shape_points.extend(decode(leg["shape"]))

    if not full_shape_points:
        return

    # Re-encode isn't needed - score_shape only needs decoded points, so
    # reuse the per-leg decode directly rather than round-tripping.
    total_length = 0.0
    risky_length = 0.0
    worst_tag = None
    for i in range(len(full_shape_points) - 1):
        lat1, lon1 = full_shape_points[i]
        lat2, lon2 = full_shape_points[i + 1]
        seg_len = _haversine_m(lat1, lon1, lat2, lon2)
        total_length += seg_len
        mid_lat, mid_lon = (lat1 + lat2) / 2, (lon1 + lon2) / 2
        tag = _nearest_tag(mid_lat, mid_lon)
        if tag is not None:
            risky_length += seg_len
            if worst_tag is None or tag > worst_tag:
                worst_tag = tag

    pct = 0.0 if total_length == 0 else 100.0 * risky_length / total_length
    trip.setdefault("summary", {})
    trip["summary"]["safety_score"] = round(max(0.0, 100.0 - pct), 1)
    trip["summary"]["pct_length_near_risky"] = round(pct, 1)
    trip["summary"]["worst_tag_encountered"] = worst_tag
