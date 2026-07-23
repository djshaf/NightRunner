"""
Computes an interpretable lighting metric for a route: what % of its
length passes near a street tagged as lit, based on edge_lamp_features.csv.

This is purely descriptive - unlike safety, it has NO effect on routing
(never sent to Valhalla as a cost factor). That also means it doesn't need
the short-edge filtering safety_factors.py needs (that was specifically to
avoid Valhalla's edge-walk matching choking on degenerate shapes; pure
Python proximity scoring here tolerates a zero-length edge fine - it just
contributes a valid grid cell like any other).

edge_lamp_features.csv has no lat/lng of its own - it's joined against
edge_features.csv via osmid to get coordinates, same pattern used for the
safety data. Where one osmid has multiple different lamp_per_km values
across its rows (about 20% of osmids, per inspection), they're AVERAGED
(per the project owner's decision), and a way counts as "lit" if that
average is > 0.
"""
import csv
import math
from typing import Iterable, List, Optional, Tuple

from polyline6 import decode

EDGE_FEATURES_PATH = "edge_features.csv"
LAMP_FEATURES_PATH = "edge_lamp_features.csv"

RISKY_RADIUS_M = 20.0   # same proximity radius used for safety scoring, for consistency
GRID_SIZE_DEG = 0.001   # ~110m grid cells at London's latitude

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


def load_lamp_status_by_osmid() -> dict:
    """
    Returns {osmid: is_lit} for every osmid that has at least one row in
    edge_lamp_features.csv, averaging lamp_per_km across rows per osmid
    (per the project owner's decision) and treating a way as "lit" if that
    average is > 0.

    Osmids with NO rows at all are simply absent from this dict - they are
    not "unlit", they're "no data". Callers that want an "assume lit by
    default" convention (e.g. build_safety_factors.py's routing-cost build,
    matching how missing safety tags default to baseline/safe) should use
    `lamp_status.get(osmid, True)` rather than treating a missing key as
    unlit.
    """
    avg_lamp_by_osmid: dict = {}
    count_by_osmid: dict = {}
    with open(LAMP_FEATURES_PATH, newline="") as f:
        for row in csv.DictReader(f):
            osmid = row["osmid"]
            lamp_per_km = float(row["lamp_per_km"])
            avg_lamp_by_osmid[osmid] = avg_lamp_by_osmid.get(osmid, 0.0) + lamp_per_km
            count_by_osmid[osmid] = count_by_osmid.get(osmid, 0) + 1

    return {
        osmid: (total / count_by_osmid[osmid]) > 0
        for osmid, total in avg_lamp_by_osmid.items()
    }


def load_lit_edges() -> Iterable[Tuple[List[Tuple[float, float]], bool]]:
    """
    Joins edge_features.csv and edge_lamp_features.csv on osmid. Yields
    (coords, is_lit) pairs for every LIT edge only (unlit edges aren't
    needed - the scoring logic below only cares about proximity to lit
    streets, not unlit ones).
    """
    lamp_status = load_lamp_status_by_osmid()
    lit_osmids = {osmid for osmid, is_lit in lamp_status.items() if is_lit}

    with open(EDGE_FEATURES_PATH, newline="") as f:
        for row in csv.DictReader(f):
            osmid = row["osmid"]
            if osmid not in lit_osmids:
                continue
            try:
                coords = [
                    (float(row["u_lat"]), float(row["u_lng"])),
                    (float(row["v_lat"]), float(row["v_lng"])),
                ]
            except (KeyError, ValueError):
                continue
            yield coords, True


def load_unlit_edges() -> Iterable[Tuple[List[Tuple[float, float]], str]]:
    """
    Like load_lit_edges(), but the inverse: yields (coords, osmid) for
    every osmid with EXPLICIT unlit evidence (a row in
    edge_lamp_features.csv whose average lamp_per_km is <= 0) - not merely
    an osmid that's absent from the file. Absence means "no data", which
    is a different thing from "confirmed unlit"; this function only
    reports the latter, since it's meant for visualizing what the data
    actually asserts, not for the "assume lit by default" convention used
    when building the routing cost (see build_safety_factors.py).

    Used by build_safety_geojson.py to render a "confirmed unlit" layer
    alongside the safety-tag layer.
    """
    lamp_status = load_lamp_status_by_osmid()
    unlit_osmids = {osmid for osmid, is_lit in lamp_status.items() if not is_lit}

    with open(EDGE_FEATURES_PATH, newline="") as f:
        for row in csv.DictReader(f):
            osmid = row["osmid"]
            if osmid not in unlit_osmids:
                continue
            try:
                coords = [
                    (float(row["u_lat"]), float(row["u_lng"])),
                    (float(row["v_lat"]), float(row["v_lng"])),
                ]
            except (KeyError, ValueError):
                continue
            yield coords, osmid


def build_grid() -> None:
    """Call once at startup. Populates the module-level spatial index of lit edges."""
    global _grid
    _grid = {}
    for coords, _is_lit in load_lit_edges():
        mid_lat = sum(c[0] for c in coords) / len(coords)
        mid_lon = sum(c[1] for c in coords) / len(coords)
        _grid.setdefault(_cell(mid_lat, mid_lon), []).append((mid_lat, mid_lon))


def _near_lit_edge(lat: float, lon: float) -> bool:
    """True if within RISKY_RADIUS_M of any lit edge's midpoint."""
    cy, cx = _cell(lat, lon)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            for (elat, elon) in _grid.get((cy + dy, cx + dx), []):
                if _haversine_m(lat, lon, elat, elon) <= RISKY_RADIUS_M:
                    return True
    return False


def annotate_trip_lighting(trip: dict) -> None:
    """
    Mutates a Valhalla trip object in place, adding a lighting_score field
    to its summary: 0-100, where 100 means the route passes near a lit
    street for its entire length.
    """
    legs = trip.get("legs", [])
    points: List[Tuple[float, float]] = []
    for leg in legs:
        points.extend(decode(leg["shape"]))

    if len(points) < 2:
        return

    total_length = 0.0
    lit_length = 0.0
    for i in range(len(points) - 1):
        lat1, lon1 = points[i]
        lat2, lon2 = points[i + 1]
        seg_len = _haversine_m(lat1, lon1, lat2, lon2)
        total_length += seg_len
        mid_lat, mid_lon = (lat1 + lat2) / 2, (lon1 + lon2) / 2
        if _near_lit_edge(mid_lat, mid_lon):
            lit_length += seg_len

    pct_lit = 0.0 if total_length == 0 else 100.0 * lit_length / total_length
    trip.setdefault("summary", {})
    trip["summary"]["lighting_score"] = round(pct_lit, 1)
