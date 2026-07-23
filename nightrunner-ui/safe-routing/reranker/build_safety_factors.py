"""
Builds `safety_factors.json`: a precomputed array that the reranker proxy
combines into `linear_cost_factors` for every /route request sent to
Valhalla - both the safety penalty AND the lighting penalty now live here,
combined per edge (previously lighting was display-only; see
lighting_factor() / load_edges_for_routing_cost() below).

Joins three files on `osmid` (OSM way ID):
  - edge_features.csv       - has u_lat/u_lng/v_lat/v_lng (the two endpoint
                               coordinates of each edge) and osmid
  - osm_safety_tags.csv     - has osmid and safety_tag_value, an ordinal RISK
                               tier (1 = baseline/low-risk, 11 = highest-risk -
                               confirmed by correlating it against crime_per_km
                               in edge_features.csv: higher tag = more crime,
                               NOT more safety, despite the column name). Where
                               an osmid has more than one safety_tag_value
                               across rows, the MAXIMUM is used (worst-case).
  - edge_lamp_features.csv  - has osmid and lamp_per_km; averaged per osmid
                               (see lighting_score.load_lamp_status_by_osmid),
                               a way counts as "lit" if that average is > 0.

An osmid with NO entry in osm_safety_tags.csv, or NO entry in
edge_lamp_features.csv, is assumed SAFE by default (baseline safety tag /
lit) rather than being skipped - missing data should never itself look
risky. This differs from load_scored_edges() below, which still skips
untagged edges entirely - that function backs the descriptive UI score and
the safety geojson viz, and its behaviour is left unchanged.

Put all three CSVs in this same folder before running:
    python build_safety_factors.py
"""
import csv
import json
import math
from typing import Iterable, List, Tuple

import lighting_score
from polyline6 import encode

EDGE_FEATURES_PATH = "edge_features.csv"
SAFETY_TAGS_PATH = "osm_safety_tags.csv"

# Valhalla's edge-walk matching (which snaps a submitted shape onto its
# graph) can fail outright on very short or zero-length shapes - a single
# bad entry causes the ENTIRE /route request to fail with a 400, not just
# that one entry being skipped. Filtering these out here, at build time,
# is more reliable than debugging failures after the fact.
MIN_SHAPE_LENGTH_M = 2.0


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

# --- Tune these ---------------------------------------------------------
# Maps the raw 1-11 risk tier to a Valhalla cost factor. SAFETY_SCALE
# controls how strongly higher-risk edges get penalised relative to the
# tag=1 baseline (which gets factor 1.0, i.e. no penalty - this matches
# ~99% of edges). Start conservative and increase once you see how routes
# actually react - very large factors on many edges can produce oddly
# circuitous routes, or in the extreme, no route at all if every path out
# of an area is penalised.
SAFETY_SCALE = 3.0

# Flat penalty multiplier applied to unlit edges (binary lit/unlit, matching
# lighting_score.py's existing descriptive-score convention - not graded by
# lamp density). Lit edges get factor 1.0 (no penalty).
LIGHTING_SCALE = 2.0


def score_to_factor(safety_tag_value: int) -> float:
    return 1.0 + SAFETY_SCALE * max(0, safety_tag_value - 1)


def lighting_factor(is_lit: bool) -> float:
    return 1.0 if is_lit else 1.0 + LIGHTING_SCALE


def load_scored_edges() -> Iterable[Tuple[List[Tuple[float, float]], int, str]]:
    """
    Joins edge_features.csv and osm_safety_tags.csv on osmid (OSM way ID) -
    NOT edge_id. A single osmid can appear on multiple graph edges (e.g. a
    way split into several segments, or a bidirectional pair), and the
    safety tags file can carry more than one safety_tag_value for the same
    osmid (~15% of osmids in practice). Where that happens, the MAXIMUM
    tag value for that osmid is used (most conservative / worst-case).

    Yields (coords, safety_tag_value, osmid) triples, where coords is the
    [(u_lat, u_lng), (v_lat, v_lng)] two-point line for that edge. Edges
    shorter than MIN_SHAPE_LENGTH_M are skipped entirely - Valhalla's
    edge-walk matching can reject these outright and fail the whole
    request, not just that one entry.
    """
    max_tag_by_osmid: dict = {}
    with open(SAFETY_TAGS_PATH, newline="") as f:
        for row in csv.DictReader(f):
            osmid = row["osmid"]
            tag = int(row["safety_tag_value"])
            if osmid not in max_tag_by_osmid or tag > max_tag_by_osmid[osmid]:
                max_tag_by_osmid[osmid] = tag

    with open(EDGE_FEATURES_PATH, newline="") as f:
        for row in csv.DictReader(f):
            osmid = row["osmid"]
            tag = max_tag_by_osmid.get(osmid)
            if tag is None:
                continue  # no safety tag for this osmid - skip rather than guess
            try:
                u_lat, u_lng = float(row["u_lat"]), float(row["u_lng"])
                v_lat, v_lng = float(row["v_lat"]), float(row["v_lng"])
            except (KeyError, ValueError):
                continue  # missing/malformed coordinates - skip
            if _haversine_m(u_lat, u_lng, v_lat, v_lng) < MIN_SHAPE_LENGTH_M:
                continue  # too short/degenerate for reliable edge-walk matching
            yield [(u_lat, u_lng), (v_lat, v_lng)], tag, osmid


def load_edges_for_routing_cost() -> Iterable[Tuple[List[Tuple[float, float]], int, bool, str]]:
    """
    Like load_scored_edges() above, but for building the actual Valhalla
    routing cost rather than the descriptive UI score / geojson viz:

      - An osmid missing from osm_safety_tags.csv is assumed SAFE by
        default (tag=1, baseline, no safety penalty) instead of being
        skipped - every edge should be considered for routing cost, not
        just the ones a safety tag happens to cover.
      - An osmid missing from edge_lamp_features.csv (i.e. absent from
        lighting_score.load_lamp_status_by_osmid()'s result) is assumed
        LIT by default (no lighting penalty), for the same reason.

    Yields (coords, safety_tag_value, is_lit, osmid) for every edge with
    valid, non-degenerate coordinates.
    """
    max_tag_by_osmid: dict = {}
    with open(SAFETY_TAGS_PATH, newline="") as f:
        for row in csv.DictReader(f):
            osmid = row["osmid"]
            tag = int(row["safety_tag_value"])
            if osmid not in max_tag_by_osmid or tag > max_tag_by_osmid[osmid]:
                max_tag_by_osmid[osmid] = tag

    lamp_status = lighting_score.load_lamp_status_by_osmid()

    with open(EDGE_FEATURES_PATH, newline="") as f:
        for row in csv.DictReader(f):
            osmid = row["osmid"]
            tag = max_tag_by_osmid.get(osmid, 1)      # missing -> assume safe/baseline
            is_lit = lamp_status.get(osmid, True)     # missing -> assume lit
            try:
                u_lat, u_lng = float(row["u_lat"]), float(row["u_lng"])
                v_lat, v_lng = float(row["v_lat"]), float(row["v_lng"])
            except (KeyError, ValueError):
                continue  # missing/malformed coordinates - skip
            if _haversine_m(u_lat, u_lng, v_lat, v_lng) < MIN_SHAPE_LENGTH_M:
                continue  # too short/degenerate for reliable edge-walk matching
            yield [(u_lat, u_lng), (v_lat, v_lng)], tag, is_lit, osmid


def build() -> None:
    """
    Writes safety_factors.json as a list of {"shape", "safety_factor",
    "lighting_factor"} entries (NOT a single combined "factor" - app.py
    combines the two at request time, since the is_daylight toggle needs
    to be able to skip the lighting factor per-request without a rebuild).
    """
    factors = []
    skipped_baseline = 0
    for coords, tag, is_lit, _osmid in load_edges_for_routing_cost():
        s_factor = score_to_factor(tag)
        l_factor = lighting_factor(is_lit)
        if s_factor == 1.0 and l_factor == 1.0:
            # Baseline safety AND lit - no penalty either way. Omitting
            # these keeps the request payload small instead of sending
            # tens of thousands of no-op entries every time.
            skipped_baseline += 1
            continue
        factors.append({
            "shape": encode(coords),
            "safety_factor": round(s_factor, 3),
            "lighting_factor": round(l_factor, 3),
        })

    with open("safety_factors.json", "w") as f:
        json.dump(factors, f)

    print(
        f"Wrote {len(factors)} routing-cost entries to safety_factors.json "
        f"(skipped {skipped_baseline} baseline/no-penalty edges)"
    )


if __name__ == "__main__":
    build()
