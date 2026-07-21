"""
Data collection and cleaning pipeline
ECS7036P Group 8: AI approach to running route optimisation for safety

Role: Data Lead (collection, cleaning, preparation)

What this script does, end to end:
  1. Pulls a routable walking/running street graph for the study area from
     OpenStreetMap via OSMnx, and turns it into an edge table.
  2. Pulls street level recorded crime for the same area from the UK Police
     open data API, month by month.
  3. Pulls street level stop and search records from the same API.
  4. Pulls Camden street lighting point data from the borough open data portal.
  5. Cleans all of the above (missing coordinates, duplicates, coordinate
     systems).
  6. Snaps crime, stop and search, and lamp points onto the nearest street
     edge, then builds TWO separate edge level feature tables:
       - edge_features.csv/.gpkg: crime + stop and search features. TWO
         different views of the same data sit side by side, not one
         replacing the other:
           (a) nearest-edge counts: each crime/stop-search snapped to its
               SINGLE closest street, summed as an overall total, split out
               per month, and spatially lagged to neighbouring streets.
           (b) network KDE (NetKDE): the ML-facing smoothing. Distance is
               measured ALONG the walkable street network (Dijkstra shortest
               path), not in a straight line, so a crime diffuses down
               actual streets rather than through walls, back gardens or
               across the canal. Crucially it is applied SEPARATELY TO EACH
               MONTH, so no future data leaks backwards. These are the
               {metric}_netkde_YYYY_MM columns and are the ones intended for
               the model.
       - edge_lamp_features.csv/.gpkg: lamp features (lamp_count,
         lamp_per_km, is_lit), kept separate, lighting is a live snapshot
         with no time dimension, a different kind of feature entirely.
  7. Saves everything to disk (GeoPackage plus plain CSVs of both feature
     tables). Both feature tables include the true OpenStreetMap way ID
     (osmid) plus u/v node IDs and their WGS84 coordinates, so every row
     can be traced back to an actual place on OpenStreetMap, not just an
     internal edge_id.

Why NetKDE, and why per month (this was a review point from the ML lead):
  A straight-line, whole-date-range radius view would smooth through
  anything within a fixed metric distance, including places you cannot
  actually walk to, and would smooth ALL months together. NetKDE fixes both:
  distance is measured along the real street graph, and smoothing is done
  separately for each month, so if we smoothed a pooled multi-year total, a
  crime from 2026 would bleed into a street's 2023 features and leak the
  future into the past; doing it inside each month keeps every month
  self-contained. The heavy lifting mirrors the ML lead's own NetKDE
  function, moved upstream into the data pipeline so it happens once here
  and does not need repeating later (and so the graph is built once, not
  rebuilt for every single month).

The data sources match the proposal Resources table:
  street network    -> OpenStreetMap (OSMnx)
  crime rates       -> data.police.uk
  stop and search   -> data.police.uk
  street lighting   -> Camden borough open data
  day / night       -> Sunrise-Sunset API (optional helper at the bottom)

Notes on data.police.uk, checked directly against their API docs and their
"About" page (which documents the anonymisation methodology in detail):
  - Crime type: available, the "category" field. 14 real categories, see
    CRIME_SEVERITY / CRIME_PERCEIVED_RISK below for the full list.
  - Crime grade / seriousness: NOT provided by the API. Two DIFFERENT
    weightings are derived below, see the long comment above CRIME_SEVERITY
    for why there are two, not one.
  - Crime by day / exact date: NOT available for crimes, only "month".
  - Stop and search DOES carry a real datetime.
  - Date range: rolling ~3 year window, resolve_months() clips to whatever
    is actually published and prints a note if it had to.
  - Location anonymisation: crime/stop-search coordinates are NOT the
    actual incident location. Each force's raw location is snapped to the
    nearest point on a master list of ~680,000 anonymous map points, each
    map point's catchment must contain at least 8 postal addresses (or
    none at all, e.g. a park). Points are NOT on a uniform grid. Our own
    Camden run gives a rough empirical spacing estimate of ~125m (see
    README). This is exactly why the NetKDE bandwidth defaults to 125m: the
    underlying location data itself is only accurate to roughly that scale.

Run this on a machine with normal internet access. It needs to reach
overpass-api.de (OSMnx), data.police.uk and opendata.camden.gov.uk.

Install once:
  pip install osmnx geopandas shapely requests pandas pyproj networkx numpy

Author: <your name>, Group 8
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import requests
from shapely.geometry import Point, shape


# --------------------------------------------------------------------------
# CONFIG. Change these, nothing else needs editing for a first run.
# --------------------------------------------------------------------------

# The study area. Camden is used first because we have its lighting data.
# Any geocodable place with an OSM boundary works here.
PLACE = "London Borough of Camden, London, England"

# What we would LIKE to pull, in YYYY-MM. data.police.uk only keeps a
# rolling ~3 year window though, so this is a wishlist, not a guarantee.
# resolve_months() below clips it to whatever is actually published when
# the script runs, and prints a note if the start date had to move.
DESIRED_START = "2021-01"
DESIRED_END = "2026-06"

# "walk" gives a graph a runner can actually use: streets plus footpaths,
# ignoring motorways. It is the right base for this project.
NETWORK_TYPE = "walk"

# Camden street lighting, Socrata SODA JSON endpoint for dataset dfq3-8wzu.
# (jx8t-gxyu is the map visualisation wrapper and has no populated columns;
# dfq3-8wzu is the underlying table with real lat/lng fields.)
CAMDEN_LIGHTING_URL = "https://opendata.camden.gov.uk/resource/dfq3-8wzu.json"

# --------------------------------------------------------------------------
# NETWORK KDE (NetKDE) - the ML-facing smoothing.
#
# Instead of snapping each crime to only its single nearest street, this
# spreads it out to nearby streets along the real walkable street network
# (Dijkstra shortest-path distance, not a straight line), weighted down the
# further away it is. This is a simplified version of Kernel Density
# Estimation (KDE), the standard technique in crime hotspot mapping:
#   Chainey, S., Tompson, L. and Uhlig, S. (2008) 'The utility of hotspot
#   mapping for predicting spatial patterns of crime', Security Journal,
#   21(1-2), pp.4-28. KDE aggregates crime within a user-specified search
#   radius into a continuous density surface, and was found to consistently
#   outperform simpler hotspot mapping techniques. Using network distance
#   rather than a straight line means a crime cannot smooth through a wall,
#   a back garden or across the canal onto a street you cannot walk to.
#
# Bandwidth, in metres of NETWORK (walking) distance. 125m matches our own
# estimate of how far apart data.police.uk's anonymised crime points
# typically are in Camden (see README); change this single value if the
# team settles on a different one (the ML lead's reference function used
# 150m).
# --------------------------------------------------------------------------
NETKDE_BANDWIDTH_M = 125

# Metrics that get the per-month NetKDE treatment. These are exactly the
# four monthly totals, smoothed along the network inside each month.
NETKDE_METRICS = ["crime_count", "crime_severity_sum",
                  "crime_perceived_risk_sum", "stop_search_count"]

# Our own coarse severity tiering of data.police.uk's 14 crime categories.
# The API itself has no seriousness/grade field, so this is informed by two
# published sources rather than invented from scratch:
#   1. Sherman, L., Neyroud, P. and Neyroud, E. (2016) 'The Cambridge Crime
#      Harm Index: Measuring Total Harm From Crime Based On Sentencing
#      Guidelines', Policing: A Journal of Policy and Practice, 10(3),
#      pp.171-183. Weights crimes by custodial sentence starting point.
#   2. Office for National Statistics (2016) 'Research outputs: developing
#      a Crime Severity Score for England and Wales using data on crimes
#      recorded by the police'. Same idea, official sentencing-based
#      weights, published on ons.gov.uk.
#
# CRIME_PERCEIVED_RISK = how unsafe it makes an area FEEL to someone
# passing through, which is not the same thing. Informed by:
#   3. Innes, M. (2004) 'Signal crimes and signal disorders: notes on
#      deviance as communicative action', The British Journal of Sociology,
#      55(3), pp.335-355. Also Innes, M. and Fielding, N. (2002) 'From
#      Community to Communicative Policing: Signal Crimes and the Problem
#      of Public Reassurance', Sociological Research Online, 7(2).
#   4. Office for National Statistics, Crime Survey for England and Wales
#      (CSEW), 'perception and Anti-Social Behaviour (ASB) by Police Force
#      Area' releases.
#
# Neither list is a precise reproduction of the cited studies' exact
# numbers, that would need the ~200 individual offence codes those studies
# use, not the 14 broad categories police.uk exposes. Both are a coarse,
# citeable simplification, and should be described as such in the report.
# --------------------------------------------------------------------------

CRIME_SEVERITY = {
    # high: 3
    "violent-crime": 3,          # police.uk labels this "Violence and sexual offences"
    "robbery": 3,
    "possession-of-weapons": 3,
    # medium: 2
    "burglary": 2,
    "criminal-damage-arson": 2,
    "drugs": 2,
    "public-order": 2,
    "theft-from-the-person": 2,
    "vehicle-crime": 2,
    # low: 1
    "anti-social-behaviour": 1,
    "bicycle-theft": 1,
    "shoplifting": 1,
    "other-theft": 1,
    "other-crime": 1,
}
DEFAULT_SEVERITY = 1

CRIME_PERCEIVED_RISK = {
    # 4: direct personal threat, the kind of crime that could happen TO
    # someone out running, right now, in that spot
    "violent-crime": 4,
    "robbery": 4,
    "theft-from-the-person": 4,   # e.g. phone/bag snatching, very relevant to a runner
    "possession-of-weapons": 4,
    # 3: visible disorder / "signal" crimes, per Innes (2004), these mark
    # an area as unsafe-feeling even without a direct threat to the runner
    "anti-social-behaviour": 3,
    "criminal-damage-arson": 3,
    "public-order": 3,
    "drugs": 3,
    # 2: property crime with low direct visibility to a passer-by
    "burglary": 2,                # happens inside homes, rarely witnessed from the street
    "vehicle-crime": 2,
    "bicycle-theft": 2,
    # 1: lowest signal value to someone just passing through
    "shoplifting": 1,
    "other-theft": 1,
    "other-crime": 1,
}
DEFAULT_PERCEIVED_RISK = 1

# Metric coordinate system for Britain, so distances are in real metres.
METRIC_CRS = 27700          # British National Grid
WGS84 = 4326                # plain lat / lon

# A point further than this from any street is treated as unmatched and dropped.
CRIME_SNAP_MAX_M = 50       # crime locations are already street snapped by the police
STOP_SEARCH_SNAP_MAX_M = 50 # same anonymisation as crime, same threshold
LAMP_SNAP_MAX_M = 25        # lamps sit on or beside the kerb

# Where outputs go.
OUT_DIR = Path("data_out")
OUT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------------------------
# 1. STREET NETWORK
# --------------------------------------------------------------------------

def get_network(place: str, network_type: str):
    """Download the street graph and return (boundary_polygon, edges_gdf).

    edges_gdf is projected to METRIC_CRS and carries a stable 'edge_id' so we
    can attach features to each street segment later. Also kept:
      - osmid: the TRUE OpenStreetMap way ID for this street (converted to
        a comma-joined string if OSMnx merged several OSM ways into one
        edge, since a raw Python list can't be written to CSV/GeoPackage).
      - u_lat/u_lng and v_lat/v_lng: WGS84 (plain GPS) coordinates of each
        edge's two endpoint nodes, sitting right next to u and v, since u
        and v on their own are just OpenStreetMap's internal node IDs.

    NOTE: 'lit' is an OSM tag only present on edges where someone actually
    mapped it. If NO edge in the whole download has that tag (true for
    Camden), the column will not exist at all afterwards. Downstream code
    has to check `if "lit" in ...` rather than assume it is always there.
    """
    print(f"[network] downloading '{place}' ...")
    graph = ox.graph_from_place(place, network_type=network_type)

    nodes_wgs = ox.graph_to_gdfs(graph, nodes=True, edges=False)
    node_lookup = pd.DataFrame({
        "lat": nodes_wgs.geometry.y,
        "lng": nodes_wgs.geometry.x,
    })

    graph = ox.project_graph(graph, to_crs=f"EPSG:{METRIC_CRS}")

    edges = ox.graph_to_gdfs(graph, nodes=False, edges=True).reset_index()
    edges["edge_id"] = edges.index.astype(int)

    if "osmid" in edges.columns:
        edges["osmid"] = edges["osmid"].apply(
            lambda v: ",".join(str(x) for x in v) if isinstance(v, (list, tuple)) else str(v)
        )

    edges = edges.merge(
        node_lookup.rename(columns={"lat": "u_lat", "lng": "u_lng"}),
        left_on="u", right_index=True, how="left",
    )
    edges = edges.merge(
        node_lookup.rename(columns={"lat": "v_lat", "lng": "v_lng"}),
        left_on="v", right_index=True, how="left",
    )

    keep = [c for c in ["edge_id", "osmid", "u", "u_lat", "u_lng", "v", "v_lat", "v_lng",
                        "key", "name", "highway", "length", "lit", "geometry"]
            if c in edges.columns]
    edges = edges[keep].copy()

    boundary = ox.geocode_to_gdf(place).to_crs(epsg=METRIC_CRS)
    boundary_poly = boundary.geometry.iloc[0]

    print(f"[network] {len(edges)} street edges")
    if "lit" not in edges.columns:
        print("[network] note: no edge in this download has an OSM 'lit' tag, "
              "is_lit in edge_lamp_features will rely entirely on whether a "
              "Camden lamp got snapped to that edge.")
    return boundary_poly, edges


# --------------------------------------------------------------------------
# 2. DATE RANGE
# --------------------------------------------------------------------------

def resolve_months(desired_start: str, desired_end: str) -> list[str]:
    """Ask data.police.uk which months it currently publishes, and clip our
    wishlist (DESIRED_START to DESIRED_END) to whatever actually exists.
    """
    resp = requests.get("https://data.police.uk/api/crimes-street-dates", timeout=30)
    resp.raise_for_status()
    available = sorted(d["date"] for d in resp.json())
    months = [m for m in available if desired_start <= m <= desired_end]
    if not months:
        raise RuntimeError(
            f"No published months fall between {desired_start} and {desired_end}. "
            f"Currently available: {available[0]} to {available[-1]}."
        )
    if months[0] > desired_start:
        print(f"[dates] note: requested start {desired_start} is older than the "
              f"earliest month currently published ({months[0]}). data.police.uk "
              f"only keeps a rolling ~3 year window, so earlier months have aged "
              f"out. Using {months[0]} onward instead ({len(months)} months total).")
    else:
        print(f"[dates] using {months[0]} to {months[-1]} ({len(months)} months)")
    return months


# --------------------------------------------------------------------------
# 3. CRIME
# --------------------------------------------------------------------------

def _poly_param(boundary_poly_wgs84, max_vertices: int = 140) -> str:
    """Turn a WGS84 polygon into the API's 'lat,lng:lat,lng:...' string."""
    poly = boundary_poly_wgs84
    tol = 0.0005
    coords = list(poly.exterior.coords)
    while len(coords) > max_vertices:
        poly = poly.simplify(tol)
        coords = list(poly.exterior.coords)
        tol *= 1.5
    return ":".join(f"{y:.5f},{x:.5f}" for x, y in coords)


def fetch_crime(boundary_poly_metric, months: list[str]) -> gpd.GeoDataFrame:
    """Pull street level crime for the area, one month at a time.

    Keeps category, our own harm-based severity tier AND our own
    perceived-risk tier (see the big comment above CRIME_SEVERITY), and
    month (YYYY-MM). data.police.uk does not publish a day or time for any
    crime, and has no severity/grade/perception field of its own.
    """
    poly_wgs = (gpd.GeoSeries([boundary_poly_metric], crs=METRIC_CRS)
                .to_crs(epsg=WGS84).iloc[0])
    poly_str = _poly_param(poly_wgs)

    url = "https://data.police.uk/api/crimes-street/all-crime"
    rows = []
    for month in months:
        print(f"[crime] {month} ...", end=" ")
        resp = requests.post(url, data={"poly": poly_str, "date": month}, timeout=120)
        if resp.status_code == 503:
            print("503: over 10,000 crimes in one call, split the area or the month")
            continue
        resp.raise_for_status()
        batch = resp.json()
        for c in batch:
            loc = c.get("location") or {}
            category = c.get("category")
            rows.append({
                "crime_id": c.get("persistent_id") or c.get("id"),
                "category": category,
                "severity": CRIME_SEVERITY.get(category, DEFAULT_SEVERITY),
                "perceived_risk": CRIME_PERCEIVED_RISK.get(category, DEFAULT_PERCEIVED_RISK),
                "month": c.get("month"),
                "lat": loc.get("latitude"),
                "lng": loc.get("longitude"),
            })
        print(f"{len(batch)} records")
        time.sleep(1)

    df = pd.DataFrame(rows)
    return _points_to_gdf(df, "lat", "lng")


# --------------------------------------------------------------------------
# 4. STOP AND SEARCH
# --------------------------------------------------------------------------

def fetch_stop_search(boundary_poly_metric, months: list[str]) -> gpd.GeoDataFrame:
    """Pull street level stop and search records for the area, month by month.

    Deliberately keeps only type, object_of_search, outcome, and datetime,
    plus coordinates. Gender, age_range, self_defined_ethnicity and
    officer_defined_ethnicity are dropped on purpose: sensitive personal
    attributes with no legitimate role in a street safety score, including
    them risks baking in discriminatory bias.
    """
    poly_wgs = (gpd.GeoSeries([boundary_poly_metric], crs=METRIC_CRS)
                .to_crs(epsg=WGS84).iloc[0])
    poly_str = _poly_param(poly_wgs)

    url = "https://data.police.uk/api/stops-street"
    rows = []
    for month in months:
        print(f"[stop-search] {month} ...", end=" ")
        resp = requests.post(url, data={"poly": poly_str, "date": month}, timeout=120)
        if resp.status_code == 503:
            print("503: too many records in one call, split the area or the month")
            continue
        if resp.status_code == 404:
            print("no data for this month")
            continue
        resp.raise_for_status()
        batch = resp.json()
        for s in batch:
            loc = s.get("location") or {}
            dt = s.get("datetime")
            rows.append({
                "type": s.get("type"),
                "object_of_search": s.get("object_of_search"),
                "outcome": s.get("outcome"),
                "datetime": dt,
                "month": dt[:7] if dt else month,
                "lat": loc.get("latitude"),
                "lng": loc.get("longitude"),
            })
        print(f"{len(batch)} records")
        time.sleep(1)

    df = pd.DataFrame(rows)
    if df.empty:
        print("[stop-search] no records at all for this area/date range")
        return gpd.GeoDataFrame(
            {"type": [], "object_of_search": [], "outcome": [], "datetime": [], "month": []},
            geometry=[], crs=f"EPSG:{METRIC_CRS}",
        )
    return _points_to_gdf(df, "lat", "lng")


# --------------------------------------------------------------------------
# 5. STREET LIGHTING (Camden)
# --------------------------------------------------------------------------

def fetch_lighting(url: str) -> gpd.GeoDataFrame:
    """Pull Camden lamp points through the Socrata API, extracting install dates."""
    rows, offset, page = [], 0, 5000
    while True:
        params = {"$limit": page, "$offset": offset}
        resp = requests.get(url, params=params, timeout=120)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        offset += page
        if len(batch) < page:
            break

    lats, lngs, install_months = [], [], []
    for r in rows:
        lat, lng = _extract_lat_lng(r)
        lats.append(lat)
        lngs.append(lng)
        
        # Socrata Floating Timestamps are ISO 8601 (e.g., '2014-12-19T00:00:00')
        install_date = r.get("install_date")
        if install_date:
            install_months.append(install_date[:7])
        else:
            install_months.append("1900-01") # Legacy infrastructure fallback

    df = pd.DataFrame(rows)
    df["lat"], df["lng"] = lats, lngs
    df["install_month"] = install_months

    for col in list(df.columns):
        if df[col].apply(lambda v: isinstance(v, (dict, list))).any():
            df = df.drop(columns=col)

    print(f"[lighting] {len(df)} lamp records")
    return _points_to_gdf(df, "lat", "lng")


def _extract_lat_lng(record: dict):
    """Handle the common Socrata coordinate shapes for one row."""
    for key in ("location", "point", "geocoded_column", "the_geom"):
        val = record.get(key)
        if isinstance(val, dict) and val.get("type") == "Point":
            lng, lat = shape(val).x, shape(val).y
            return lat, lng
        if isinstance(val, dict) and "latitude" in val and "longitude" in val:
            return float(val["latitude"]), float(val["longitude"])
    for lat_k, lng_k in (("latitude", "longitude"), ("lat", "lng"),
                         ("lat", "long"), ("y", "x")):
        if lat_k in record and lng_k in record:
            try:
                return float(record[lat_k]), float(record[lng_k])
            except (TypeError, ValueError):
                pass
    return None, None


# --------------------------------------------------------------------------
# SHARED CLEANING
# --------------------------------------------------------------------------

def _points_to_gdf(df: pd.DataFrame, lat_col: str, lng_col: str) -> gpd.GeoDataFrame:
    """Common cleaning: drop missing coords, drop duplicates, project to metres."""
    before = len(df)
    df = df.dropna(subset=[lat_col, lng_col]).copy()
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lng_col] = pd.to_numeric(df[lng_col], errors="coerce")
    df = df.dropna(subset=[lat_col, lng_col])
    df = df.drop_duplicates()

    geom = [Point(xy) for xy in zip(df[lng_col], df[lat_col])]
    gdf = gpd.GeoDataFrame(df, geometry=geom, crs=f"EPSG:{WGS84}")
    gdf = gdf.to_crs(epsg=METRIC_CRS)
    print(f"[clean] kept {len(gdf)} of {before} points")
    return gdf


# --------------------------------------------------------------------------
# 6. SNAP POINTS TO EDGES, BUILD FEATURES
# --------------------------------------------------------------------------

def snap_counts(points: gpd.GeoDataFrame, edges: gpd.GeoDataFrame,
                max_dist: float, out_name: str,
                weight_col: str | None = None) -> pd.Series:
    """Nearest-edge method: match each point to its SINGLE nearest street
    edge. Counts by default, or sums weight_col per edge if given (used for
    severity/perceived-risk). This is method (a), see the module docstring;
    still used for the monthly breakdown and the spatial lag."""
    if points.empty:
        return pd.Series(0, index=edges["edge_id"], name=out_name)

    joined = gpd.sjoin_nearest(
        points, edges[["edge_id", "geometry"]],
        how="left", max_distance=max_dist, distance_col="_dist",
    )
    joined = joined.dropna(subset=["edge_id"])

    if weight_col is not None:
        agg = joined.groupby("edge_id")[weight_col].sum()
    else:
        agg = joined.groupby("edge_id").size()

    agg = agg.reindex(edges["edge_id"], fill_value=0)
    agg.name = out_name
    return agg


def snap_counts_by_month(points: gpd.GeoDataFrame, edges: gpd.GeoDataFrame,
                         max_dist: float, prefix: str,
                         weight_col: str | None = None) -> pd.DataFrame:
    """Same idea as snap_counts, pivoted into one column per month present
    in points['month']. Column names use underscores, e.g. crime_count_2024_01.
    """
    if points.empty or "month" not in points.columns:
        return pd.DataFrame(index=pd.Index(edges["edge_id"], name="edge_id"))

    months = sorted(m for m in points["month"].dropna().unique())
    cols = {}
    for m in months:
        safe_m = str(m).replace("-", "_")
        sub = points[points["month"] == m]
        cols[f"{prefix}_{safe_m}"] = snap_counts(
            sub, edges, max_dist, f"{prefix}_{safe_m}", weight_col=weight_col
        )
    return pd.DataFrame(cols)


def add_spatial_lag(features: pd.DataFrame, edges: gpd.GeoDataFrame,
                    cols: list[str]) -> pd.DataFrame:
    """Spatial lag: for each edge, the total of `cols` on its directly
    neighbouring edges, where "neighbouring" means sharing a junction node
    (u or v) with it. Adds <col>_lag_sum and <col>_lag_mean for each col,
    applied only to the overall totals to keep the column count sane.
    """
    node_edges = defaultdict(set)
    for eid, u, v in zip(edges["edge_id"], edges["u"], edges["v"]):
        node_edges[u].add(eid)
        node_edges[v].add(eid)

    neighbours = {}
    for eid, u, v in zip(edges["edge_id"], edges["u"], edges["v"]):
        neigh = (node_edges[u] | node_edges[v]) - {eid}
        neighbours[eid] = neigh

    lookup = features.set_index("edge_id")
    edge_ids = features["edge_id"].tolist()

    for col in cols:
        vals = lookup[col]
        sums, means = [], []
        for eid in edge_ids:
            neigh = neighbours.get(eid, set())
            if neigh:
                neigh_vals = vals.loc[list(neigh)]
                sums.append(neigh_vals.sum())
                means.append(neigh_vals.mean())
            else:
                sums.append(0)
                means.append(0.0)
        features[f"{col}_lag_sum"] = sums
        features[f"{col}_lag_mean"] = means

    return features


# --------------------------------------------------------------------------
# 6b. NETWORK KDE (NetKDE): the ML-facing smoothing.
#
# Distance is measured ALONG the walkable street network (Dijkstra shortest
# path), not in a straight line, and the smoothing is applied SEPARATELY to
# each month so no future data leaks into the past. This mirrors the ML
# lead's own NetKDE function, moved upstream so it runs once here.
# --------------------------------------------------------------------------

def build_edge_graph(edges: gpd.GeoDataFrame) -> nx.Graph:
    """Build ONE undirected weighted graph (weight = street length in metres)
    and reuse it for every month. The street network doesn't change month to
    month, so there is no reason to rebuild it each time.

    Parallel edges between the same pair of junction nodes (different OSM
    ways joining the same two junctions, distinguished by 'key') collapse to
    a single graph edge, keeping the SHORTEST of the parallel lengths.
    """
    G = nx.Graph()
    for u, v, length in zip(edges["u"], edges["v"], edges["length"]):
        if G.has_edge(u, v):
            if length < G[u][v]["weight"]:
                G[u][v]["weight"] = length
        else:
            G.add_edge(u, v, weight=float(length))
    return G


def _edge_to_pair_lookup(edges: gpd.GeoDataFrame, G: nx.Graph) -> dict:
    """Map every edge_id to the (u, v) node pair that exists in G, so a
    smoothed value reaches EVERY edge_id sharing that junction pair, not
    just whichever parallel edge 'won' when the graph was built.
    """
    lookup = {}
    for eid, u, v in zip(edges["edge_id"], edges["u"], edges["v"]):
        if G.has_edge(u, v):
            lookup[eid] = (u, v)
    return lookup


def netkde_smooth_month(month_df: pd.DataFrame, G: nx.Graph, edge_to_pair: dict,
                        cols_to_smooth: list[str], bandwidth: float) -> dict:
    """NetKDE for ONE month only. For every edge that has any event this
    month, spread that event's value outward from the edge's start node
    along the network (Dijkstra), decaying with a Gaussian kernel
    (sigma = bandwidth / 3). Returns {(u, v): {col: smoothed_value}}.

    Same logic as the ML lead's calculate_netkde_for_month, kept close on
    purpose so the two match.
    """
    smoothed = {pair: {col: 0.0 for col in cols_to_smooth} for pair in set(G.edges())}

    present = [c for c in cols_to_smooth if c in month_df.columns]
    if not present:
        return smoothed

    event_mask = (month_df[present] > 0).any(axis=1)
    event_rows = month_df[event_mask]
    sigma = bandwidth / 3.0

    for _, source_row in event_rows.iterrows():
        pair = edge_to_pair.get(source_row["edge_id"])
        if pair is None:
            continue
        source_u = pair[0]

        ego = nx.ego_graph(G, source_u, radius=bandwidth, distance="weight")
        try:
            path_lengths = nx.single_source_dijkstra_path_length(
                ego, source_u, weight="weight"
            )
        except nx.NetworkXNoPath:
            path_lengths = {source_u: 0.0}

        for u, v in ego.edges():
            dist_u = path_lengths.get(u, bandwidth)
            dist_v = path_lengths.get(v, bandwidth)
            dist = min(dist_u, dist_v)
            decay = np.exp(-0.5 * (dist / sigma) ** 2)

            target = (u, v) if (u, v) in smoothed else (v, u)
            if target not in smoothed:
                continue
            for col in present:
                val = source_row[col]
                if val > 0:
                    smoothed[target][col] += val * decay

    return smoothed


def netkde_by_month(edges: gpd.GeoDataFrame, monthly_wide: pd.DataFrame,
                    metrics: list[str], months: list[str],
                    bandwidth: float = NETKDE_BANDWIDTH_M) -> pd.DataFrame:
    """Apply netkde_smooth_month separately to each resolved month and stitch
    the results into {metric}_netkde_YYYY_MM columns. The naming matches the
    plain monthly breakdown, so the ML wide-to-long reshape that already
    grabs columns ending in _YYYY_MM will pick these up automatically.

    monthly_wide must hold the per-month nearest-edge totals (columns like
    crime_count_2024_01), indexed/keyed by edge_id.
    """
    print(f"[netkde] network-distance smoothing, bandwidth {bandwidth}m, "
          f"{len(metrics)} metrics x {len(months)} months (built graph once)")

    G = build_edge_graph(edges)
    edge_to_pair = _edge_to_pair_lookup(edges, G)

    out = pd.DataFrame(index=pd.Index(edges["edge_id"], name="edge_id"))

    for month in months:
        safe_m = str(month).replace("-", "_")
        # source columns for this month, renamed back to the bare metric name
        source = {m: f"{m}_{safe_m}" for m in metrics
                  if f"{m}_{safe_m}" in monthly_wide.columns}
        if not source:
            continue

        month_df = monthly_wide.reset_index()[["edge_id"] + list(source.values())].copy()
        month_df = month_df.rename(columns={v: k for k, v in source.items()})

        smoothed = netkde_smooth_month(month_df, G, edge_to_pair,
                                       list(source.keys()), bandwidth)

        for m in source:
            col = f"{m}_netkde_{safe_m}"
            out[col] = [smoothed.get(edge_to_pair.get(eid), {}).get(m, 0.0)
                        for eid in edges["edge_id"]]

    return out


def build_crime_stop_features(edges: gpd.GeoDataFrame, crime: gpd.GeoDataFrame,
                              stops: gpd.GeoDataFrame, months: list[str]) -> gpd.GeoDataFrame:
    """Crime and stop-search features only, NOT lamps (see build_lamp_features).

    Produces, per edge:
      - nearest-edge overall totals across the full resolved date range:
        crime_count, crime_severity_sum, crime_perceived_risk_sum,
        stop_search_count (+ per-km and average helpers)
      - a per-month breakdown of the same, as separate columns
      - a spatial lag of the overall totals (see add_spatial_lag)
      - per-month NetKDE columns (network distance, ML-facing, see
        netkde_by_month): {metric}_netkde_YYYY_MM
    """
    crime_count = snap_counts(crime, edges, CRIME_SNAP_MAX_M, "crime_count")
    crime_severity_sum = snap_counts(crime, edges, CRIME_SNAP_MAX_M,
                                     "crime_severity_sum", weight_col="severity")
    crime_perceived_risk_sum = snap_counts(crime, edges, CRIME_SNAP_MAX_M,
                                           "crime_perceived_risk_sum",
                                           weight_col="perceived_risk")
    stop_search_count = snap_counts(stops, edges, STOP_SEARCH_SNAP_MAX_M, "stop_search_count")

    out = (edges.merge(crime_count, on="edge_id")
                .merge(crime_severity_sum, on="edge_id")
                .merge(crime_perceived_risk_sum, on="edge_id")
                .merge(stop_search_count, on="edge_id"))

    monthly_frames = [
        snap_counts_by_month(crime, edges, CRIME_SNAP_MAX_M, "crime_count"),
        snap_counts_by_month(crime, edges, CRIME_SNAP_MAX_M, "crime_severity_sum",
                             weight_col="severity"),
        snap_counts_by_month(crime, edges, CRIME_SNAP_MAX_M, "crime_perceived_risk_sum",
                             weight_col="perceived_risk"),
        snap_counts_by_month(stops, edges, STOP_SEARCH_SNAP_MAX_M, "stop_search_count"),
    ]
    # keep a clean per-month wide table (edge_id + all *_YYYY_MM columns) so
    # NetKDE can smooth each month from it
    monthly_wide = pd.DataFrame(index=pd.Index(edges["edge_id"], name="edge_id"))
    for monthly in monthly_frames:
        if not monthly.empty:
            monthly_wide = monthly_wide.join(monthly, how="left")
            out = out.merge(monthly, left_on="edge_id", right_index=True, how="left")
    monthly_cols = [c for f in monthly_frames for c in f.columns]
    if monthly_cols:
        out[monthly_cols] = out[monthly_cols].fillna(0).astype(int)
        monthly_wide[monthly_cols] = monthly_wide[monthly_cols].fillna(0).astype(int)

    length_km = (out["length"] / 1000.0).clip(lower=1e-6)
    out["crime_per_km"] = out["crime_count"] / length_km
    out["crime_severity_per_km"] = out["crime_severity_sum"] / length_km
    crime_count_or_nan = out["crime_count"].where(out["crime_count"] != 0)
    out["crime_avg_severity"] = (out["crime_severity_sum"] / crime_count_or_nan).fillna(0.0).astype(float)
    out["crime_perceived_risk_per_km"] = out["crime_perceived_risk_sum"] / length_km
    out["crime_avg_perceived_risk"] = (out["crime_perceived_risk_sum"] / crime_count_or_nan).fillna(0.0).astype(float)
    out["stop_search_per_km"] = out["stop_search_count"] / length_km

    lag_cols = ["crime_count", "crime_severity_sum", "crime_perceived_risk_sum", "stop_search_count"]
    out = add_spatial_lag(out, edges, lag_cols)

    # NetKDE, per month, network distance, ML-facing
    netkde = netkde_by_month(edges, monthly_wide, NETKDE_METRICS, months, NETKDE_BANDWIDTH_M)
    out = out.merge(netkde, left_on="edge_id", right_index=True, how="left")
    netkde_cols = list(netkde.columns)
    if netkde_cols:
        out[netkde_cols] = out[netkde_cols].fillna(0.0)

    out.attrs["monthly_columns"] = monthly_cols
    out.attrs["lag_columns"] = [f"{c}_lag_sum" for c in lag_cols] + [f"{c}_lag_mean" for c in lag_cols]
    out.attrs["netkde_columns"] = netkde_cols
    return out


def build_lamp_features(edges: gpd.GeoDataFrame, lamps: gpd.GeoDataFrame, months: list[str]) -> gpd.GeoDataFrame:
    """Lamp features, evaluated month-by-month to reflect cumulative installations."""
    joined = gpd.sjoin_nearest(
        lamps, edges[["edge_id", "geometry"]],
        how="left", max_distance=LAMP_SNAP_MAX_M, distance_col="_dist"
    )
    joined = joined.dropna(subset=["edge_id"])
    
    keep_cols = [c for c in ["edge_id", "osmid", "u", "v", "length", "lit", "geometry"] if c in edges.columns]
    out = edges[keep_cols].copy()
    length_km = (out["length"] / 1000.0).clip(lower=1e-6)
    osm_lit = out["lit"].astype(str).str.lower().eq("yes") if "lit" in out else False

    # Static fallback
    lamp_count = joined.groupby("edge_id").size()
    out["lamp_count"] = out["edge_id"].map(lamp_count).fillna(0).astype(int)
    out["lamp_per_km"] = out["lamp_count"] / length_km
    out["is_lit"] = (osm_lit | (out["lamp_count"] > 0)).astype(int)

    # Monthly cumulative features
    for m in months:
        safe_m = str(m).replace("-", "_")
        active_lamps = joined[joined["install_month"] <= m]
        active_count = active_lamps.groupby("edge_id").size()
        
        out[f"lamp_count_{safe_m}"] = out["edge_id"].map(active_count).fillna(0).astype(int)
        out[f"lamp_per_km_{safe_m}"] = out[f"lamp_count_{safe_m}"] / length_km
        out[f"is_lit_{safe_m}"] = (osm_lit | (out[f"lamp_count_{safe_m}"] > 0)).astype(int)

    final_cols = [c for c in out.columns if c != "lit"] 
    return out[final_cols]


# --------------------------------------------------------------------------
# OPTIONAL: day / night boundaries for the study area
# --------------------------------------------------------------------------

def fetch_sun_times(lat: float, lng: float, dates: list[str]) -> pd.DataFrame:
    """Sunrise and sunset (UTC) for the area centroid, one row per date."""
    rows = []
    for d in dates:
        r = requests.get("https://api.sunrise-sunset.org/json",
                         params={"lat": lat, "lng": lng, "date": d, "formatted": 0},
                         timeout=60)
        r.raise_for_status()
        res = r.json().get("results", {})
        rows.append({"date": d, "sunrise": res.get("sunrise"),
                     "sunset": res.get("sunset")})
        time.sleep(1)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------

def main():
    boundary_poly, edges = get_network(PLACE, NETWORK_TYPE)

    months = resolve_months(DESIRED_START, DESIRED_END)

    crime = fetch_crime(boundary_poly, months)
    crime.to_file(OUT_DIR / "crime_points.gpkg", driver="GPKG")

    stops = fetch_stop_search(boundary_poly, months)
    stops.to_file(OUT_DIR / "stop_search_points.gpkg", driver="GPKG")

    lamps = fetch_lighting(CAMDEN_LIGHTING_URL)
    lamps.to_file(OUT_DIR / "lamp_points.gpkg", driver="GPKG")

    crime_features = build_crime_stop_features(edges, crime, stops, months)
    crime_features.to_file(OUT_DIR / "edge_features.gpkg", driver="GPKG")
    crime_features.drop(columns="geometry").to_csv(OUT_DIR / "edge_features.csv", index=False)

    lamp_features = build_lamp_features(edges, lamps, months)
    lamp_features.to_file(OUT_DIR / "edge_lamp_features.gpkg", driver="GPKG")
    lamp_features.drop(columns="geometry").to_csv(OUT_DIR / "edge_lamp_features.csv", index=False)

    print("\n[done] wrote:")
    for f in ["crime_points.gpkg", "stop_search_points.gpkg", "lamp_points.gpkg",
              "edge_features.gpkg", "edge_features.csv",
              "edge_lamp_features.gpkg", "edge_lamp_features.csv"]:
        print("  ", OUT_DIR / f)

    monthly_cols = crime_features.attrs.get("monthly_columns", [])
    lag_cols = crime_features.attrs.get("lag_columns", [])
    netkde_cols = crime_features.attrs.get("netkde_columns", [])
    print(f"\n[summary] {len(crime_features)} edges")
    print(f"[summary] totals summed across ALL resolved months "
          f"({months[0]} to {months[-1]}, {len(months)} months):")
    print(f"  crime_count total          = {int(crime_features['crime_count'].sum())}")
    print(f"  crime_severity_sum total    = {int(crime_features['crime_severity_sum'].sum())}")
    print(f"  crime_perceived_risk_sum    = {int(crime_features['crime_perceived_risk_sum'].sum())}")
    print(f"  stop_search total           = {int(crime_features['stop_search_count'].sum())}")
    print(f"  lamp_count total            = {int(lamp_features['lamp_count'].sum())}")
    print(f"[summary] {len(monthly_cols)} monthly breakdown columns")
    print(f"[summary] {len(lag_cols)} spatial lag columns")
    print(f"[summary] {len(netkde_cols)} NetKDE columns "
          f"(ML-facing, network distance, per month, bandwidth {NETKDE_BANDWIDTH_M}m), e.g.:")
    for c in netkde_cols[:4]:
        print(f"    {c}")


if __name__ == "__main__":
    main()
