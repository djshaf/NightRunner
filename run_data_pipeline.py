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
     edge, then builds one tidy edge level feature table for the models to
     train on.
  7. Saves everything to disk (GeoPackage plus a plain CSV of the features).

The data sources match the proposal Resources table:
  street network    -> OpenStreetMap (OSMnx)
  crime rates       -> data.police.uk
  stop and search   -> data.police.uk
  street lighting   -> Camden borough open data
  day / night       -> Sunrise-Sunset API (optional helper at the bottom)

Notes on data.police.uk, checked directly against their API docs:
  - Crime type: available, the "category" field (e.g. burglary,
    violent-crime, anti-social-behaviour). There are 14 real categories,
    see CRIME_SEVERITY below for the full list.
  - Crime grade / seriousness: NOT provided by the API, there is no
    official severity score. CRIME_SEVERITY below is our own simplification
    of two published methodologies onto the API's 14 categories, see the
    comment above that constant for the citations. It is not a reproduction
    of either paper's exact weights, just a coarse low/medium/high tier
    informed by their relative ordering of offence types.
  - Crime by day / exact date: NOT available for crimes, only "month"
    (YYYY-MM), that is a hard limit of the crimes-street endpoint.
  - Stop and search, by contrast, DOES carry a real datetime (down to the
    minute in most forces), see fetch_stop_search() below.
  - Date range: the API only keeps a rolling window of roughly the last
    three years. DESIRED_START below may be older than what is actually
    published by the time this runs, resolve_months() checks the live
    availability list and clips to whatever currently exists, printing a
    note if it had to.

Run this on a machine with normal internet access. It needs to reach
overpass-api.de (OSMnx), data.police.uk and opendata.camden.gov.uk.

Install once:
  pip install osmnx geopandas shapely requests pandas pyproj

Author: <your name>, Group 8
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import geopandas as gpd
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

# Our own coarse severity tiering of data.police.uk's 14 crime categories.
# The API itself has no seriousness/grade field, so this is informed by two
# published sources rather than invented from scratch:
#   1. Sherman, L., Neyroud, P. and Neyroud, E. (2016) 'The Cambridge Crime
#      Harm Index: Measuring Total Harm From Crime Based On Sentencing
#      Guidelines', Policing: A Journal of Policy and Practice, 10(3),
#      pp.171-183. Weights crimes by the custodial sentence starting point
#      for that offence type, violent and weapons offences score far higher
#      than property/theft offences.
#   2. Office for National Statistics (2016) 'Research outputs: developing
#      a Crime Severity Score for England and Wales using data on crimes
#      recorded by the police'. Same broad idea, official weights derived
#      from sentencing data, published on ons.gov.uk and updated yearly.
# Both agree on the same rough ordering (violence/weapons high, burglary/
# drugs/damage mid, theft/anti-social low), which is what this 1 to 3 tier
# reflects. It is a simplification, not a reproduction of either paper's
# exact per-offence weights, that would need the ~200 individual offence
# codes those studies use, not the 14 broad categories police.uk exposes.
# Worth a sanity check / citation in the report rather than treated as an
# official score.
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
DEFAULT_SEVERITY = 1  # fallback if police.uk ever adds a category not listed above

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
    can attach features to each street segment later. u_lat/u_lng and
    v_lat/v_lng give the WGS84 (plain GPS) coordinates of each edge's two
    endpoint nodes, sitting right next to u and v, since u and v on their
    own are just OpenStreetMap's internal node IDs and are not human or
    GIS-tool readable on their own.
    """
    print(f"[network] downloading '{place}' ...")
    graph = ox.graph_from_place(place, network_type=network_type)

    # grab node coordinates in WGS84 BEFORE projecting, node IDs (the index
    # here) stay identical after projection, only the coordinate values
    # would change, so this lookup remains valid to join against u / v later
    nodes_wgs = ox.graph_to_gdfs(graph, nodes=True, edges=False)
    node_lookup = pd.DataFrame({
        "lat": nodes_wgs.geometry.y,
        "lng": nodes_wgs.geometry.x,
    })

    graph = ox.project_graph(graph, to_crs=f"EPSG:{METRIC_CRS}")

    # edges only; keep the geometry and the length OSMnx already computed
    edges = ox.graph_to_gdfs(graph, nodes=False, edges=True).reset_index()
    edges["edge_id"] = edges.index.astype(int)

    # attach WGS84 lat/lng for each edge's u and v endpoint, right next to
    # the u / v columns themselves
    edges = edges.merge(
        node_lookup.rename(columns={"lat": "u_lat", "lng": "u_lng"}),
        left_on="u", right_index=True, how="left",
    )
    edges = edges.merge(
        node_lookup.rename(columns={"lat": "v_lat", "lng": "v_lng"}),
        left_on="v", right_index=True, how="left",
    )

    keep = [c for c in ["edge_id", "u", "u_lat", "u_lng", "v", "v_lat", "v_lng",
                        "key", "name", "highway", "length", "lit", "geometry"]
            if c in edges.columns]
    edges = edges[keep].copy()

    # the boundary polygon, in the same metric CRS, for clipping and for the
    # crime / stop and search queries
    boundary = ox.geocode_to_gdf(place).to_crs(epsg=METRIC_CRS)
    boundary_poly = boundary.geometry.iloc[0]

    print(f"[network] {len(edges)} street edges")
    return boundary_poly, edges


# --------------------------------------------------------------------------
# 2. DATE RANGE
# --------------------------------------------------------------------------

def resolve_months(desired_start: str, desired_end: str) -> list[str]:
    """Ask data.police.uk which months it currently publishes, and clip our
    wishlist (DESIRED_START to DESIRED_END) to whatever actually exists.

    The crimes-street-dates endpoint only ever lists a rolling ~3 year
    window. Anything older has aged out and simply will not be there, no
    error, the month is just missing from the list.
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
    """Turn a WGS84 polygon into the API's 'lat,lng:lat,lng:...' string.

    The API rejects very long polygons, so we simplify the outline until it
    has few enough vertices.
    """
    poly = boundary_poly_wgs84
    tol = 0.0005
    coords = list(poly.exterior.coords)
    while len(coords) > max_vertices:
        poly = poly.simplify(tol)
        coords = list(poly.exterior.coords)
        tol *= 1.5
    # API wants lat,lng (note the order: y then x)
    return ":".join(f"{y:.5f},{x:.5f}" for x, y in coords)


def fetch_crime(boundary_poly_metric, months: list[str]) -> gpd.GeoDataFrame:
    """Pull street level crime for the area, one month at a time.

    Keeps category (crime type), our own severity tier, and month (YYYY-MM)
    per record. data.police.uk does not publish a day or time for any
    crime, and has no severity/grade field of its own, only the category.
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
                "month": c.get("month"),
                "lat": loc.get("latitude"),
                "lng": loc.get("longitude"),
            })
        print(f"{len(batch)} records")
        time.sleep(1)  # be polite to a free public API

    df = pd.DataFrame(rows)
    return _points_to_gdf(df, "lat", "lng")


# --------------------------------------------------------------------------
# 4. STOP AND SEARCH
# --------------------------------------------------------------------------

def fetch_stop_search(boundary_poly_metric, months: list[str]) -> gpd.GeoDataFrame:
    """Pull street level stop and search records for the area, month by month.

    Deliberately keeps only type, object_of_search, outcome, and datetime,
    plus the coordinates. The raw API also returns gender, age_range,
    self_defined_ethnicity and officer_defined_ethnicity for each record.
    Those are left out on purpose: they are sensitive personal attributes
    with no legitimate role in a street safety score, and folding them into
    a model risks baking in discriminatory bias rather than measuring
    anything about the street itself. Only the count and nature of
    stop-and-search activity at a location is kept.

    Unlike crime records, these do carry a real datetime (the docs note
    some forces only supply a date and pad the time to midnight, so treat
    exact times with some caution), which is more granular than the
    month-only data crime records give us.
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
            # some forces do not supply stop and search for every month
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
                "month": dt[:7] if dt else None,
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
    """Pull Camden lamp points through the Socrata API, with paging.

    Socrata datasets store coordinates in a few different shapes, so this
    reads whatever it finds. Once you see the real data, check which branch
    fires and simplify if you like.
    """
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

    lats, lngs = [], []
    for r in rows:
        lat, lng = _extract_lat_lng(r)
        lats.append(lat)
        lngs.append(lng)
    df = pd.DataFrame(rows)
    df["lat"], df["lng"] = lats, lngs

    # drop nested / dict-valued columns (e.g. the raw 'location' GeoJSON
    # object, 'organisation_uri') - we already pulled lat/lng out of them,
    # and unhashable columns like these break drop_duplicates() downstream.
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

    # exact duplicate rows add nothing. NOTE: dedup on the full row, not just
    # lat/lng - crime and stop-search records from data.police.uk are
    # deliberately snapped to a shared anonymised point, so many genuinely
    # different incidents share the same coordinates on purpose. Deduping by
    # coordinates alone would wrongly collapse those into one. (fetch_lighting()
    # already strips out any dict/list-valued columns before this runs, so
    # full-row dedup is safe for all three point sources.)
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
    """Match each point to its nearest street edge.

    By default counts points per edge. If weight_col is given, sums that
    column per edge instead (used for the crime severity score).
    """
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


def build_features(edges: gpd.GeoDataFrame, crime: gpd.GeoDataFrame,
                   stops: gpd.GeoDataFrame, lamps: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Attach crime, stop-search, and lamp counts to each edge, derive simple features."""
    crime_count = snap_counts(crime, edges, CRIME_SNAP_MAX_M, "crime_count")
    crime_severity_sum = snap_counts(crime, edges, CRIME_SNAP_MAX_M,
                                     "crime_severity_sum", weight_col="severity")
    stop_search_count = snap_counts(stops, edges, STOP_SEARCH_SNAP_MAX_M, "stop_search_count")
    lamp_count = snap_counts(lamps, edges, LAMP_SNAP_MAX_M, "lamp_count")

    out = (edges.merge(crime_count, on="edge_id")
                .merge(crime_severity_sum, on="edge_id")
                .merge(stop_search_count, on="edge_id")
                .merge(lamp_count, on="edge_id"))

    # length in km, guarding against zero
    length_km = (out["length"] / 1000.0).clip(lower=1e-6)
    out["crime_per_km"] = out["crime_count"] / length_km
    out["crime_severity_per_km"] = out["crime_severity_sum"] / length_km
    out["crime_avg_severity"] = (out["crime_severity_sum"]
                                 / out["crime_count"].replace(0, pd.NA)).fillna(0)
    out["stop_search_per_km"] = out["stop_search_count"] / length_km
    out["lamp_per_km"] = out["lamp_count"] / length_km
    osm_lit = out["lit"].astype(str).str.lower().eq("yes") if "lit" in out else False
    out["is_lit"] = (osm_lit | (out["lamp_count"] > 0)).astype(int)
    return out


# --------------------------------------------------------------------------
# OPTIONAL: day / night boundaries for the study area
# --------------------------------------------------------------------------

def fetch_sun_times(lat: float, lng: float, dates: list[str]) -> pd.DataFrame:
    """Sunrise and sunset (UTC) for the area centroid, one row per date.

    Not an edge feature. It lets the app know when it is dark, which is the
    context the whole safety idea rests on.
    """
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

    features = build_features(edges, crime, stops, lamps)
    features.to_file(OUT_DIR / "edge_features.gpkg", driver="GPKG")

    # a plain table for model training, geometry dropped
    flat = features.drop(columns="geometry")
    flat.to_csv(OUT_DIR / "edge_features.csv", index=False)

    print("\n[done] wrote:")
    for f in ["crime_points.gpkg", "stop_search_points.gpkg", "lamp_points.gpkg",
              "edge_features.gpkg", "edge_features.csv"]:
        print("  ", OUT_DIR / f)
    print(f"\n[summary] {len(features)} edges, "
          f"{int(features['crime_count'].sum())} crimes snapped, "
          f"{int(features['stop_search_count'].sum())} stop and searches snapped, "
          f"{int(features['lamp_count'].sum())} lamps snapped")


if __name__ == "__main__":
    main()
