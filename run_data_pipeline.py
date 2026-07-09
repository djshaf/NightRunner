"""
Data collection and cleaning pipeline
ECS7036P Group 8: AI approach to running route optimisation for safety

Role: Data Lead (collection, cleaning, preparation)

What this script does, end to end:
  1. Pulls a routable walking/running street graph for the study area from
     OpenStreetMap via OSMnx, and turns it into an edge table.
  2. Pulls street level recorded crime for the same area from the UK Police
     open data API, month by month.
  3. Pulls Camden street lighting point data from the borough open data portal.
  4. Cleans all three (missing coordinates, duplicates, coordinate systems).
  5. Snaps crime points and lamp points onto the nearest street edge, then
     builds one tidy edge level feature table for the models to train on.
  6. Saves everything to disk (GeoPackage plus a plain CSV of the features).

The four data sources match the proposal Resources table:
  street network  -> OpenStreetMap (OSMnx)
  crime rates     -> data.police.uk
  street lighting -> Camden borough open data
  day / night     -> Sunrise-Sunset API (optional helper at the bottom)

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

# Months of crime to pull, in YYYY-MM. The police API is monthly only.
# Keep this list to whatever range is currently published (see notes at end).
MONTHS = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06"]

# "walk" gives a graph a runner can actually use: streets plus footpaths,
# ignoring motorways. It is the right base for this project.
NETWORK_TYPE = "walk"

# Camden street lighting, Socrata SODA JSON endpoint for dataset dfq3-8wzu.
# (jx8t-gxyu is the map visualisation wrapper and has no populated columns;
# dfq3-8wzu is the underlying table with real lat/lng fields.)
CAMDEN_LIGHTING_URL = "https://opendata.camden.gov.uk/resource/dfq3-8wzu.json"

# Metric coordinate system for Britain, so distances are in real metres.
METRIC_CRS = 27700          # British National Grid
WGS84 = 4326                # plain lat / lon

# A point further than this from any street is treated as unmatched and dropped.
CRIME_SNAP_MAX_M = 50       # crime locations are already street snapped by the police
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
    can attach features to each street segment later.
    """
    print(f"[network] downloading '{place}' ...")
    graph = ox.graph_from_place(place, network_type=network_type)
    graph = ox.project_graph(graph, to_crs=f"EPSG:{METRIC_CRS}")

    # edges only; keep the geometry and the length OSMnx already computed
    edges = ox.graph_to_gdfs(graph, nodes=False, edges=True).reset_index()
    edges["edge_id"] = edges.index.astype(int)

    keep = [c for c in ["edge_id", "u", "v", "key", "name", "highway",
                        "length", "lit", "geometry"] if c in edges.columns]
    edges = edges[keep].copy()

    # the boundary polygon, in the same metric CRS, for clipping and for the
    # crime query
    boundary = ox.geocode_to_gdf(place).to_crs(epsg=METRIC_CRS)
    boundary_poly = boundary.geometry.iloc[0]

    print(f"[network] {len(edges)} street edges")
    return boundary_poly, edges


# --------------------------------------------------------------------------
# 2. CRIME
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
    """Pull street level crime for the area, one month at a time."""
    # the API speaks lat / lon, so hand it the polygon in WGS84
    poly_wgs = (gpd.GeoSeries([boundary_poly_metric], crs=METRIC_CRS)
                .to_crs(epsg=WGS84).iloc[0])
    poly_str = _poly_param(poly_wgs)

    url = "https://data.police.uk/api/crimes-street/all-crime"
    rows = []
    for month in months:
        print(f"[crime] {month} ...", end=" ")
        # POST keeps us safe when the polygon string is long
        resp = requests.post(url, data={"poly": poly_str, "date": month}, timeout=120)
        if resp.status_code == 503:
            print("503: over 10,000 crimes in one call, split the area or the month")
            continue
        resp.raise_for_status()
        batch = resp.json()
        for c in batch:
            loc = c.get("location") or {}
            rows.append({
                "crime_id": c.get("persistent_id") or c.get("id"),
                "category": c.get("category"),
                "month": c.get("month"),
                "lat": loc.get("latitude"),
                "lng": loc.get("longitude"),
            })
        print(f"{len(batch)} records")
        time.sleep(1)  # be polite to a free public API

    df = pd.DataFrame(rows)
    return _points_to_gdf(df, "lat", "lng")


# --------------------------------------------------------------------------
# 3. STREET LIGHTING (Camden)
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
    # a) GeoJSON point column, e.g. record['location'] = {'type':'Point',...}
    for key in ("location", "point", "geocoded_column", "the_geom"):
        val = record.get(key)
        if isinstance(val, dict) and val.get("type") == "Point":
            lng, lat = shape(val).x, shape(val).y
            return lat, lng
        # b) Socrata "location" object with latitude / longitude keys
        if isinstance(val, dict) and "latitude" in val and "longitude" in val:
            return float(val["latitude"]), float(val["longitude"])
    # c) plain columns
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
    # lat/lng - crime records from data.police.uk are deliberately snapped to
    # a shared anonymised point, so many genuinely different crimes share the
    # same coordinates on purpose. Deduping by coordinates alone would wrongly
    # collapse those into one. (fetch_lighting() already strips out any
    # dict/list-valued columns before this runs, so full-row dedup is safe
    # for both crime and lighting.)
    df = df.drop_duplicates()

    geom = [Point(xy) for xy in zip(df[lng_col], df[lat_col])]
    gdf = gpd.GeoDataFrame(df, geometry=geom, crs=f"EPSG:{WGS84}")
    gdf = gdf.to_crs(epsg=METRIC_CRS)
    print(f"[clean] kept {len(gdf)} of {before} points")
    return gdf


# --------------------------------------------------------------------------
# 4. SNAP POINTS TO EDGES, BUILD FEATURES
# --------------------------------------------------------------------------

def snap_counts(points: gpd.GeoDataFrame, edges: gpd.GeoDataFrame,
                max_dist: float, out_name: str) -> pd.Series:
    """Match each point to its nearest street edge and count per edge."""
    if points.empty:
        return pd.Series(0, index=edges["edge_id"], name=out_name)

    joined = gpd.sjoin_nearest(
        points, edges[["edge_id", "geometry"]],
        how="left", max_distance=max_dist, distance_col="_dist",
    )
    joined = joined.dropna(subset=["edge_id"])
    counts = joined.groupby("edge_id").size()
    counts = counts.reindex(edges["edge_id"], fill_value=0)
    counts.name = out_name
    return counts


def build_features(edges: gpd.GeoDataFrame, crime: gpd.GeoDataFrame,
                   lamps: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Attach crime and lamp counts to each edge and derive simple features."""
    crime_count = snap_counts(crime, edges, CRIME_SNAP_MAX_M, "crime_count")
    lamp_count = snap_counts(lamps, edges, LAMP_SNAP_MAX_M, "lamp_count")

    out = edges.merge(crime_count, on="edge_id").merge(lamp_count, on="edge_id")

    # length in km, guarding against zero
    length_km = (out["length"] / 1000.0).clip(lower=1e-6)
    out["crime_per_km"] = out["crime_count"] / length_km
    out["lamp_per_km"] = out["lamp_count"] / length_km
    # a simple lit flag: either OSM tagged it lit, or we snapped a lamp to it
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

    crime = fetch_crime(boundary_poly, MONTHS)
    crime.to_file(OUT_DIR / "crime_points.gpkg", driver="GPKG")

    lamps = fetch_lighting(CAMDEN_LIGHTING_URL)
    lamps.to_file(OUT_DIR / "lamp_points.gpkg", driver="GPKG")

    features = build_features(edges, crime, lamps)
    features.to_file(OUT_DIR / "edge_features.gpkg", driver="GPKG")

    # a plain table for model training, geometry dropped
    flat = features.drop(columns="geometry")
    flat.to_csv(OUT_DIR / "edge_features.csv", index=False)

    print("\n[done] wrote:")
    for f in ["crime_points.gpkg", "lamp_points.gpkg",
              "edge_features.gpkg", "edge_features.csv"]:
        print("  ", OUT_DIR / f)
    print(f"\n[summary] {len(features)} edges, "
          f"{int(features['crime_count'].sum())} crimes snapped, "
          f"{int(features['lamp_count'].sum())} lamps snapped")


if __name__ == "__main__":
    main()
