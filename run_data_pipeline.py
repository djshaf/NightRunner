"""
Data collection and cleaning pipeline
ECS7036P Group 8: AI approach to running route optimisation for safety

Builds two edge-level (one row per street) feature tables for Camden from
OpenStreetMap, UK Police crime + stop-and-search, and Camden street lighting.
See README.md for data sources, method, dataset links, and how to run.
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

try:
    import polyline  # optional; only needed for the polyline6 CSV column
except ImportError:
    polyline = None


# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
PLACE = "London Borough of Camden, London, England"
DESIRED_START = "2021-01"
DESIRED_END = "2026-06"
NETWORK_TYPE = "walk"
CAMDEN_LIGHTING_URL = "https://opendata.camden.gov.uk/resource/dfq3-8wzu.json"

# NetKDE smoothing bandwidth, in metres of network (walking) distance.
# 125m matches the estimated spacing of the police API's anonymised points
# in Camden (see README). Change this one value to retune.
NETKDE_BANDWIDTH_M = 125
NETKDE_METRICS = ["crime_count", "crime_severity_sum",
                  "crime_perceived_risk_sum", "stop_search_count"]

# Two independent weightings of the API's 14 crime categories (see README
# for the sources behind each). severity = legal harm; perceived_risk = how
# unsafe it feels.
CRIME_SEVERITY = {
    "violent-crime": 3,
    "robbery": 3,
    "possession-of-weapons": 3,
    "burglary": 2,
    "criminal-damage-arson": 2,
    "drugs": 2,
    "public-order": 2,
    "theft-from-the-person": 2,
    "vehicle-crime": 2,
    "anti-social-behaviour": 1,
    "bicycle-theft": 1,
    "shoplifting": 1,
    "other-theft": 1,
    "other-crime": 1,
}
DEFAULT_SEVERITY = 1

CRIME_PERCEIVED_RISK = {
    "violent-crime": 4,
    "robbery": 4,
    "theft-from-the-person": 4,
    "possession-of-weapons": 4,
    "anti-social-behaviour": 3,
    "criminal-damage-arson": 3,
    "public-order": 3,
    "drugs": 3,
    "burglary": 2,
    "vehicle-crime": 2,
    "bicycle-theft": 2,
    "shoplifting": 1,
    "other-theft": 1,
    "other-crime": 1,
}
DEFAULT_PERCEIVED_RISK = 1

METRIC_CRS = 27700          # British National Grid (metres)
WGS84 = 4326                # lat / lon

CRIME_SNAP_MAX_M = 50
STOP_SEARCH_SNAP_MAX_M = 50
LAMP_SNAP_MAX_M = 25

OUT_DIR = Path(__file__).parent / "data_out"
OUT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------------------------
# 1. STREET NETWORK
# --------------------------------------------------------------------------

def get_network(place: str, network_type: str):
    """Download the street graph; return (boundary_polygon, edges_gdf) with
    edge_id, osmid, u/v node ids and their WGS84 coordinates."""
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
    """Clip [desired_start, desired_end] to the months the police API
    actually publishes (rolling ~3-year window)."""
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
    """Pull street-level crime month by month, tagging severity and
    perceived_risk per incident."""
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
    """Pull stop-and-search month by month. Gender, age and ethnicity fields
    are deliberately excluded (see README)."""
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
    """Pull Camden lamp points (paged), extracting install month."""
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
        install_date = r.get("install_date")
        if install_date:
            install_months.append(install_date[:7])
        else:
            install_months.append("1900-01")

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
    """Drop missing coords, drop duplicates, project to metres."""
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
    """Match each point to its single nearest edge; count, or sum weight_col."""
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
    """snap_counts pivoted into one column per month, e.g. crime_count_2024_01."""
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
    """Add <col>_lag_sum / <col>_lag_mean over edges sharing a junction node."""
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
# 6b. NETWORK KDE (NetKDE) - per-month, network-distance smoothing for the model
# --------------------------------------------------------------------------

def build_edge_graph(edges: gpd.GeoDataFrame) -> nx.Graph:
    """One undirected weighted graph (weight = length), reused for every
    month. Parallel edges collapse, keeping the shortest length."""
    G = nx.Graph()
    for u, v, length in zip(edges["u"], edges["v"], edges["length"]):
        if G.has_edge(u, v):
            if length < G[u][v]["weight"]:
                G[u][v]["weight"] = length
        else:
            G.add_edge(u, v, weight=float(length))
    return G


def _edge_to_pair_lookup(edges: gpd.GeoDataFrame, G: nx.Graph) -> dict:
    """Map every edge_id to its (u, v) pair in G."""
    lookup = {}
    for eid, u, v in zip(edges["edge_id"], edges["u"], edges["v"]):
        if G.has_edge(u, v):
            lookup[eid] = (u, v)
    return lookup


def netkde_smooth_month(month_df: pd.DataFrame, G: nx.Graph, edge_to_pair: dict,
                        cols_to_smooth: list[str], bandwidth: float) -> dict:
    """NetKDE for one month: diffuse each event along the network with a
    Gaussian kernel (sigma = bandwidth / 3). Returns {(u, v): {col: value}}."""
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
    """Run netkde_smooth_month per month into {metric}_netkde_YYYY_MM columns."""
    print(f"[netkde] network-distance smoothing, bandwidth {bandwidth}m, "
          f"{len(metrics)} metrics x {len(months)} months (built graph once)")

    G = build_edge_graph(edges)
    edge_to_pair = _edge_to_pair_lookup(edges, G)

    out = pd.DataFrame(index=pd.Index(edges["edge_id"], name="edge_id"))

    for month in months:
        safe_m = str(month).replace("-", "_")
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
    """Crime + stop-search features: totals, per-month breakdown, spatial lag,
    and per-month NetKDE columns."""
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

    lamp_count = joined.groupby("edge_id").size()
    out["lamp_count"] = out["edge_id"].map(lamp_count).fillna(0).astype(int)
    out["lamp_per_km"] = out["lamp_count"] / length_km
    out["is_lit"] = (osm_lit | (out["lamp_count"] > 0)).astype(int)

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
# 7. EXPORT HELPERS
# --------------------------------------------------------------------------

def export_with_polyline6(gdf: gpd.GeoDataFrame, out_dir: Path, file_prefix: str):
    """Save to GPKG (native geometry) and CSV. If the 'polyline' package is
    installed, the CSV also gets a polyline6 string column for the router."""
    df_out = gdf.copy()

    if polyline is not None:
        def geom_to_poly6(geom):
            if not geom or getattr(geom, 'is_empty', True):
                return ""
            coords = [(y, x) for x, y in geom.coords]  # Shapely (lng,lat) -> Polyline6 (lat,lng)
            return polyline.encode(coords, 6)
        df_out["polyline6"] = df_out["geometry"].apply(geom_to_poly6)
    else:
        print("[export] note: 'polyline' not installed, skipping polyline6 column "
              "(run 'pip install polyline' to include it).")

    gdf.to_file(out_dir / f"{file_prefix}.gpkg", driver="GPKG")
    df_csv = df_out.drop(columns=["geometry"])
    df_csv.to_csv(out_dir / f"{file_prefix}.csv", index=False)


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
    export_with_polyline6(crime_features, OUT_DIR, "edge_features")

    lamp_features = build_lamp_features(edges, lamps, months)
    export_with_polyline6(lamp_features, OUT_DIR, "edge_lamp_features")

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
