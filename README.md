# Data Collection and Cleaning

**Module:** ECS7036P
**Group:** 8
**Script:** `run_data_pipeline.py`

## Purpose

This script builds a street level dataset for a running route safety model. It combines four sources into a single feature table, with one row per street segment.

## Data Sources

The street network is obtained from OpenStreetMap through OSMnx, using the pedestrian network type. Crime data and stop and search data are both obtained from the UK Police open data API, retrieved by month. Street lighting data is obtained from Camden council's open data portal.

## Date Range

Requests January 2021 to June 2026. data.police.uk only keeps a rolling ~3 year window, so `resolve_months()` checks what is actually published each run and clips the request accordingly, printing a note if the start date had to move. Earliest currently available: 2023-06.

## Crime Severity

Category (crime type) comes directly from the API. Severity does not, the API has no seriousness field, so a 1 to 3 tier has been added based on Sherman, Neyroud and Neyroud (2016) 'The Cambridge Crime Harm Index', Policing: A Journal of Policy and Practice, 10(3), and the ONS (2016) Crime Severity Score methodology. Both weight by sentencing severity; violent/weapons offences score highest, theft/anti-social lowest. This is a coarse simplification onto police.uk's 14 categories, not a reproduction of either paper's exact weights.

## Stop and Search

Type, object of search, outcome, and datetime are kept. Gender, age range, and ethnicity fields from the raw API are deliberately excluded, sensitive personal data with no legitimate role in a safety score, risks encoding bias otherwise.

## Cleaning Procedure

Records with missing coordinates are removed. Coordinate fields are converted to numeric values, and records that fail conversion are removed. Exact duplicate rows are removed. All coordinates are reprojected to British National Grid to enable distance calculations in metres.

Each crime, stop and search, and lamp point is matched to its nearest street edge, within fifty metres for crime and stop and search, twenty five for lamps. Points beyond this are excluded. Counts are aggregated per edge into density and severity features, plus a lit street indicator.

## Street Network References

u and v are OpenStreetMap's internal node IDs, not human readable. u_lat, u_lng, v_lat, v_lng sit immediately to their right, giving each endpoint's plain WGS84 coordinates.

## Outputs

Five files in `data_out`: `crime_points.gpkg`, `stop_search_points.gpkg`, `lamp_points.gpkg`, `edge_features.gpkg`, `edge_features.csv`. The CSV is the flat table for model training.

## Requirements

Python with `osmnx`, `geopandas`, `shapely`, `requests`, `pandas`, `pyproj`, and an internet connection to OpenStreetMap, the police API, and Camden's open data portal.
