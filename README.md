# Data Collection and Cleaning

**Module:** ECS7036P
**Group:** 8
**Script:** `run_data_pipeline.py`

## Purpose

This script builds a street level dataset for a running route safety model. It combines three sources into a single feature table, with one row per street segment.

## Data Sources

The street network is obtained from OpenStreetMap through OSMnx, using the pedestrian network type. Crime data is obtained from the UK Police open data API, retrieved by month. Street lighting data is obtained from Camden council's open data portal.

## Cleaning Procedure

Records with missing coordinates are removed. Coordinate fields are converted to numeric values, and records that fail conversion are removed. Exact duplicate rows are removed. All coordinates are reprojected to British National Grid to enable distance calculations in metres.

Each crime and lamp point is matched to its nearest street edge, within fifty metres for crime and twenty five metres for lamps. Points beyond this threshold are excluded. Crime and lamp counts are then aggregated per edge, and used to derive crime density, lamp density, and a lit street indicator.

## Outputs

The script produces four files in a `data_out` folder: `crime_points.gpkg`, `lamp_points.gpkg`, `edge_features.gpkg`, and `edge_features.csv`. The CSV file is the flat feature table intended for model training.

## Requirements

Python with `osmnx`, `geopandas`, `shapely`, `requests`, `pandas`, and `pyproj`, and an active internet connection to reach OpenStreetMap, the police API, and Camden's open data portal.
