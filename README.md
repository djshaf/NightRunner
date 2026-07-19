**Module:** ECS7036P
**Group:** 8
**Script:** `run_data_pipeline.py`

# Data Collection and Cleaning

## Purpose

This script builds a street level dataset for a running route safety model. Crime, stop and search, and lighting are each turned into edge level features, one row per street segment.

## Data Sources

The street network is obtained from OpenStreetMap through OSMnx, using the pedestrian network type. Crime data and stop and search data are both obtained from the UK Police open data API, retrieved by month. Street lighting data is obtained from Camden Council's open data portal.

## Date Range

Requests from January 2021 to June 2026. data.police.uk only keeps a rolling ~3 year window, so `resolve_months()` checks what is actually published each run and clips the request accordingly, printing a note if the start date had to move. Earliest currently available: 2023-06. Console output states exactly which months are included.

## Crime Weightings

`crime_severity` is harm-based (1 to 3), from Sherman, Neyroud and Neyroud (2016) 'The Cambridge Crime Harm Index', Policing: A Journal of Policy and Practice, 10(3), and the ONS (2016) Crime Severity Score methodology, both weighted by sentencing severity.

`crime_perceived_risk` is a separate new column (1 to 4), how unsafe a crime makes an area feel, not how harmful it legally is. Based on Innes (2004) 'Signal crimes and signal disorders', British Journal of Sociology, 55(3), and Innes and Fielding (2002), Sociological Research Online, 7(2): the Signal Crimes Perspective, visible disorder like anti-social behaviour disproportionately drives fear of crime independent of its statistical severity. Also informed by ONS Crime Survey for England and Wales (CSEW) perception/ASB releases. Under this column, anti-social behaviour (3) scores above burglary (2), the reverse of the severity ordering, deliberately.

Both are coarse simplifications onto police.uk's 14 categories, not a reproduction of either study's exact weights; cite accordingly.

## Location Anonymisation and Average Spacing

Crime/stop-search coordinates are not exact; each is snapped to the nearest of roughly 680,000 anonymous map points (mostly street centres), each with a catchment of at least 8 postal addresses. No official average spacing figure is published; our own Camden data gives a rough estimate, about 1,384 distinct snap points across Camden's ~21.8 sq km in one 6-month pull, roughly one point every ~125m as an upper bound.

## Stop and Search

Type, object of search, outcome, and datetime are kept. Gender, age range, and ethnicity fields from the raw API are deliberately excluded, sensitive personal data with no legitimate role in a safety score.

## Monthly Breakdown

`edge_features.csv` carries both an overall total per edge (`crime_count`, `crime_severity_sum`, `crime_perceived_risk_sum`, `stop_search_count`, summed across the full resolved date range) and a per-month breakdown as separate columns, e.g. `crime_count_2024_01`. Lamps are a live snapshot with no time dimension, so are not split by month.

## Spatial Lag (Neighbouring Streets)

`<col>_lag_sum` / `<col>_lag_mean` give the total/average of that column across directly neighbouring edges (edges sharing a junction node), applied to the four overall totals only. **Naming clash to be aware of:** this is a spatial lag (nearby streets, same month). The ML/NN pipeline section below also uses "lag" (e.g. `crime_count_lag_1`) for a temporal lag (same street, previous months). Same word, different axis, do not treat them as the same column.

## Radius-Weighted Surroundings Density

Alongside the nearest-edge counts above, `edge_features.csv` also has a second, separate method: for four radii (50m, 75m, 125m, 150m), every crime/stop-search within that radius of a street contributes to it, weighted down linearly to zero at the radius edge (a crime right on the street counts fully, one at the edge of the radius counts almost nothing). This is a simplified Kernel Density Estimation (KDE), the standard technique in crime hotspot mapping (Chainey, Tompson and Uhlig, 2008), following the general principle that nearby things matter more than distant ones (Tobler's First Law of Geography, 1970). 125m was chosen deliberately; it matches our own estimate of how far apart data.police.uk's anonymised points are in Camden (see Location Anonymisation above), so it's roughly the smallest radius that reliably spans that uncertainty. Columns: `crime_density_r{R}` / `stop_search_density_r{R}` (weighted count), and `severity_wavg_r{R}` / `perceived_risk_wavg_r{R}` (distance-weighted average) for each radius.

## Lamp Features Kept Separate

Lamp features (`lamp_count`, `lamp_per_km`, `is_lit`) are no longer merged into `edge_features.csv`. They live in their own `edge_lamp_features.csv`/`.gpkg`, joinable back on `edge_id`. Note: no OSM edge in Camden carries a `lit` tag, so `is_lit` here depends entirely on whether a lamp snapped to that edge.

## Cleaning Procedure

Records with missing coordinates are removed. Coordinate fields are converted to numeric values, and records that fail conversion are removed. Exact duplicate rows are removed. All coordinates are reprojected to British National Grid to enable distance calculations in metres.

## Street Network References

`osmid` is the true OpenStreetMap way ID for that street (comma-joined if OSMnx merged several OSM ways into one edge), so any row can be looked up directly on openstreetmap.org. u and v are OpenStreetMap's internal node IDs, not human-readable on their own. u_lat, u_lng, v_lat, v_lng sit immediately to their right, giving each endpoint's plain WGS84 coordinates. All four (`osmid`, `u`, `v`, plus the lat/lng pairs) are included in both `edge_features` and `edge_lamp_features`.

## Street Network References

u and v are OpenStreetMap's internal node IDs, not human-readable. u_lat, u_lng, v_lat, v_lng sit immediately to their right, giving each endpoint's plain WGS84 coordinates.

## Outputs

Seven files in `data_out`: `crime_points.gpkg`, `stop_search_points.gpkg`, `lamp_points.gpkg`, `edge_features.gpkg`, `edge_features.csv`, `edge_lamp_features.gpkg`, `edge_lamp_features.csv`.

## Requirements

Python with `osmnx`, `geopandas`, `shapely`, `requests`, `pandas`, `pyproj`, and an internet connection to OpenStreetMap, the police API, and Camden's open data portal.

# ML/NN pipeline

## Purpose
Processes historical street-level crime and stop-and-search features, trains a machine learning model to forecast future crime risk, and transforms those predictions into physical routing penalties for integration with pedestrian routing engines.

## Data Restructuring
The initial wide-format dataset is melted and pivoted into a model-ready long-format time-series structure. Each row now represents a single street segment (edge_id) during a single specific month.

## Feature Engineering
Temporal lag features are engineered to provide the model with historical context. Using a grouped shift operation strictly isolated by edge_id, the pipeline generates 1-month, 2-month, and 3-month lookback columns (e.g., crime_count_lag_1, crime_severity_sum_lag_2, stop_search_count_lag_1). All-time cumulative columns and concurrent target metrics are deliberately dropped from the feature set to prevent data leakage.

## Validation Strategy
Data is split chronologically. Random train/test splits (which cause time-travel data leakage) are strictly avoided. The model trains on historical months and validates on a held-out future cutoff period.

## Model Architecture
An XGBoost Regressor is used as the baseline spatial-temporal forecaster. The algorithm natively handles categorical spatial attributes (e.g., highway tags like "residential" or "pedestrian") without requiring one-hot encoding. Early stopping is configured using the evaluation set to halt tree construction when predictive accuracy plateaus.

## Future Prediction and Cost Scaling
To forecast an unknown future month, the latest available month's actual data is programmatically shifted forward to become the new lag_1. Once raw future crime counts are predicted, they are converted into a spatial density metric (predicted_crime / length). This density is normalized via Min-Max scaling into an edge_safety_cost scale. A minimum base cost of 1.0 is strictly enforced to ensure downstream graph traversal algorithms do not encounter zero-weight edges.

## Outputs
One file in the output directory: osm_safety_tags.csv. This contains only the edge_id and the safety_tag_value, providing a clean mapping table to inject custom penalty tags (e.g., safety_cost=8) into the raw .osm.pbf map file before building the Valhalla routing graph. Node-level aggregations are excluded as Valhalla requires continuous edge penalties to prevent routing down the length of dangerous streets.

## References:

Boeing, G. (2017) 'OSMnx: new methods for acquiring, constructing, analyzing, and visualizing complex street networks', Computers, Environment and Urban Systems, 65, pp. 126–139.

Chainey, S., Tompson, L. and Uhlig, S. (2008) 'The utility of hotspot mapping for predicting spatial patterns of crime', Security Journal, 21(1–2), pp. 4–28.

Innes, M. (2004) 'Signal crimes and signal disorders: notes on deviance as communicative action', The British Journal of Sociology, 55(3), pp. 335–355.

Innes, M. and Fielding, N. (2002) 'From community to communicative policing: "signal crimes" and the problem of public reassurance', Sociological Research Online, 7(2). Available at: https://doi.org/10.5153/sro.724

London Borough of Camden (2026) Camden Street Lighting Location [Dataset]. Available at: https://opendata.camden.gov.uk/resource/dfq3-8wzu.json (Accessed: 17 July 2026).

Office for National Statistics (2016) Research outputs: developing a Crime Severity Score for England and Wales using data on crimes recorded by the police. Available at: https://www.ons.gov.uk/peoplepopulationandcommunity/crimeandjustice/articles/researchoutputsdevelopingacrimeseverityscoreforenglandandwalesusingdataoncrimesrecordedbythepolice/2016-11-29 (Accessed: 17 July 2026).

Office for National Statistics (2022) Crime Survey for England and Wales (CSEW) perception and Anti-Social Behaviour (ASB) by Police Force Area (PFA) for 2014 and 2021. Available at: https://www.ons.gov.uk/aboutus/transparencyandgovernance/freedomofinformationfoi/crimesurveyforenglandandwalescsewperceptionandantisocialbehaviourasbbypoliceforceareapfafor2014and2021 (Accessed: 17 July 2026).

OpenStreetMap contributors (2026) OpenStreetMap. Available at: https://www.openstreetmap.org (Accessed: 17 July 2026).

Sherman, L., Neyroud, P. and Neyroud, E. (2016) 'The Cambridge Crime Harm Index: measuring total harm from crime based on sentencing guidelines', Policing: A Journal of Policy and Practice, 10(3), pp. 171–183.

Single Online Home National Digital Team (2026) About data.police.uk. Available at: https://data.police.uk/about/ (Accessed: 17 July 2026).

Tobler, W. (1970) 'A computer movie simulating urban growth in the Detroit region', Economic Geography, 46(sup1), pp. 234–240.
