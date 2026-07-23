**Module:** ECS7036P
**Group:** 8

**# AI Approach To Running Route Optimisation For Safety**

# Data Collection and Cleaning
**Script:** `run_data_pipeline.py`

## Purpose

This script builds a street level dataset for a running route safety model. Crime, stop and search, and lighting are each turned into edge level features, one row per street segment.

## Data Sources

The street network is obtained from OpenStreetMap through OSMnx, using the pedestrian network type. Crime data and stop and search data are both obtained from the UK Police open data API, retrieved by month. Street lighting data is obtained from Camden Council's open data portal.

## Date Range

Requests from January 2021 to June 2026. data.police.uk only keeps a rolling ~3 year window, so `resolve_months()` checks what is actually published each run and clips the request accordingly, printing a note if the start date had to move. Earliest currently available: 2023-06. Console output states exactly which months are included.

## Crime Weightings

`crime_severity` is harm-based (1 to 3), from Sherman, Neyroud and Neyroud (2016) 'The Cambridge Crime Harm Index', Policing: A Journal of Policy and Practice, 10(3), and the ONS (2016) Crime Severity Score methodology, both weighted by sentencing severity.

`crime_perceived_risk` is a separate new column (1 to 4), how unsafe a crime makes an area feel, not how harmful it legally is. Based on Innes (2004) 'Signal crimes and signal disorders', British Journal of Sociology, 55(3), and Innes and Fielding (2002), Sociological Research Online, 7(2): the Signal Crimes Perspective, visible disorder like anti-social behaviour disproportionately drives fear of crime independent of its statistical severity. Also informed by ONS Crime Survey for England and Wales (CSEW) perception/ASB releases. Under this column, anti-social behaviour (3) scores above burglary (2), the reverse of the severity ordering, deliberately.

Both are coarse simplifications onto police.uk's 14 categories, and should be cited as such rather than as a reproduction of either study's exact weights.

## Location Anonymisation and Average Spacing

Crime/stop-search coordinates are not exact; each is snapped to the nearest of roughly 680,000 anonymous map points (mostly street centres), each with a catchment of at least 8 postal addresses. No official average spacing figure is published; our own Camden data gives a rough estimate, about 1,384 distinct snap points across Camden's ~21.8 sq km in one 6-month pull, roughly one point every ~125m as an upper bound.

## Stop and Search

Type, object of search, outcome, and datetime are kept. Gender, age range, and ethnicity fields from the raw API are deliberately excluded, sensitive personal data with no legitimate role in a safety score.

## Monthly Breakdown

`edge_features.csv` carries both an overall total per edge (`crime_count`, `crime_severity_sum`, `crime_perceived_risk_sum`, `stop_search_count`, summed across the full resolved date range) and a per-month breakdown as separate columns, e.g. `crime_count_2024_01`. Lamps are a live snapshot with no time dimension, so are not split by month.

## Spatial Lag (Neighbouring Streets)

<col>_lag_sum / <col>_lag_mean give the total and average of that column across an edge's directly neighbouring edges (edges sharing a junction node), computed for the four overall totals only. This is a spatial lag: it aggregates over nearby streets within the same month. It is distinct from the temporal lag used in the ML/NN pipeline section below (e.g. crime_count_lag_1), which aggregates over previous months for the same street. Both use the term "lag", but along different axes, spatial versus temporal, and should not be treated as the same type of column.

## Network KDE (NetKDE) Smoothing

The nearest-edge counts above assign each crime to a single street. `edge_features.csv` also distributes each crime's weight across nearby streets using Network KDE (NetKDE). Closer crimes contribute more, and distance is measured along the walkable street graph using Dijkstra shortest paths rather than in a straight line, so a crime's influence spreads along streets a runner could actually take, not through a wall, a back garden or across the canal. This is a simplified Kernel Density Estimation (KDE), the standard technique in crime hotspot mapping (Chainey, Tompson and Uhlig, 2008), adapted here to use network distance instead of straight-line distance.

The smoothing is applied separately within each month, so no future month can leak backwards into an earlier one; pooling all months together first would let a crime from 2026 bleed into a street's 2023 figures. The columns are named `{metric}_netkde_{year}_{month}`, e.g. `crime_count_netkde_2024_01`, matching the monthly breakdown naming so the downstream wide-to-long reshape picks them up automatically. Bandwidth defaults to 125m, matching the estimated spacing of data.police.uk's anonymised points in Camden (see Location Anonymisation above); this can be adjusted via `NETKDE_BANDWIDTH_M` in the script. The street graph is built once and reused for every month, rather than rebuilt separately for each one.

## Lamp Features Kept Separate

Lamp features (\lamp_count`, `lamp_per_km`, `is_lit`) are kept in their own `edge_lamp_features.csv`/`.gpkg` file, separate from `edge_features.csv`, and can be joined back on `edge_id`. Note: no OSM edge in Camden carries a `lit` tag, so `is_lit` here depends entirely on whether a lamp snapped to that edge.

## Cleaning Procedure

Records with missing coordinates are removed. Coordinate fields are converted to numeric values, and records that fail conversion are removed. Exact duplicate rows are removed. All coordinates are reprojected to British National Grid to enable distance calculations in metres.

## Street Network References

`osmid` is the true OpenStreetMap way ID for that street (comma-joined if OSMnx merged several OSM ways into one edge), so any row can be looked up directly on openstreetmap.org. u and v are OpenStreetMap's internal node IDs, not human-readable on their own. u_lat, u_lng, v_lat, v_lng sit immediately to their right, giving each endpoint's plain WGS84 coordinates. All four (`osmid`, `u`, `v`, plus the lat/lng pairs) are included in both `edge_features` and `edge_lamp_features`.

## Outputs

Seven files in `data_out`: `crime_points.gpkg`, `stop_search_points.gpkg`, `lamp_points.gpkg`, `edge_features.gpkg`, `edge_features.csv`, `edge_lamp_features.gpkg`, `edge_lamp_features.csv`.

## Requirements

Python with `osmnx`, `geopandas`, `shapely`, `requests`, `pandas`, `pyproj`, and an internet connection to OpenStreetMap, the police API, and Camden's open data portal.

# ML/NN pipeline

## Running the pipeline 
Notebook.ipynb is run monthly in order to train the model, process and output the safety scores to osm_safety_tags.csv for front end intake. 

## Purpose
Processes historical street-level crime and stop-and-search features, trains a machine learning model to forecast future perceived crime risk, and transforms those predictions into physical routing penalties for integration with pedestrian routing engines.

## Outputs
One file in the output directory: osm_safety_tags.csv. This contains only the edge_id, osmid, is_lit (binary 0 or 1 value) and the safety_tag_value, providing a clean mapping table to inject custom penalty tags (e.g., safety_cost=8) into the raw .osm.pbf map file before building the Valhalla routing graph. Node-level aggregations are excluded as Valhalla requires continuous edge penalties to prevent routing down the length of dangerous streets.

# Running the UI (nightrunner-ui)

A safety and street-lighting-aware pedestrian routing tool built on [Valhalla](https://github.com/valhalla/valhalla) for the Camden borough, in the `nightrunner-ui/` folder.

## Requirement

Docker Desktop must be installed and running first: https://www.docker.com/products/docker-desktop/. Everything else (Python, Node) runs inside it automatically. Nothing else to install.

## Setup

**Mac:** double-click `nightrunner-ui/setup.command`. First time, macOS will block it as from an "unidentified developer". Right-click it, choose Open, then Open again in the dialog. Only needed once.

**Windows:** download `NightRunnerSetup.exe` from the Releases page and double-click it (no Python needed). First time, Windows may warn "Windows protected your PC" (SmartScreen). Click More info, then Run anyway. Alternatively, double-click `nightrunner-ui/setup.bat` if Python is already installed.

The script checks Docker, builds and starts everything, then opens two browser tabs once ready:
- http://localhost:3000 - the routing app itself (pick start/end points, get a safety and lighting-weighted route)
- http://localhost:8080 - a debug/QA map of the underlying model's output: streets above the safety-risk baseline, and streets confirmed unlit, as two independently toggleable layers

Safe to re-run any time.

## If it doesn't work

Make sure Docker Desktop is actually open and ready, and check nothing else is using ports 3000, 5050, 8002, or 8080. For anything else, run `docker compose logs` inside `nightrunner-ui/safe-routing`.


---

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

Valhalla contributors (2026) Valhalla. Available at: https://github.com/valhalla/valhalla (Accessed: 21 July 2026).

Schneider, W. (2026) BBBike extracts. Available at: https://extract.bbbike.org (Accessed: 21 July 2026).
