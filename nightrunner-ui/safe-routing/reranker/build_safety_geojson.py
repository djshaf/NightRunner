"""
Generates safety_edges.geojson for visualizing the model's output on a map
(safety_map.html): every edge flagged above the safety baseline (tag > 1),
PLUS every edge with confirmed unlit lighting data - both layers in one
file, distinguished by their "feature_role" property.

Reuses the same data loading as build_safety_factors.py for safety
(load_scored_edges()) and lighting_score.py for lighting
(load_unlit_edges()) - if you update either of those, re-run this too.

Run after build_safety_factors.py:
    python build_safety_geojson.py
"""
import json

from build_safety_factors import load_scored_edges, score_to_factor
from lighting_score import load_unlit_edges

# Colour ramp: baseline-adjacent (tag=2) -> lighter orange, worst (tag=11) -> dark red
TAG_COLORS = {
    2: "#FFD580", 3: "#FFB84D", 4: "#FF9900", 5: "#FF6600",
    6: "#FF3300", 7: "#E60000", 8: "#CC0000", 9: "#B30000",
    10: "#990000", 11: "#660000",
}

# Distinct from the red/orange safety ramp above, so the two layers read
# clearly together on the same map.
UNLIT_COLOR = "#1f3d99"


def build() -> None:
    features = []
    for coords, tag, osmid in load_scored_edges():
        if tag <= 1:
            continue  # skip the ~99% baseline edges - not what we're highlighting
        # GeoJSON wants [lon, lat] order, coords are stored as (lat, lon)
        lonlat = [[lon, lat] for lat, lon in coords]
        props = {
            "osmid": osmid,
            "safety_tag_value": tag,
            "cost_factor": round(score_to_factor(tag), 2),
            "color": TAG_COLORS.get(tag, "#660000"),
        }
        features.append({
            "type": "Feature",
            "properties": {**props, "feature_role": "edge"},
            "geometry": {"type": "LineString", "coordinates": lonlat},
        })
        # Midpoint marker so short (few-metre) edges stay visible when
        # zoomed out - a thin line alone can vanish at normal zoom levels.
        mid = lonlat[len(lonlat) // 2]
        features.append({
            "type": "Feature",
            "properties": {**props, "feature_role": "midpoint"},
            "geometry": {"type": "Point", "coordinates": mid},
        })

    for coords, osmid in load_unlit_edges():
        lonlat = [[lon, lat] for lat, lon in coords]
        props = {
            "osmid": osmid,
            "lit": False,
            "color": UNLIT_COLOR,
        }
        features.append({
            "type": "Feature",
            "properties": {**props, "feature_role": "lighting_edge"},
            "geometry": {"type": "LineString", "coordinates": lonlat},
        })
        mid = lonlat[len(lonlat) // 2]
        features.append({
            "type": "Feature",
            "properties": {**props, "feature_role": "lighting_midpoint"},
            "geometry": {"type": "Point", "coordinates": mid},
        })

    geojson = {"type": "FeatureCollection", "features": features}
    with open("safety_edges.geojson", "w") as f:
        json.dump(geojson, f)

    edge_count = sum(1 for f in features if f["properties"]["feature_role"] == "edge")
    lighting_count = sum(1 for f in features if f["properties"]["feature_role"] == "lighting_edge")
    print(
        f"Wrote {edge_count} risky edges + {lighting_count} unlit edges "
        f"(+ midpoint markers) to safety_edges.geojson"
    )


if __name__ == "__main__":
    build()
