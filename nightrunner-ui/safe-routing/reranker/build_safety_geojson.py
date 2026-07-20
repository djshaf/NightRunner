"""
Generates safety_edges.geojson - every edge your safety model flagged
above the baseline (tag > 1), with its coordinates and risk info attached,
for visualizing on a map. Reuses the same data loading as
build_safety_factors.py, so if you update load_scored_edges() there,
re-run this too.

Run after build_safety_factors.py:
    python build_safety_geojson.py
"""
import json

from build_safety_factors import load_scored_edges, score_to_factor

# Colour ramp: baseline-adjacent (tag=2) -> lighter orange, worst (tag=11) -> dark red
TAG_COLORS = {
    2: "#FFD580", 3: "#FFB84D", 4: "#FF9900", 5: "#FF6600",
    6: "#FF3300", 7: "#E60000", 8: "#CC0000", 9: "#B30000",
    10: "#990000", 11: "#660000",
}


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

    geojson = {"type": "FeatureCollection", "features": features}
    with open("safety_edges.geojson", "w") as f:
        json.dump(geojson, f)

    edge_count = sum(1 for f in features if f["properties"]["feature_role"] == "edge")
    print(f"Wrote {edge_count} risky edges (+ midpoint markers) to safety_edges.geojson")


if __name__ == "__main__":
    build()
