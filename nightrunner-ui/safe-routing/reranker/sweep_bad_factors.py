"""
Borough-wide version of find_bad_factors.py.

find_bad_factors.py only proves that shapes near ONE hand-picked route are
edge-walk-matchable - since Valhalla only edge-walk-matches shapes near the
route actually being computed, a trim built from a single location pair
says nothing about shapes elsewhere in Camden. This script instead
recursively splits Camden's overall bounding box into a quadtree of cells
(auto-subdividing any cell with too many entries - testing thousands of
shapes against Valhalla in one request is known to crash it outright, not
fail gracefully), and bisects each cell's own entries using two of that
cell's own real shape coordinates as test locations. That guarantees every
entry in the file gets edge-walk-tested at least once, near its own actual
location, rather than only the ones that happen to lie on whatever route
you tested by hand.

Caveat, to be upfront about it: picking two points from within a cell and
routing between them is a heuristic, not a formal guarantee every entry in
that cell sits on the resulting path - Valhalla can still route around
some of them. It's a large improvement over testing one single route
across the whole borough, not a mathematical certainty of full coverage.

safety_factors.json entries carry separate safety_factor/lighting_factor
fields (see app.py) - this script combines them into Valhalla's expected
"factor" key itself before posting, using the lighting-penalty-active case
(equivalent to is_daylight=False), same as find_bad_factors.py.

This is meant to be run ONCE per meaningful change to the underlying
safety/lighting CSVs, against a FRESH, untrimmed safety_factors.json
(regenerate with build_safety_factors.py first, so this sweep sees, and
can test, every entry - not just whatever a previous find_bad_factors.py
run already removed). It is much slower than find_bad_factors.py - expect
well over 10 minutes for the full borough, depending on how many bad
shapes exist and how MAX_ENTRIES_PER_CELL is tuned.

The resulting safety_factors.json + sweep_excluded_shapes.json should then
be committed, exactly like find_bad_factors.py's output - see
README_valhalla_routing.md for why this matters (regenerating
safety_factors.json from scratch wipes any trim, silently).

Run from your reranker/ folder:
    python3 sweep_bad_factors.py
"""
import json
import time
from datetime import datetime, timezone

import httpx

from polyline6 import decode

VALHALLA_URL = "http://localhost:8002"
TIMEOUT_S = 180.0  # generous - larger cells can be slow to edge-walk-match

# Any cell with more entries than this gets recursively split into 4
# smaller cells instead of being tested directly. NOT empirically tuned
# against this exact dataset in this session - chosen to stay comfortably
# under the point at which testing the full ~30k+ entry set crashes
# Valhalla outright. If you see Valhalla itself crash (rather than a clean
# non-200 response) partway through a sweep, lower this and re-run.
MAX_ENTRIES_PER_CELL = 3000

# Safety valve on recursion depth, in case a cluster of entries can't be
# separated below MAX_ENTRIES_PER_CELL by lat/lon splitting alone (e.g.
# many entries stacked at near-identical coordinates) - stops infinite
# subdivision. At depth 12 a starting bbox is split into up to 4**12 cells,
# which should separate any realistic dataset well before this is hit.
MAX_DEPTH = 12


def _combined_factor(entry: dict) -> float:
    """safety_factor * lighting_factor, i.e. the lighting-penalty-active case."""
    return round(entry["safety_factor"] * entry["lighting_factor"], 3)


def _shape_bbox_and_points(entry: dict):
    points = decode(entry["shape"])
    if not points:
        return None
    lats = [p[0] for p in points]
    lons = [p[1] for p in points]
    return (min(lats), max(lats), min(lons), max(lons)), points


def _overall_bbox(entries_with_points):
    lats, lons = [], []
    for _entry, (bbox, _points) in entries_with_points:
        lats.extend([bbox[0], bbox[1]])
        lons.extend([bbox[2], bbox[3]])
    return (min(lats), max(lats), min(lons), max(lons))


def _midpoint(bbox):
    min_lat, max_lat, min_lon, max_lon = bbox
    return ((min_lat + max_lat) / 2, (min_lon + max_lon) / 2)


def _test_locations_for_cell(cell_entries_with_points):
    """
    Two real, on-graph coordinates drawn from this cell's own decoded shape
    points (never arbitrary bbox corners, which could land inside a
    building or park with nothing routable nearby). Picks the point with
    the smallest lat+lon sum and the point with the largest as a cheap
    proxy for "opposite corners of the cell", to bias the test route
    toward spanning the cell's full extent rather than two arbitrarily
    close points - see the module docstring for why this is a heuristic,
    not a guarantee.
    """
    all_points = [
        p
        for _entry, (_bbox, points) in cell_entries_with_points
        for p in points
    ]
    origin = min(all_points, key=lambda p: p[0] + p[1])
    dest = max(all_points, key=lambda p: p[0] + p[1])
    return [
        {"lat": origin[0], "lon": origin[1]},
        {"lat": dest[0], "lon": dest[1]},
    ]


def test_entries(entries_with_points, test_locations):
    """Returns True if this set of entries causes ANY non-200 Valhalla response."""
    valhalla_factors = [
        {"shape": entry["shape"], "factor": _combined_factor(entry)}
        for entry, _ in entries_with_points
    ]
    payload = {
        "locations": test_locations,
        "costing": "pedestrian",
        "linear_cost_factors": valhalla_factors,
    }
    t0 = time.time()
    r = httpx.post(f"{VALHALLA_URL}/route", json=payload, timeout=TIMEOUT_S)
    print(f"      tested {len(entries_with_points)} entries in {time.time()-t0:.1f}s -> status {r.status_code}")
    return r.status_code != 200


def bisect(entries_with_points, test_locations, bad_entries):
    if not entries_with_points:
        return
    if len(entries_with_points) == 1:
        bad_entries.append(entries_with_points[0][0])
        return
    mid = len(entries_with_points) // 2
    left, right = entries_with_points[:mid], entries_with_points[mid:]
    if test_entries(left, test_locations):
        bisect(left, test_locations, bad_entries)
    if test_entries(right, test_locations):
        bisect(right, test_locations, bad_entries)


def sweep_cell(bbox, entries_with_points, bad_entries, cell_stats, depth=0):
    if not entries_with_points:
        return

    label = f"depth={depth} n={len(entries_with_points)} bbox={tuple(round(v, 4) for v in bbox)}"

    if len(entries_with_points) > MAX_ENTRIES_PER_CELL and depth < MAX_DEPTH:
        min_lat, max_lat, min_lon, max_lon = bbox
        mid_lat, mid_lon = (min_lat + max_lat) / 2, (min_lon + max_lon) / 2
        quadrants = [
            (min_lat, mid_lat, min_lon, mid_lon),
            (min_lat, mid_lat, mid_lon, max_lon),
            (mid_lat, max_lat, min_lon, mid_lon),
            (mid_lat, max_lat, mid_lon, max_lon),
        ]
        buckets = {q: [] for q in quadrants}
        for item in entries_with_points:
            _entry, (ebbox, _points) = item
            mlat, mlon = _midpoint(ebbox)
            placed = False
            for q in quadrants:
                qlat0, qlat1, qlon0, qlon1 = q
                if qlat0 <= mlat <= qlat1 and qlon0 <= mlon <= qlon1:
                    buckets[q].append(item)
                    placed = True
                    break
            if not placed:
                # Float edge case - shouldn't happen since mid_lat/mid_lon
                # are exactly the bbox midpoint, but don't silently drop
                # an entry if it does.
                buckets[quadrants[0]].append(item)
        print(f"  {label} -> splitting")
        for q, items in buckets.items():
            sweep_cell(q, items, bad_entries, cell_stats, depth + 1)
        return

    if len(entries_with_points) > MAX_ENTRIES_PER_CELL:
        print(f"  {label} -> WARNING: still over MAX_ENTRIES_PER_CELL at MAX_DEPTH, testing anyway")
    else:
        print(f"  {label} -> testing directly")

    test_locations = _test_locations_for_cell(entries_with_points)
    bad_before = len(bad_entries)
    if test_entries(entries_with_points, test_locations):
        bisect(entries_with_points, test_locations, bad_entries)
    cell_stats.append({
        "bbox": bbox,
        "depth": depth,
        "entry_count": len(entries_with_points),
        "test_locations": test_locations,
        "bad_count": len(bad_entries) - bad_before,
    })


def main():
    all_entries = json.load(open("safety_factors.json"))
    print(f"Sweeping {len(all_entries)} entries across Camden...")

    entries_with_points = []
    skipped_undecodable = 0
    for entry in all_entries:
        result = _shape_bbox_and_points(entry)
        if result is None:
            skipped_undecodable += 1
            continue
        entries_with_points.append((entry, result))
    if skipped_undecodable:
        print(f"  (skipped {skipped_undecodable} entries with empty/undecodable shapes)")

    overall_bbox = _overall_bbox(entries_with_points)
    print(f"Overall bounding box: {overall_bbox}")

    bad_entries = []
    cell_stats = []
    sweep_cell(overall_bbox, entries_with_points, bad_entries, cell_stats)

    print(f"\nFound {len(bad_entries)} problematic entrie(s) across {len(cell_stats)} tested cell(s).")

    bad_shapes = {e["shape"] for e in bad_entries}
    good_entries = [e for e in all_entries if e["shape"] not in bad_shapes]

    with open("safety_factors.json", "w") as f:
        json.dump(good_entries, f)
    print(f"Wrote {len(good_entries)} good entries back to safety_factors.json "
          f"(excluded {len(bad_entries)} bad ones).")

    with open("sweep_excluded_shapes.json", "w") as f:
        json.dump({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "overall_bbox": overall_bbox,
            "max_entries_per_cell": MAX_ENTRIES_PER_CELL,
            "total_entries_tested": len(all_entries),
            "excluded_count": len(bad_entries),
            "cells_tested": len(cell_stats),
            "reason": "Non-200 Valhalla /route response (edge-walk match failure) "
                      "when this shape was included in linear_cost_factors, found "
                      "via borough-wide grid sweep rather than a single test route.",
            "cell_stats": cell_stats,
            "excluded_entries": bad_entries,
        }, f, indent=2)
    print("Wrote sweep report to sweep_excluded_shapes.json "
          "(record only - not read by any code).")

    print("Restart the reranker (docker compose restart reranker) to pick this up.")


if __name__ == "__main__":
    main()
