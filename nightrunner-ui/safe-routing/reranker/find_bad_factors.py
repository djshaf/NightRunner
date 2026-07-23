"""
Finds which entries in safety_factors.json cause Valhalla /route requests
to fail (e.g. error 233 "Failed to edge walk line feature", or error 499
"Could not find destination candidate in shape-walked path" - both stem
from the same underlying cause, bad shapes in our injected factors) by
testing directly against your local Valhalla instance (bypassing the
reranker entirely). Any non-200 response is treated as "bad", not just
233 specifically - we've seen other error codes come from the same root
cause and a narrower check would silently miss them.

Uses binary search: splits the list in half, tests each half, and keeps
recursing into whichever half(s) still fail - much faster than testing
all entries one by one. Note the FIRST test (the full, unfiltered set)
is the slowest call by far, since Valhalla has to edge-walk-match every
submitted shape; each half tested afterwards has progressively less work
to do, so the search speeds up quickly after the first call.

safety_factors.json entries carry separate safety_factor/lighting_factor
fields (app.py combines them into Valhalla's expected "factor" key at
request time, so the daylight toggle can skip the lighting half without a
rebuild). This script bypasses app.py entirely, so it does that same
combination itself in _combined_factor() before posting to Valhalla -
using the lighting-penalty-active case (equivalent to is_daylight=False),
since that's the default/safety-conscious behaviour and the more
conservative case to shape-test against. The bisection and the final
good_entries write-back still operate on the ORIGINAL two-field entries,
so the file's schema stays exactly what app.py expects.

Also writes excluded_shapes.json alongside safety_factors.json - a record
of exactly which entries were removed and why, purely for documentation/
reproducibility (nothing reads this file back in; it's not consumed by
app.py or anything else). This matters because trimming is NOT persistent:
re-running build_safety_factors.py from the raw CSVs regenerates
safety_factors.json from scratch and silently reintroduces every bad shape
this script removed - excluded_shapes.json is the paper trail for what got
cut and why, in case that ever needs re-deriving or explaining.

Run from your reranker/ folder (needs a working pair of Camden locations -
adjust TEST_LOCATIONS below if these don't work for you):
    python3 find_bad_factors.py
"""
import json
import time
from datetime import datetime, timezone

import httpx

VALHALLA_URL = "http://localhost:8002"
TIMEOUT_S = 180.0  # generous - the full unfiltered set can be slow to edge-walk-match

# The specific route that's actually failing (from the real curl test) -
# using generic/default coordinates elsewhere in Camden did NOT reproduce
# the error, suggesting Valhalla only edge-walk-matches shapes near the
# route actually being computed, not the whole submitted list regardless
# of location. Testing with the real failing coordinates is what matters.
TEST_LOCATIONS = [
    {"lat": 51.5392533, "lon": -0.142757},
    {"lat": 51.5387612, "lon": -0.1377894},
]


def _combined_factor(entry: dict) -> float:
    """safety_factor * lighting_factor, i.e. the lighting-penalty-active case."""
    return round(entry["safety_factor"] * entry["lighting_factor"], 3)


def test_factors(factors):
    """Returns True if this set of factors causes ANY non-200 response, False if fine."""
    valhalla_factors = [
        {"shape": e["shape"], "factor": _combined_factor(e)}
        for e in factors
    ]
    payload = {
        "locations": TEST_LOCATIONS,
        "costing": "pedestrian",
        "linear_cost_factors": valhalla_factors,
    }
    t0 = time.time()
    r = httpx.post(f"{VALHALLA_URL}/route", json=payload, timeout=TIMEOUT_S)
    print(f"    tested {len(factors)} entries in {time.time()-t0:.1f}s -> status {r.status_code}")
    return r.status_code != 200


def bisect(factors, bad_entries):
    if not factors:
        return
    if len(factors) == 1:
        bad_entries.append(factors[0])
        return
    mid = len(factors) // 2
    left, right = factors[:mid], factors[mid:]
    if test_factors(left):
        bisect(left, bad_entries)
    if test_factors(right):
        bisect(right, bad_entries)


def main():
    all_factors = json.load(open("safety_factors.json"))
    print(f"Testing {len(all_factors)} entries against {VALHALLA_URL}...")

    if not test_factors(all_factors):
        print("No error with the full set right now - problem may be intermittent, "
              "or your Valhalla container was restarted/rebuilt since the last failure.")
        return

    bad_entries = []
    bisect(all_factors, bad_entries)

    print(f"\nFound {len(bad_entries)} problematic entrie(s):")
    for e in bad_entries:
        print(e)

    good_entries = [f for f in all_factors if f not in bad_entries]
    with open("safety_factors.json", "w") as f:
        json.dump(good_entries, f)
    print(f"\nWrote {len(good_entries)} good entries back to safety_factors.json "
          f"(excluded {len(bad_entries)} bad ones).")

    with open("excluded_shapes.json", "w") as f:
        json.dump({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "test_locations": TEST_LOCATIONS,
            "total_entries_tested": len(all_factors),
            "excluded_count": len(bad_entries),
            "reason": "Non-200 Valhalla /route response (edge-walk match failure, "
                      "e.g. error 233 or 499) when this shape was included in "
                      "linear_cost_factors.",
            "excluded_entries": bad_entries,
        }, f, indent=2)
    print(f"Wrote {len(bad_entries)} excluded entries to excluded_shapes.json "
          "(record only - not read by any code).")

    print("Restart the reranker (docker compose restart reranker) to pick this up.")


if __name__ == "__main__":
    main()
