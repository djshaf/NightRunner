"""
Finds which entries in safety_factors.json cause Valhalla's error 233
("Failed to edge walk line feature") by testing directly against your
local Valhalla instance (bypassing the reranker entirely).

Uses binary search: splits the list in half, tests each half, and keeps
recursing into whichever half(s) still fail - much faster than testing
all entries one by one. Note the FIRST test (the full, unfiltered set)
is the slowest call by far, since Valhalla has to edge-walk-match every
submitted shape; each half tested afterwards has progressively less work
to do, so the search speeds up quickly after the first call.

Run from your reranker/ folder (needs a working pair of Camden locations -
adjust TEST_LOCATIONS below if these don't work for you):
    python3 find_bad_factors.py
"""
import json
import time

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


def test_factors(factors):
    """Returns True if this set of factors causes a 233 error, False if fine."""
    payload = {
        "locations": TEST_LOCATIONS,
        "costing": "pedestrian",
        "linear_cost_factors": factors,
    }
    t0 = time.time()
    r = httpx.post(f"{VALHALLA_URL}/route", json=payload, timeout=TIMEOUT_S)
    print(f"    tested {len(factors)} entries in {time.time()-t0:.1f}s -> status {r.status_code}")
    if r.status_code == 400 and r.json().get("error_code") == 233:
        return True
    return False


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
    print("Restart the reranker (docker compose restart reranker) to pick this up.")


if __name__ == "__main__":
    main()
