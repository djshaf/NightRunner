"""
Chooses which route (main or one of the alternates) should be recommended
to the user, and reorders the response so that route becomes "main".

Rule (see conversation with the project owner for the reasoning):
  1. Start with the fastest route.
  2. If its safety_score is already >= SAFETY_THRESHOLD, recommend it as-is.
  3. Otherwise, look only at routes within TIME_BUDGET_MULTIPLIER extra time
     of the fastest, and pick whichever of those has the highest safety_score.
  4. If nothing qualifies (shouldn't normally happen, since the fastest
     route always qualifies for its own budget), fall back to the fastest
     route anyway - never leave the response without a recommendation.
"""
from typing import List

SAFETY_THRESHOLD = 90.0
TIME_BUDGET_MULTIPLIER = 1.2


def _route_time(trip: dict) -> float:
    return trip.get("summary", {}).get("time", float("inf"))


def _route_safety(trip: dict) -> float:
    # Missing safety_score (e.g. annotation failed) is treated as worst-case
    # so it's never preferred over a route we actually could score.
    return trip.get("summary", {}).get("safety_score", -1.0)


def choose_recommended_index(trips: List[dict]) -> int:
    """
    trips[0] is always the current main route, trips[1:] are alternates,
    in their original order. Returns the index of the route that should
    become the new main route.
    """
    if not trips:
        return 0

    fastest_idx = min(range(len(trips)), key=lambda i: _route_time(trips[i]))
    fastest_time = _route_time(trips[fastest_idx])
    fastest_safety = _route_safety(trips[fastest_idx])

    if fastest_safety >= SAFETY_THRESHOLD:
        return fastest_idx

    time_budget = fastest_time * TIME_BUDGET_MULTIPLIER
    candidates = [
        i for i in range(len(trips))
        if _route_time(trips[i]) <= time_budget
    ]

    if not candidates:
        return fastest_idx  # shouldn't happen - fastest always fits its own budget

    return max(candidates, key=lambda i: _route_safety(trips[i]))


def apply_recommendation(body: dict) -> None:
    """
    Mutates a Valhalla-shaped response dict in place: reorders so the
    recommended route becomes body["trip"], with the rest as alternates
    in their original relative order.
    """
    if "trip" not in body:
        return

    alternates = body.get("alternates", [])
    trips = [body["trip"]] + [alt["trip"] for alt in alternates if "trip" in alt]

    recommended_idx = choose_recommended_index(trips)

    if recommended_idx == 0:
        return  # fastest/main route was already the recommendation - nothing to reorder

    # Swap: the recommended trip becomes the new main "trip"; the old main
    # trip takes its place in the alternates list (position preserved for
    # everything else).
    new_main = trips[recommended_idx]
    old_main = trips[0]

    alt_list_idx = recommended_idx - 1  # trips[1:] maps to alternates[0:]
    body["trip"] = new_main
    alternates[alt_list_idx]["trip"] = old_main
    body["alternates"] = alternates
