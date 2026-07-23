# Safety-optimised pedestrian routing on Valhalla

## What's here
- `docker-compose.yml` — runs Valhalla, the safety-factor proxy, and the
  web-app frontend together.
- `reranker/` — the proxy service:
  - `polyline6.py` — encode/decode for Valhalla's 6-digit-precision
    polylines (round-trip tested).
  - `build_safety_factors.py` — joins `edge_features.csv`,
    `osm_safety_tags.csv`, and `edge_lamp_features.csv` on `osmid` and
    writes `safety_factors.json`: one `{shape, safety_factor,
    lighting_factor}` entry per edge that needs either penalty. Missing
    safety-tag or lamp data defaults to "safe"/"lit" (no penalty) rather
    than being skipped.
  - `app.py` — the actual proxy. Combines each entry's `safety_factor` and
    `lighting_factor` into the single `factor` Valhalla expects (skipping
    the lighting half if the request sets `is_daylight: true`, default
    `false`), injects the result into every `/route` call's
    `linear_cost_factors`, and forwards everything else untouched.
  - `find_bad_factors.py` — binary-searches `safety_factors.json` for
    entries that make Valhalla's edge-walk matching fail (any non-200
    `/route` response), removes them, and records what it removed in
    `excluded_shapes.json`.

## `safety_factors.json` is already generated and trimmed - don't casually regenerate it
This repo ships a `safety_factors.json` that's already been built from the
committed CSVs **and** had every edge-walk-failing shape removed via
`find_bad_factors.py` (removing bad shapes isn't optional here — as of this
writing, ~46% of raw entries fail edge-walk matching, likely because
`build_safety_factors.py` currently builds each shape from a straight
2-point line between edge endpoints rather than the real multi-point curve
already sitting in `edge_features.csv`'s `polyline6` column — a known,
not-yet-fixed issue).

**Do not run `build_safety_factors.py` again unless the underlying CSVs
actually change.** Regenerating it from scratch silently wipes the trim and
reintroduces every bad shape, with no warning. If you do need to regenerate
it (e.g. updated safety/lighting data), you MUST re-run
`find_bad_factors.py` afterward - it currently takes on the order of 10+
minutes against the full Camden dataset - before restarting the reranker.
`excluded_shapes.json` (written by that script) is a plain record of what
was excluded and why; nothing reads it back in.

## What I can't do from here
I'm running in a sandboxed container, not on your laptop, so I can't
literally start these containers on `/Users/raenafadzil/...`. Everything
above is written and syntax-checked, but you'll need to run it locally.

## Steps to actually run this

1. **Get an OSM extract for your borough** (a `.osm.pbf` file) — e.g. from
   [Geofabrik](https://download.geofabrik.de/) or the
   [BBBike extract tool](https://extract.bbbike.org/) for a tighter
   boundary. Put it in `./valhalla_data/`.

2. **Clone the frontend as a sibling folder**:
   ```
   git clone https://github.com/valhalla/web-app ../web-app
   ```

3. **Safety/lighting factors are already committed** - `safety_factors.json`
   is in `reranker/` and ready to use as-is. Skip straight to step 4 unless
   you've actually changed `edge_features.csv`, `osm_safety_tags.csv`, or
   `edge_lamp_features.csv`, in which case see the warning above before
   regenerating.

4. **Set `VITE_CENTER_COORDS`** in `docker-compose.yml` to your borough's
   centre point, and update the `tile_urls` environment variable (or drop
   your `.pbf` filename directly) for the `valhalla` service.

5. **Bring it all up**:
   ```
   docker compose up --build
   ```
   First boot will take a while — Valhalla builds graph tiles from your
   `.pbf` on first start. Once running:
   - Valhalla API: http://localhost:8002
   - Safety-factor proxy: http://localhost:5050
   - Frontend: http://localhost:3000

6. **Verify the safety/lighting weighting is actually doing something**:
   pick two points either side of a street flagged as unsafe or unlit,
   request a route through the proxy (port 5050, not 8002 directly), and
   confirm it detours. If it doesn't, `SAFETY_SCALE`/`LIGHTING_SCALE` in
   `build_safety_factors.py` may be too low, or the edge geometry isn't
   matching (check `docker compose logs reranker` and Valhalla's own logs —
   failed edge-walk matches show up as non-200 `/route` responses).

## Things I flagged as unverified, worth double-checking yourself
- The exact numeric bounds Valhalla enforces on `factor` in
  `linear_cost_factors` specifically (I confirmed the parameter and its
  general behaviour from the current API docs, but not a hard min/max for
  this exact field).
- Whether there's a practical limit on how many `linear_cost_factors`
  entries one request can carry before latency or request size becomes a
  problem for a full borough's worth of edges — test with your real data
  volume.
- The `docker/README.md` in your already-downloaded `valhalla/valhalla`
  repo is the authoritative source for the scripted image's environment
  variables if anything above needs adjusting for your Valhalla version.
