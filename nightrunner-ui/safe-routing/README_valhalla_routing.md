# Safety-optimised pedestrian routing on Valhalla

## What's here
- `docker-compose.yml` — runs Valhalla, the safety-factor proxy, and the
  web-app frontend together.
- `reranker/` — the proxy service:
  - `polyline6.py` — encode/decode for Valhalla's 6-digit-precision
    polylines (round-trip tested).
  - `build_safety_factors.py` — turns your safety model's output into
    `safety_factors.json`, the payload Valhalla actually consumes. **You
    need to fill in `load_scored_edges()` with your real data loading** -
    I haven't invented a format for your model's output since I don't have
    it in front of me.
  - `app.py` — the actual proxy. Injects `safety_factors.json` into every
    `/route` call and forwards everything else untouched.

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

3. **Generate your safety factors** (after filling in `load_scored_edges()`):
   ```
   cd reranker
   pip install -r requirements.txt
   python build_safety_factors.py
   cd ..
   ```

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
   - Safety-factor proxy: http://localhost:5000
   - Frontend: http://localhost:3000

6. **Verify the safety weighting is actually doing something**: pick two
   points either side of a street your model scored as unsafe, request a
   route, and confirm it detours. If it doesn't, your `K` value in
   `build_safety_factors.py` is probably too low, or the edge geometry
   isn't matching (check Valhalla's logs — failed edge-walk matches are
   usually visible there).

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
