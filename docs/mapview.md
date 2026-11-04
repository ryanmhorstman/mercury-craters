# Map viewer

A local, zoomable map of the whole planet: the MESSENGER MDIS basemap with the
thesis catalog and the global CNN run (`detectors.cnn.model_name`) drawn on top.

```
python -m craters.mapview        # then open http://localhost:8766
```

Needs `data/interim/basemap_166m_cog.tif` (with overviews; the COG made in
Phase 0 has them), `data/detections/global_<model>.parquet`
(`python -m craters.phase3 global`) and `data/processed/catalog_registered_labels.parquet`.
Startup takes a few seconds. Settings are under `mapview:` in `config/default.yaml`.

## What it shows

- **Matched** (yellow): a catalog crater and a CNN detection paired by the same
  whole-planet Hungarian match as `craters.compare_global`; the catalog circle is
  solid, the CNN's dashed. **Catalog only** (blue) and **CNN only** (orange) are the
  unmatched ones. Both sets are D ≥ 5 km, so the totals reproduce
  `outputs/global/` (20,391 catalog, 38,462 CNN).
- Circles are drawn as ellipses widened east-west by 1/cos(latitude), so they
  sit true to size on the equirectangular map. Craters under 1.5 px radius on
  screen are counted but not drawn.
- **CNN score cutoff**: raising it hides weaker detections, and the catalog
  partners of hidden detections become catalog only. The match is not re-run
  at each cutoff, so this is a close approximation. This is the way to compare a
  stricter CNN with the catalog's "clearly defined rim" criterion: as the cutoff
  rises, CNN-only craters (mostly degraded small ones) drop out much faster than
  matched ones.
- **Review verdicts** from `adjudication/verdicts.csv` (T1–T3 only): ✓ real,
  ✗ not real, ? unsure. Catalog verdicts attach by crater id, detection verdicts
  to the nearest global detection within 0.5 km.
- Outlines of T1–T3 and V1, the ±60° training limit and the areas left out of
  training (dotted).
- Counts per category are for crater centres in the current view.
- Click a circle for its position, diameter, score and verdict (the smallest
  circle containing the click wins).

## How it works

`src/craters/mapview.py` is a standard-library HTTP server like the review app.
`/tile/{z}/{x}/{y}.png` reads a window of the COG with rasterio, and GDAL picks
the overview level. The map uses Leaflet's simple CRS, with the planet
(0–360° E) as 2 × 1 tiles at zoom 0. Tile edges fall on multiples of 180/2^z
degrees, so no tile straddles the raster's 180° edge, and x wraps so panning
across 0°/360° is seamless. `tile_window` holds that math and is tested in
`tests/test_mapview.py`. Native resolution is zoom ≈ 7.5. Tiles are served
natively up to zoom 8 and upsampled up to zoom 10. The contrast stretch comes
from the coarsest overview, so it is the same at every zoom.

The page (`src/craters/map_page/index.html`) loads the crater sets as one JSON
document and draws them on canvas tiles, redrawn when a filter changes.
