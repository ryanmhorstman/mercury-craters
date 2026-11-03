# Phase 1 — Ground truth and tiling

Status: awaiting review.

Run everything with:

```
conda run -n mercury-craters python -m craters.phase1     # ~15 s (plus ~3 min the first time, for the basemap copy)
conda run -n mercury-craters python -m pytest              # 16 tests
```

## What it produces

| Output | What |
|---|---|
| `data/processed/catalog.parquet` | 20,662 unique craters: `id, lat, lon, diameter_km, source`, plus `sheet_row, also_in, morph_type, name, split` |
| `data/processed/catalog_duplicates.csv` | the 437 merged entries, each with the id it was merged into |
| `data/interim/basemap_166m_cog.tif` | tiled + compressed copy of the basemap (2.6 GB, with overviews). The raw file is untouched. |
| `data/processed/tiles.parquet` | 4,722 tile definitions (see below) |
| `outputs/phase1/*.png` | overlay previews: basemap and DEM hillshade side by side |

## Catalog

- Read from the workbook in read-only mode. Longitude is east 0–360, Simple Craters converted m → km.
- **Duplicates:** one "same crater" rule for everything (center ≤ 0.5 D, diameter ratio 0.75–1.33,
  in `config/default.yaml`). Sheets are visited in priority order (Peak Ring → ProtoBasins →
  Central Pits → Central Peaks → Simple), so the more specific record wins and the other sheet goes
  in `also_in`. 437 merged, of which 358 are Central Peaks also listed in Simple Craters and 38 are
  within Simple Craters. The 4 Central Pit ↔ Central Peak merges are expected: every pit in the
  thesis is a summit pit, i.e. a pit *on* a peak.
- **ProtoBasins are included**, as agreed.

| Craters ≥ 5 km | 5–10 | 10–20 | 20–50 | 50+ | All |
|---|---:|---:|---:|---:|---:|
| T1 | 436 | 218 | 106 | 20 | 780 |
| T2 | 330 | 159 | 74 | 18 | 581 |
| T3 | 324 | 185 | 92 | 17 | 618 |
| V1 (validation) | 308 | 116 | 56 | 8 | 488 |
| train | 5,824 | 3,828 | 1,830 | 307 | 11,789 |
| excluded | 2,082 | 2,204 | 1,527 | 322 | 6,135 |

## Tiling

- **Each tile is resampled onto its own stereographic projection, centered on the tile.** Stereographic
  preserves shape, so craters are round at any latitude. The tests check a synthetic 200 km crater
  at 60°N: it comes out round, where the raw raster is 2:1.
- The resampling inverts each tile pixel to lat/lon and samples the source raster with
  wrapped longitude, so tiles straddling either raster's seam work (basemap seam at 180°E, DEM
  seam at 0°). Tested both ways.
- **Two levels**, both 1024 px:

| Level | Pixel | Tile | Overlap | For craters | Tiles (test / val / train) |
|---|---|---|---|---|---|
| fine | 166 m | 170 km | 42.6 km | 5–40 km | 266 / 94 / 2,271 |
| coarse | 665 m (block-averaged) | 681 km | 341 km | 30–400 km | 38 / 13 / 328 |

- **The overlap rule:** tile centers sit on a grid of step = size − overlap, and every tile
  extends overlap/2 past its cell. So any crater no wider than the overlap lies entirely inside
  at least one tile. A randomized test on 300 craters checks this.
- **Merging repeat detections across overlaps:** `merge_overlap_detections` applies the same
  "same crater" rule, keeping the highest-scoring copy and recording how many tiles saw it.
- A tile's split is set by its center. Training tiles stay at least 5° from every test and
  validation box.

## What the previews show (please look: `outputs/phase1/`)

1. **Alignment is right.** Rachmaninoff's catalog circle sits on its rim in both the basemap and the
   DEM, and so do Praxiteles and the sample tiles. The longitude handling is correct for both rasters.
2. **There are obvious uncataloged craters, even in the test regions.** In the T2 sample (52.5°N)
   a ~40 km crater and a pair of ~15 km craters have no catalog circle. In T1 a degraded ~40 km
   crater beside a large central-peak crater is missing. The thesis survey seems to favor fresh,
   well-defined craters. That means:
   - precision will look low against the original catalog;
   - **Phase 4 adjudication will matter a lot**, and "how much did past-me miss" will have a real answer;
   - for the deep detector's training labels, uncataloged craters act as negatives. That's a known
     source of under-detection, and something to mitigate in Phase 3 (e.g. ignore unlabeled
     detections during training rather than penalizing them, or train on a cleaned subset).
3. **The mosaic has visible seams** where source images meet (brightness steps, straight edges).
   The classical Hough baseline will probably fire on these.
4. **At the fine level the DEM is very soft.** It's 665 m data shown at 166 m pixels, as expected:
   fine for DeepMoon on large craters, useless at 5 km.

## Known simplifications

- Stereographic scale varies < 0.1% across a fine tile, so diameters use the nominal pixel size.
- Fine tiles at 60° latitude sample the equirectangular basemap at about 2× its east-west
  pixel density, with bilinear interpolation and no pre-filtering. The slight east-west
  softening is negligible at 5 km (30 px).

## Viewer

```
conda run -n mercury-craters python -m craters.viewer --lat 27.77 --lon 57.08 --level coarse
conda run -n mercury-craters python -m craters.viewer --tile-id fine_r020_c0099 --source basemap
```
