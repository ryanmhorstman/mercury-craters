# Project plan (as written by Ryan, 2026-10-30)

I'm building a crater-detection pipeline for Mercury and comparing it against
my master's thesis catalog. For my thesis I hand-cataloged every crater
≥ 5 km in my study region using JMARS. My automation attempt back then never
worked. I want to see how modern methods compare to my manual counts.

All the data is in this repo/folder. Work in phases and stop for my review
after each one.

## Phase 0 — Inventory (no code yet)
- Find and describe every data file: format, CRS/projection, resolution,
  coverage, and columns. Expected: my thesis catalog (probably CSV/shapefile
  with lat, lon, diameter) and MESSENGER imagery and/or a DEM.
- Report:
  - my study region's bounding box
  - how many catalog craters there are and the diameter distribution
  - whether the imagery fully covers the region
  - the pixel scale, and how many pixels across a 5 km crater spans
- Flag anything missing or inconsistent: longitude conventions (0–360 vs
  ±180, east vs west positive), units (km vs m, radius vs diameter),
  duplicates.
- Propose a repo structure and environment. Ask me about compute (local GPU
  or not) before choosing models.

## Phase 1 — Ground truth + tiling
- Load my catalog into one clean table: id, lat, lon, diameter_km, source.
  Save it as Parquet. Data files are gitignored; record paths in a config.
- Tile the region for detection. IMPORTANT: don't detect on raw
  equirectangular tiles at higher latitudes — craters stretch into ellipses.
  Reproject each tile to a local orthographic or stereographic projection
  centered on that tile, then convert detections back to lat/lon.
- Use overlapping tiles. De-duplicate detections across overlaps.
- Write a viewer that overlays catalog craters as circles on a tile, so I
  can sanity-check alignment before any detection runs.

## Phase 2 — Matching + metrics
Build the evaluation before building any detector:
- A detection matches a catalog crater when:
  - the center distance is ≤ 0.5 × the catalog diameter, and
  - the diameter ratio is within 0.75–1.33
  Use one-to-one Hungarian matching. Put the thresholds in config.
- Metrics: precision, recall, F1, and diameter error. Report them overall
  AND by diameter bin (5–10, 10–20, 20–50, 50+ km).
- Size-frequency distribution: plot cumulative SFD on log–log axes (my
  catalog vs. each detector), with Poisson error bars.
- Add a test with a synthetic catalog against itself → perfect scores.

## Phase 3 — Detectors (simplest first)
1. Classical baseline: edge detection + Hough circles, or template matching
   on the imagery. This is roughly what I tried in grad school, so it's the
   "past me" benchmark.
2. Pretrained deep model: DeepMoon (Silburt et al. 2019) was trained on
   lunar DEMs and transferred to Mercury. Use it if the weights are
   available and we have a DEM. Note the DEM's resolution limits: if a 5 km
   crater is only ~7 pixels across in the DEM, expect poor recall below
   ~10–15 km, and say so in the results.
3. Image-based detector: fine-tune a modern object detector (e.g. YOLO-style
   or a U-Net rim segmenter) on the imagery, where 5 km craters have enough
   pixels.
   - Split train/test BY REGION (contiguous blocks), never randomly by
     crater, or the scores will be inflated.
   - Train only on my catalog's training region.
4. Optional, last: a Claude vision experiment on a small sample of tiles.
   Just for comparison, clearly labeled as exploratory.

## Phase 4 — Adjudication
My catalog isn't perfect either. Some "false positives" are real craters I
missed.
- Build a small review tool: show each unmatched detection and each missed
  catalog crater, and let me mark it real / not real / unsure.
- Save the verdicts. Recompute metrics against the adjudicated catalog, and
  report both the original and adjusted numbers. The question "how much did
  past-me miss?" is half the fun.

## Phase 5 — Write-up
- One notebook or report:
  - the region map
  - a metrics table per detector and diameter bin
  - SFD comparison plots
  - example tiles showing hits, misses and false positives
  - a short honest discussion of limitations
- If published Mercury catalogs are available (e.g. Fassett et al. 2011 for
  ≥ 20 km), compare against them in the overlapping range as an outside
  check.
- Optional later: a lightweight web viewer to browse tiles with toggleable
  layers (catalog, each detector, adjudicated). Build it last.

## Goals (added 2026-10-30)
1. **First:** reproduce in days what took a year of manual JMARS counting, and see how it compares.
2. **Later:** roll this out as a general-purpose crater counting tool (other bodies, other mosaics,
   other catalogs). Known groundwork when we get there: pass the body radius from config through
   everything (it's currently fixed to Mercury in code, and `body.radius_m` in the config isn't
   used yet), plus a generic catalog reader and a standalone `detect` command.

## General
- Reproducible: fixed seeds, all paths and thresholds in config, one
  command per phase.
- Keep the code readable over clever — this may end up as a portfolio piece.
- Never modify my original thesis data files.
