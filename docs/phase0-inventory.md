# Phase 0 — Data inventory

Status: awaiting review. No code yet, and the original files have not been modified.

## 1. What was found

The only data on disk is the thesis catalog workbook. There is no MESSENGER imagery,
no DEM, no shapefile and no CSV.

| File | Format | Notes |
|---|---|---|
| `Mercury Crater Catalogue 1.0.xlsx` (copied to `data/raw/`, not public) | Excel, 5 sheets, 860 KB | The catalog. Read-only for this project. |
| `Thesis\appendixA–C.tex` | LaTeX tables | Rounded copies of the Central Peaks, Peak Ring and Central Pit sheets. Not needed as a data source. |
| `…\Thesis\*.tif / *.png` | Figures | Thesis plots and maps, not georeferenced imagery. |

CRS: none is stored. The values are planetocentric-looking latitude and **east-positive
longitude, 0–360°** (min 0.027, max 360.0, none negative). JMARS uses ocentric east 0–360
by default, so that's the working assumption. **Please confirm.**

## 2. The catalog

| Sheet | Rows | Columns | Diameter range |
|---|---:|---|---|
| Central Pits | 32 | lon E, lat N, Diameter (km), Dp, Dp/Dc, Dpk, Dpk/Dc, Type (all `SP`), Preservational State (all 1), Geologic Unit (all `N/A`) | 13.6–47.4 km |
| Peak Ring Basins | 86 | lon E, lat N, diameter (no unit, km by magnitude), Dr, Dr/Dc, Features, Name | 85–323 km |
| ProtoBasins | 45 | lon E, lat N, diameter (km), Dr, Dr/Dc, Type, Features, Name | 71–192 km |
| Central Peaks | 1,764 | lon E, lat N, diameter (km), Dpk, Type (`C` 850 / `S` 913 / blank 1), Dpk/Dc | 8.2–251 km |
| Simple Craters | 19,172 | lon E, lat N, **Diameter (meters)** | 2.79–40.0 km |
| **Total** | **21,099** | | |

Diameter distribution (all sheets, before de-duplication):

| Bin | <5 km | 5–10 | 10–20 | 20–50 | 50+ |
|---|---:|---:|---:|---:|---:|
| Count | 271 | 9,313 | 6,823 | 3,976 | 716 |

### Study region bounding box

There isn't one. The catalog is **global**: longitude 0–360°E, latitude 88.4°S–88.8°N.
By 30° latitude band, south to north: 689 / 3,375 / 6,030 / 6,287 / 3,772 / 946.
This contradicts "every crater ≥ 5 km in my study region", and it changes Phase 2–3 (see §5).

## 3. Issues to flag

1. **Units differ by sheet.** Simple Craters is in meters; everything else is km.
   Peak Ring Basins' `diameter` header has no unit (km by magnitude).
2. **Radius vs diameter.** Every column is labeled diameter, and the values agree with
   named craters (Rachmaninoff ≈ 293 km here, ~290 km in the literature). Fine.
3. **Exact duplicates (3):**
   - Peak Ring Basins row 32 = ProtoBasins row 7 (82.82°E, 52.52°N, 192.3 km)
   - Peak Ring Basins rows 35 and 37 are identical (296.36°E, 14.90°S, 205.6 km)
   - ProtoBasins row 33 = Central Peaks row 2 (281.05°E, 52.31°N, 129.3 km)
4. **Near-duplicates across sheets.** Under the Phase 2 match rule (center ≤ 0.5 D,
   D ratio 0.75–1.33) there are 365 Central Peaks ↔ Simple Craters pairs,
   13 Central Pits ↔ Simple, 21 Central Peaks ↔ ProtoBasins and 64 within Simple.
   Combined with Simple craters up to 40 km (Mercury's simple-to-complex transition
   is ~10–12 km), the "Simple Craters" sheet looks like a **general crater list**
   rather than morphologically simple craters only. It isn't in any thesis appendix.
   **Where did it come from, and what was its completeness cutoff?**
5. **The 5 km threshold.** 271 Simple Craters are < 5 km (min 2.79 km). The ≥ 5 km
   cut will drop them.
6. **Central Peaks `Type` S vs C:** meaning unknown. My guess is single vs cluster peak.
   One row (1765) has no type.
7. **Text glitches (harmless):** a mangled name `A?vaghosa` (Aśvaghoṣa) in ProtoBasins.
   Appendix C's LaTeX header swaps "longitude N / latitude E", but the workbook is correct.

## 4. Imagery and DEM

### Downloaded 2026-10-30 into `data/raw/` (gitignored)

Source: USGS Astrogeology, `https://planetarymaps.usgs.gov/mosaic/<file>`, which redirects to
USGS's `asc-pds-services` S3 bucket. Both files were last modified on the server 2022-11-10,
and downloaded sizes match the server's Content-Length.

| | Basemap | DEM |
|---|---|---|
| File | `Mercury_MESSENGER_MDIS_Basemap_BDR_Mosaic_Global_166m.tif` | `Mercury_Messenger_USGS_DEM_Global_665m_v2.tif` |
| Size | 4.25 GB (BigTIFF, uncompressed, striped) | 0.53 GB (uncompressed, striped) |
| Raster | 92,160 × 46,080, 1 band, uint8 | 23,040 × 11,520, 1 band, int16 |
| Pixel scale | 166.31 m/px (256 px/°) | 665.24 m/px (64 px/°) |
| Projection | Equirectangular (Simple Cylindrical), sphere R = 2,439.4 km, std parallel 0° | same |
| **Central meridian** | **0° → longitudes −180 to 180** | **180° → longitudes 0 to 360** |
| Values | DN 1–254, nodata 0 | height = raw × 0.5 m relative to R = 2,439.4 km, nodata −32768 |
| 5 km crater | **30 px** | **7.5 px** |

Coverage (sampled 60×60 points per test region, plus every 2° of longitude in 10° latitude bands):
**0% nodata everywhere**, including the poles and T1–T3. DEM heights in the test regions run
from about −3.9 to +4.5 km.

Things Phase 1 must handle:
- **The two rasters use different central meridians.** The catalog is 0–360°E. Convert
  longitude per raster through one function (pyproj), never by hand, and add a test that a
  known crater (e.g. Rachmaninoff) falls on the same feature in both.
- **The basemap is uncompressed and striped**, so reading windows works but is slow-ish. Build a
  tiled, compressed copy (or a VRT) in `data/interim/` once, and never modify the raw file.
- "Global, no nodata" doesn't mean uniform quality. Lighting and resolution in the source images
  vary, and the polar regions are mostly near-terminator imaging. The viewer should make that visible.

### Other sources still to fetch later

Not downloaded yet (Phases 3 and 5).

| Product | Pixel scale | 5 km crater | Approx. size |
|---|---|---|---|
| Fassett et al. 2011 crater catalog (≥ 20 km) | n/a | n/a | small, for the outside check |
| DeepMoon weights (Silburt et al. 2019) | n/a | n/a | ~100 MB (availability to confirm) |


The DEM matches the warning in the plan: at ~7.5 px per 5 km crater, DeepMoon-on-DEM
recall should collapse below ~10–15 km. The 166 m/px basemap is the right input for 5–10 km.

## 5. What a global catalog changes

- **Evaluation region.** The plan can't use "the study region". Instead, define a few
  evaluation regions (e.g. 2–4 blocks of ~30°×30° at different latitudes), and split
  train/test by region across the globe.
- **Recall and precision both need a completeness assumption.** If the catalog only lists
  craters with peaks, pits or rings plus the general list, then degraded or flat-floored
  complex craters may be missing, and good detections will count as false positives.
  Phase 4 adjudication covers this, but we should pick evaluation regions where you believe
  your catalog was complete.

### Evaluation regions (chosen 2026-10-30; Ryan asked me to pick)

How I chose them: I binned the 5–50 km craters into 30°×30° cells and checked two things
per cell: area density, and the share of 5–10 km craters. On a fully counted surface,
5–10 km craters should be roughly half or more of the 5–50 km count, so a low share
means small craters were probably under-counted.

What the grid shows:
- **Poleward of 60°S, small craters are almost absent** (5–10 km share 0.00–0.12 in most
  cells). The catalog isn't complete at 5 km there.
- **There's a low-density stripe at 180–240°E in every latitude band** (e.g. 114 vs ~300–500
  per 10⁶ km² nearby). A north–south stripe isn't a geologic pattern, so it's most likely
  a cataloging or imaging gap. Smooth plains could explain part of it. Worth a look in the
  Phase 1 viewer.
- Several 60°S–30°S cells also have a low small-crater share (0.13–0.32).

Held-out test regions (never used for training or tuning):

| Region | Box | 5–10 | 10–20 | 20–50 | 50+ | Total | Why |
|---|---|---:|---:|---:|---:|---:|---|
| T1 equatorial | 270–300°E, 0–30°N | 437 | 223 | 118 | 23 | 801 | dense, small-crater share 0.56; least projection distortion |
| T2 mid-north | 300–330°E, 30–60°N | 330 | 160 | 77 | 19 | 586 | share 0.58; tests the reprojection at higher latitude |
| T3 southern | 60–90°E, 30°S–0 | 324 | 187 | 95 | 19 | 625 | share 0.53; a different hemisphere and lighting |

Training pool: everything within ±60° latitude except the test boxes plus a 5° buffer
around each (so overlapping tiles can't leak), the 180–240°E stripe and the low-share
cells above. That leaves **12,671 craters ≥ 5 km** (6,244 / 4,057 / 2,038 / 332 by bin).
Validation for tuning comes from a contiguous block inside the training pool, never from T1–T3.

Caveats:
- I chose these regions using the catalog itself, so they favor cells where your counting
  looks complete. Recall there is "recall where past-me was thorough". Phase 4 adjudication
  remains the real check.
- There are only ~60 craters in the 50+ km bin across all three test regions, so those
  metrics will have wide error bars.
- Excluded areas still get detections for the SFD and the write-up. They're just not scored
  or trained on.

## 6. Proposed repo structure and environment

```
mercury-craters/
  config/default.yaml        # data paths, thresholds, bins, seeds, regions
  data/                      # gitignored: raw/ (downloads), interim/, processed/
  src/craters/
    catalog.py               # load workbook → clean Parquet (Phase 1)
    tiling.py                # overlapping tiles, local stereographic reprojection
    match.py, metrics.py     # Hungarian matching, P/R/F1, SFD (Phase 2)
    detectors/               # hough.py, deepmoon.py, yolo.py, claude_vision.py
    review/                  # adjudication tool (Phase 4)
  scripts/                   # one entry point per phase: python -m craters.phase1 …
  tests/                     # synthetic catalog → perfect scores, etc.
  notebooks/report.ipynb     # Phase 5
  docs/PLAN.md, docs/phase0-inventory.md
```

Environment: Python 3.11 with conda-forge (GDAL, rasterio and pyproj install much more
reliably there on Windows): rasterio, pyproj, numpy, pandas, pyarrow, scipy (Hungarian),
scikit-image / OpenCV (Hough), matplotlib, pytest. Deep-learning packages
(PyTorch / ultralytics, TensorFlow for DeepMoon) wait until compute is decided.

### Compute (answered 2026-10-30: local GPU)

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 Ti, 8 GB VRAM, driver 591.86 (CUDA 13.1) |
| RAM | 16 GB |
| Python | only 3.15 from the Microsoft Store, no conda |

What this means:
- **Python env:** install Miniforge and create a Python 3.12 env. 3.15 is too new for
  reliable GDAL/rasterio and PyTorch wheels. Pin everything in `environment.yml`.
- **16 GB RAM:** never load the ~4 GB global mosaic whole. Read tiles with rasterio windows
  (or a VRT), which tiling needs anyway.
- **Image detector on 8 GB VRAM:** a small/medium YOLO (ultralytics) at 640 px tiles with
  batch ~8–16, or a U-Net rim segmenter at 256–512 px. Use mixed precision. Training
  should take hours, not days.
- **DeepMoon:** it's an old Keras/TensorFlow model, and TensorFlow dropped native Windows GPU
  support after 2.10. Run its inference on CPU in a separate small env (the model is small and
  this is inference only), or use WSL2 if that's too slow. Keep it out of the main env.
- **Claude vision experiment (optional):** API calls, no local GPU. It costs money, so we'll
  confirm the sample size and budget before running it.

## 7. Questions before Phase 1

1. ~~**Compute**~~ answered above.
2. ~~**Region**~~ answered: evaluation regions T1–T3 above.
3. ~~**Simple Craters sheet**~~ answered by the thesis (see below): it's the rest of Ryan's
   own global JMARS survey.
4. ~~**Conventions**~~ east-positive confirmed by named craters (see below).
5. ~~**Downloads**~~ done; see §4.
6. ~~**Central Peaks type**~~ answered: `S` = simple peak, `C` = complex peak.

### What the thesis says (Horstman 2016, NAU M.S.)

- **The survey is global and entirely Ryan's own:** "a total of 21,591 craters across the surface",
  of which 1,878 (8.70%) are complex: 1,760 central peaks (911 simple peak, 849 complex peak),
  86 peak-ring basins and 32 central pits. Those numbers match the workbook sheets. So
  **Simple Craters = every surveyed crater not classified as complex**, not strictly bowl-shaped
  ones. That explains entries up to 40 km and the overlaps with the complex sheets (likely
  re-entries of the same crater).
- **Count gap:** the workbook has 19,172 + 1,927 = 21,099 rows against 21,591 in the text
  (−492). ProtoBasins (45) isn't in the thesis at all and may have been added later.
  Not blocking, but worth knowing.
- **How it was measured:** JMARS crater-counting tool, three points on the rim → diameter, lat,
  lon; exported to CSV, then Excel. Interior features measured with the JMARS distance tool.
- **What imagery he used:** the MDIS global mosaic in JMARS at **~250 m/pixel**. Our basemap is
  166 m/px, so the detectors see sharper images than past-Ryan did. Note this in the write-up.
- **Measurement error:** about ±10% on diameter (e.g. 8.2 ± 0.8 km, 251.3 ± 25.1 km), from repeat
  measurements plus resolution. That sits well inside the 0.75–1.33 match ratio.
- **No minimum-diameter cutoff is stated.** The data drops off steeply below ~5 km (min 2.79 km),
  so 5 km as the evaluation floor is an inference, not a stated rule.
- **Simple-to-complex transition** is cited as ~10 km (Pike 1988), with the smallest peak crater
  in the survey at 8.2 km.

### Longitude convention check
Named craters land where east-positive longitude puts them: Rachmaninoff at 57.08°E, 27.77°N
(published ~57.6°E, 27.6°N) and Praxiteles at 299.77°E, 27.20°N (published ~59.8°W = 300.2°E).
**Planetocentric latitude, east-positive 0–360° longitude, confirmed.**
