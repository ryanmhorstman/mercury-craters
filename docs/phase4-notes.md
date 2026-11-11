# Phase 4 — Adjudication

```
conda run -n mercury-craters python -m craters.phase4 build     # review queue from data/detections/*.parquet
conda run -n mercury-craters python -m craters.phase4 review    # review app: http://localhost:8765
conda run -n mercury-craters python -m craters.phase4 report    # original vs adjusted metrics, any time
```

## What gets reviewed

Built from the Hough and CNN results on T1–T3 (craters ≥ 5 km):

| | 5–10 | 10–20 | 20–50 | 50+ | Total |
|---|---:|---:|---:|---:|---:|
| Catalog craters at least one detector missed | 891 | 410 | 149 | 35 | 1,485 |
| Unmatched detections (merged across detectors) | 1,022 | 81 | 110 | 25 | 1,238 |
| **To review** | 1,913 | 491 | 259 | 60 | **2,723** |

Another 73 detections were auto-marked as **duplicates**: a second circle on a crater that's
already catalogued, left over by one-to-one matching. They count as false positives and are
never shown, because "is there a crater here?" would wrongly be answered yes.

## How the review works

- **Blind and random.** Every item is shown the same way: a basemap chip centered on a dashed circle.
  The app doesn't say whether the circle came from your catalog or a detector, and items come in a
  fixed random order (seed 42). Press **I** after a verdict to see its source.
- **The question:** "Is the dashed circle a real impact crater of about that size?" Size within ~25%,
  center within half a diameter, which is the same tolerance as the matching rule. Half a diameter
  is the circle's radius, so the visual test is: **is the real crater's center inside the dashed
  circle?** An offset circle on the right-sized crater counts as real if so, and not real if the
  crater's center falls outside it. Right on the line: unsure.
- **Keys:**
  - **1** real, **2** not a crater, **3** unsure;
  - **←/→** move, **U** jump to the next unreviewed item;
  - **Z** wide view (3× the field), **O** hide the circle.
- **Saving:** every verdict is appended to `adjudication/verdicts.csv` the moment you make it. That
  file is tracked in git, since it's hand-made data. Changing your mind appends a new row; the latest
  wins. You can stop and resume any time.
- **Chips** render on demand from the basemap (about 0.3 s each, cached in `data/interim/chips/`).
  The next three are prefetched.

## How verdicts become numbers

Each item gets p(real): 1 or 0 when reviewed. Otherwise (unreviewed, or "unsure"), it gets the
real-rate of the human-reviewed items of the same kind and diameter bin. Because the order is
random, **any number of reviews is a fair sample.** The report always says how many of the 2,723
are reviewed, and the numbers become exact once all are.

For each detector:
- TP′ = original TP + Σ p over unmatched detections it proposed
- FP′ = Σ (1 − p) over unmatched detections it proposed
- FN′ = Σ p over catalog craters it missed + Σ p over new craters only *other* detectors found

A test checks that a fully reviewed queue gives exactly the same numbers as re-scoring each
detector against the corrected catalog.

**Catalog summary** ("how much did past-me miss?"):
- *new real craters*: Σ p over unmatched detections;
- *catalog entries not real*: Σ (1 − p) over missed catalog craters;
- *completeness*: real catalog craters / (real catalog craters + new real craters).

Completeness is an upper bound: craters that neither you nor any detector found can't be counted.

## Registration correction (found during review, 2026-10-31)

```
conda run -n mercury-craters python -m craters.phase4 register   # writes data/processed/catalog_registered.parquet
```

**What the first 308 verdicts revealed:** 65 of 92 reviewed 5–10 km catalog craters were marked
"not real". Investigation showed that's mostly a coordinate problem, not a catalog problem:
- **Both detectors place matched craters south of the catalog position in every region.**
  Typically 1–2 km in T1 and T3, and 2–7 km in T2.
- **The two detectors agree on north-south within ~0.5 km in nearly every 10° cell, but differ
  by a steady ~2 km east-west.** That east-west gap is a lighting bias: Hough always sits further
  west. So the north-south offset belongs to the catalog, and east-west can't be pinned down from
  the detectors.
- **About half of the "not real" small catalog craters had a same-sized detected crater ~0.5
  diameters away, ~2 km south.** That's just past the matching rule and the review rule.
- **Likely cause:** the 2015 JMARS MDIS mosaic and the USGS basemap use different control networks.
  The shift varies smoothly from place to place, as you'd expect from differently registered
  source images.

**The correction** (`src/craters/registration.py`, settings under `registration:` in the config):
- **North-south only, and local.** Each catalog crater moves by the median north offset of
  matched craters ≥ 10 km (both detectors pooled) within 5°, with at least 15 pairs. Craters
  without enough pairs nearby are left alone.
- **Two passes,** because the first pass can't see the largest shifts, which push some craters past
  the matching limit. The median remaining offset went from −1.39 km to −0.16 km.
- **Shifts applied** (median / largest): T1 −1.3 / −3.8 km, T2 −2.9 / −10.0 km, T3 −1.2 / −3.5 km.
  Map: `outputs/phase4/registration_shift_map.png`.
- **Originals are kept** in `lat_catalog` / `lon_catalog`. The thesis workbook and `catalog.parquet`
  are untouched. Scoring uses the registered copy when `evaluation.use_registered_catalog` is true.
- **Circularity is limited:** the shift field comes from craters ≥ 10 km, where a few km is well
  inside the matching tolerance anyway. It's a smooth field, not a per-crater snap to detections.

**Effect on Phase 3 scores (T1–T3):**
- CNN F1 0.557 → 0.586. At 5–10 km, recall rose from 0.377 to 0.420 and precision from 0.412 to 0.459.
- Hough F1 0.370 → 0.414.
- Bins at 20 km and up are unchanged.
- The earlier numbers are kept in `outputs/phase2/metrics_original_positions.csv`.

**Effect on the review:**
- **Verdicts are now position-aware:** a verdict counts only if its circle hasn't moved more
  than 0.25 km or changed size by more than 5%.
- **146 still apply:** 137 on detection circles, plus 9 on catalog circles that barely moved.
- **140 catalog circles moved** and are back in the queue. Their old rows stay in the file but are ignored.
- **The shift is confirmed independently:** 7 catalog craters you'd marked "not real" now match a
  detection, and 11 detections you'd marked "real new craters" now match catalog craters.
- **Chips are cached by position,** so a moved circle always gets a freshly centered image.

**Not corrected:**
- The east-west registration (unknown to about ±1–2 km).
- The original CNN's training labels. A retrain on registered labels (`cnn_reg`) is in
  phase3-notes.md §3.3b: higher precision, lower 5–10 km recall.

## CNN twin fix (2026-11-03)

The CNN models drew many craters twice: a second, weaker circle ~0.7–0.8 D west of the first
(phase3-notes.md §3.3c). After the fix, `phase4 build` gives **3,552 items to review instead of
4,599**. The verdicts file is untouched; verdicts on circles that no longer exist simply no longer
apply.
- **346 of the 3,552 items are reviewed**, down from 462 of 4,599.
- **135 reviewed detection items disappeared.** 126 of them had a kept same-size circle within 1 D
  (median 0.72 D), so they were twins. Their verdicts: **93 real**, 28 not real, 5 unsure.
  - Only 8 pairs had both members reviewed: 4 both "real", 2 one of each, 2 with no "real".
  - So direct double-counting was small. The bigger effect was on the extrapolation: reviewed
    twins were "real" 75% of the time, against 57% for other 5–10 km detections, and that raised
    the rate applied to every unreviewed 5–10 km detection.
  - **Why so many ghosts were "real" is open.** A circle 0.7–0.8 D off a crater has the crater's
    centre outside it, which the review rule calls "not real". The likely reading is that a
    neighbouring crater, or the dashed circle overlapping most of the crater, was accepted. The
    list is in `data/detections_pre_twinfix/reviewed_items_removed_by_twin_fix.csv` for spot
    checks.
- **Reviewed 5–10 km `cnn_reg_f2` detections judged real: 65% → 56%** (68% → 58% leaving out
  "unsure"). The "73%" in older notes came from an earlier verdict count. Adjusted 5–10 km
  precision, which counts matched detections too, goes from 0.72 to **0.75**: the twins had also
  been unmatched detections, and most of the remaining ones are matched.
- **Estimated new real craters in T1–T3: 2,046 → 1,198. Catalog completeness: 0.44 → 0.57**
  (5–10 km: 0.31 → 0.44).

| `cnn_reg_f2`, T1–T3 | Raw P / R / F1 | Verdict-adjusted P / R / F1 |
|---|---|---|
| all, before | 0.354 / 0.819 / 0.494 | 0.754 / 0.875 / 0.810 |
| all, after | 0.508 / 0.789 / **0.618** | 0.800 / 0.802 / **0.801** |
| 5–10 km, before | 0.242 / 0.789 / 0.370 | 0.723 / 0.886 / 0.797 |
| 5–10 km, after | 0.381 / 0.749 / 0.505 | 0.751 / 0.786 / 0.768 |
| 10–20 km, before | 0.750 / 0.846 / 0.795 | 0.851 / 0.874 / 0.863 |
| 10–20 km, after | 0.828 / 0.830 / 0.829 | 0.923 / 0.858 / 0.889 |

Also fixed: `phase2` and `phase4` used to read every file in `data/detections/`, including
`global_cnn_reg_f2.parquet` once the global run existed. That file is now skipped there and
compared by `compare_global` instead.

The north-south registration wasn't rerun. It uses detections ≥ 10 km, where twins are rare and
fall outside the 0.5·D matching anyway.

## Assumptions to keep in mind

- Catalog craters that *every* detector matched aren't reviewed and are assumed real.
- Your verdicts set the reference, so the adjusted numbers measure agreement with present-you
  looking at a 166 m mosaic. Past-you used a ~250 m mosaic in JMARS.
- At 5–10 km a chip shows the crater about 60 px across (2× zoom of the 166 m data).
  Very degraded small craters will be hard calls; "unsure" is fine.

## How much review is enough?

Roughly 200–300 verdicts give per-bin rates to within about ±5–10 percentage points for the large
bins. 5–10 km has the most items and matters most, so the random order naturally spends most
of your time there. About 2–3 seconds per item means ~10–15 minutes per 300.

## False positives of the published run (`--calibrated`)

```
conda run -n mercury-craters python -m craters.phase4 build --calibrated    # queue
conda run -n mercury-craters python -m craters.phase4 review --calibrated   # review app: http://localhost:8765
conda run -n mercury-craters python -m craters.phase4 report --calibrated   # estimate, any time
```

The verdicts above were given on the raw runs, before calibration. To estimate how
many detections of the published run (`global_cnn_reg_f2_calibrated`) are false, its
unmatched detections in T1–T3 get a queue of their own,
`data/processed/review_items_calibrated.csv`:

| | 5–10 | 10–20 | 20–50 | 50+ | Total |
|---|---:|---:|---:|---:|---:|
| Detections | 1,008 | 598 | 312 | 62 | 1,980 |
| Matched to the thesis | 595 | 462 | 234 | 45 | 1,336 |
| **To review** | 413 | 136 | 72 | 17 | **638** |

4 more are auto-marked duplicates. The queue is in random order (seed 42), so any first
N items are a random sample: about 300 give the overall rate to within about ±6 points.
Verdicts go to the same `adjudication/verdicts.csv`. A verdict carries over only to the
identical circle, and calibration resized every detection, so this review starts from zero.

The report counts matched detections as real, reviewed ones as judged, and the rest at
their size bin's reviewed rate, with a 95% interval
(`outputs/phase4/calibrated_false_positives.csv`). A bin with no verdicts yet has no estimate.

Unlike the Phase 4 queue, this one holds no catalog craters, so it isn't blind: every circle
is a detection the thesis doesn't list.

### Result (2026-11-11): about 1 in 5 detections is false, mostly at 5–10 km

The first 304 items of the queue are reviewed, a random sample of 302 of the 638 (8 unsure).
Matched detections count as real; the 95% intervals come from the review sample alone.

| D (km) | detections | in the thesis | unmatched judged real | est. false | precision (95%) |
|---|---:|---:|---:|---:|---|
| all | 1,980 | 1,336 | 124 of 294 (42%) | 377 (19%) | **0.81** (0.80–0.82) |
| 5–10 | 1,008 | 595 | 57 of 193 (30%) | 291 (29%) | 0.71 (0.70–0.73) |
| 10–20 | 598 | 462 | 28 of 56 (50%) | 68 (11%) | 0.89 (0.87–0.90) |
| 20–50 | 312 | 234 | 32 of 37 (86%) | 16 (5%) | 0.95 (0.93–0.96) |
| 50+ | 62 | 45 | 7 of 8 | 2 (3%) | 0.97 (0.92–0.98) |

- Against the thesis alone, the run's precision is 0.675. Counting the real craters the thesis
  doesn't list, it is about 0.81.
- Most false detections are 5–10 km: 291 of about 377. From 20 km up, about 95% are real,
  in line with the DEM check (87–90% backed by the thesis or the DEM, a strict lower bound).
- About 267 of the 644 unmatched detections are real craters missing from the thesis.
- The queue wasn't blind (every circle was an unmatched detection), and the verdicts measure
  agreement with the author looking at the 166 m mosaic now.
