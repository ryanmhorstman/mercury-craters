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
- The CNN's training labels, which carry the uncorrected shift. Retraining on registered
  labels is a likely improvement for 5–10 km.

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
