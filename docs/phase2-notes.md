# Phase 2 — Matching and metrics

Status: awaiting review. No detectors yet. This phase builds and checks the scoring.

```
conda run -n mercury-craters python -m craters.phase2     # ~5 s
conda run -n mercury-craters python -m pytest              # 24 tests
```

## How a detector is scored

**Input:** `data/detections/<detector>.parquet` with `lat, lon, diameter_km, score`
(east longitude 0–360). Every file there is picked up automatically.

**Matching** (`src/craters/match.py`, thresholds in `config/default.yaml` → `match`):
- A detection *may* match a catalog crater when the center distance ≤ 0.5 × catalog diameter
  and the diameter ratio (detection / catalog) is in 0.75–1.33. Both bounds are inclusive.
- Among allowed pairs, **Hungarian one-to-one assignment** minimizes
  `distance / D_catalog + |ln(ratio)|`. It's solved per connected group of candidate pairs, so it
  scales to the whole catalog. A test shows a case where greedy matching would lose a crater
  and Hungarian doesn't.
- Distances are great-circle on the 2,439.4 km sphere.

**Counting** (`src/craters/metrics.py`). Matching runs on everything; counting is scoped to the
region boxes and to D ≥ 5 km (the first bin edge):

| | Counted when |
|---|---|
| TP | matched, and the catalog crater is in the region and ≥ 5 km |
| FN | catalog crater in the region, ≥ 5 km, unmatched |
| FP | detection in the region, ≥ 5 km, unmatched |
| ignored | detection matched to a catalog crater just outside the region or just under 5 km |

Without the "ignored" case, a correct detection of a 4.8 km crater measured as 5.2 km would
be scored as a false positive. Same for craters straddling a region edge.

**Metrics:** precision, recall, F1, signed median diameter error (+ means the detector says bigger),
median absolute diameter error, median center offset (as a fraction of D), and 95% Wilson
intervals on precision and recall. These matter for the 50+ km bin, which has only 55 craters
across T1–T3. Rows: overall, then 5–10, 10–20, 20–50, 50+ km. **By bin, TP and FN go by the
catalog diameter and FP by the detection diameter.**

**SFD:** cumulative N(≥D) per km² at standard √2 diameter steps (Crater Analysis Techniques
Working Group, 1979), 1σ Poisson error bars (√N / area), log–log. Catalog vs each detector,
for T1, T2, T3 and all three combined.

## Sanity runs (labeled `sanity:` in every output)

| Run | Expected | Got (T-all) |
|---|---|---|
| `sanity:catalog_self` (catalog vs itself) | 100% everywhere, 0 error | 1,979 / 1,979 TP, 0 FP, 0 FN, 0.0% error ✔ |
| `sanity:noisy` recall, 5–10 km | ~68–71% (set) | 69% ✔ |
| `sanity:noisy` recall, 50+ km | ~89% (set) | 89% ✔ |
| `sanity:noisy` false positives | 300 injected | 300 ✔ |
| `sanity:noisy` diameter bias | 0 | −0.0% ✔ |

`sanity:noisy` is a fake detector: it perturbs the catalog with a size-dependent miss rate,
center and size jitter, and power-law false alarms. It is not a crater detector. The test suite
also checks it against injected values.

## Outputs (`outputs/phase2/`)

- `metrics.csv`: every detector × region (T1, T2, T3, T-all) × bin, all columns
- `metrics.md`: the same as a readable table
- `sfd_T-all.png`, `sfd_T1.png`, `sfd_T2.png`, `sfd_T3.png`

## Things to keep in mind for Phase 3

- From the Phase 1 previews, the catalog is missing real craters, so **precision against the
  original catalog is a lower bound**. Phase 4 adjudication will show the adjusted numbers.
- Detector tuning (score thresholds, Hough parameters) uses validation region V1 only.
  T1–T3 are scored once per detector.
