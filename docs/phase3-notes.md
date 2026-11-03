# Phase 3 — Detectors

```
conda run -n mercury-craters python -m craters.phase3 hough   # tune on V1, run on T1-T3 (~6 min, first run renders tiles)
conda run -n mercury-craters python -m craters.phase2         # score everything
```

Every detector is tuned on validation region V1 only and run once on T1–T3.
Shared plumbing is in `src/craters/detect.py`:
- it uses every tile *touching* a region, not just those centered in it;
- it caches rendered tiles in `data/interim/tiles/`;
- it keeps only circles that sit wholly inside a tile and within that level's size range;
- it converts them to lat/lon, then merges repeats across overlaps and between the two levels
  (fine: 5–40 km, coarse: 30–400 km).

## 3.1 Classical baseline ("past me")

`src/craters/detectors/hough.py`, in order:
1. local contrast normalization (CLAHE), to even out lighting and mosaic seams;
2. per radius octave, shrink the image so the band's smallest radius is ~10 px;
3. Gaussian blur, Canny edges, then OpenCV Hough circles.

**Score = rim support:** the fraction of the circle lying on edge pixels.

**Tuning (V1):** 13 parameter sets × 8 score cutoffs. Full table in
`outputs/phase3/hough_tuning_V1.csv`. Winner: `HOUGH_GRADIENT_ALT`, perfectness 0.85, Canny 150,
no score cutoff, V1 F1 0.359. It's inside the grid: 0.8 and 0.9 both score lower.

**Results, T1–T3 (1,979 catalog craters ≥ 5 km):**

| Diameter | Catalog | TP | FP | FN | Precision | Recall (95% CI) | F1 | Median diam. error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **all** | 1,979 | 593 | 630 | 1,386 | **48%** | **30%** (28–32%) | **37%** | −6.2% |
| 5–10 | 1,090 | 288 | 548 | 802 | 34% | 26% (24–29%) | 30% | −7.1% |
| 10–20 | 562 | 160 | 29 | 402 | 85% | 28% (25–32%) | 43% | −6.9% |
| 20–50 | 272 | 125 | 45 | 147 | 74% | 46% (40–52%) | 57% | −3.3% |
| 50+ | 55 | 20 | 8 | 35 | 71% | 36% (25–50%) | 48% | −11.4% |

What the numbers and example tiles (`outputs/phase2/examples/hough_T*.png`) show:
- **It finds fresh, sharp-rimmed craters and misses degraded ones.** Recall is about 30%, roughly where a
  2015-era automated attempt would plausibly land.
- **Most false positives are 5–10 km** (548 of 630). Some are real uncataloged craters, as in the T1
  example. Phase 4 will measure how many.
- **Diameters come out ~6% small.** The circle locks onto the sharp inner edge of the sunlit rim
  rather than the rim crest that JMARS three-point picks target.
- **Recall at 10–20 km is no better than at 5–10 km,** even though the larger craters span more pixels.
  Partly that's because many 10–20 km craters are degraded. Partly it's because a single global
  setting has to work across all radius bands.

## 3.3 Trained image detector (CNN)

```
conda run -n mercury-craters python -m craters.phase3 cache-tiles   # once, ~4 min, 3.2 GB
conda run -n mercury-craters python -m craters.phase3 cnn           # train ~40 min on an RTX 3060 Ti, then run on T1-T3
```

**Model** (`src/craters/detectors/cnn.py`): CenterNet-style. A ResNet-34 backbone (ImageNet weights,
torchvision, BSD) with a feature-pyramid decoder to 1/4 resolution, predicting:
- a crater-center heatmap,
- a sub-pixel center offset,
- log radius.

Local maxima of the heatmap are craters, so the output is circles, not boxes. I chose this over
YOLO/ultralytics because that package is AGPL, which matters for the later general-purpose tool.

**Training** (`src/craters/train_cnn.py`), training split only:
- **Views:** fine tiles give random 512 px crops at 166 m/px, labeled 5–40 km; coarse tiles are shrunk
  2× to 512 px (1,330 m/px) and labeled 30–400 km.
- **Ignore regions:** catalog craters outside a view's size range are ignored, so the model is never
  penalized for finding a 3 km crater the catalog has but we don't score.
- **Augmentation:** random zoom 0.75–1.33×, the 8 flips/rotations, brightness/contrast/gamma.
- **Schedule:** 40 epochs × 4,800 samples, batch 12, AdamW, one-cycle learning rate, mixed precision.
  Peak GPU memory 1.8 GB.
- **Selection:** every 4 epochs the model runs end-to-end on V1. The best V1 F1 picks the checkpoint
  and score cutoff: **epoch 16, score ≥ 0.15, V1 F1 0.600.** V1 F1 then drifted to ~0.50 while the
  training loss kept falling. Precision against the catalog dropped as recall rose, which looks like
  some combination of overfitting and the model learning craters the catalog doesn't have.
  V1 has only 488 craters, so epoch-to-epoch V1 F1 is noisy (±0.05).

**Results, T1–T3 (scored once):**

| Diameter | Catalog | CNN precision | CNN recall (95% CI) | CNN F1 | Hough F1 | CNN median diam. error |
|---|---:|---:|---:|---:|---:|---:|
| **all** | 1,979 | **57%** | **54%** (52–57%) | **56%** | 37% | −5.3% |
| 5–10 | 1,090 | 41% | 38% (35–41%) | 39% | 30% | −5.6% |
| 10–20 | 562 | 81% | 69% (65–73%) | 75% | 43% | −6.8% |
| 20–50 | 272 | 66% | 83% (78–87%) | 74% | 57% | −2.8% |
| 50+ | 55 | 72% | 91% (80–96%) | 81% | 48% | −2.8% |

By region (F1): T1 53%, T2 55%, T3 61%. Hough's were 40%, 39% and 32%.

What it shows:
- **Above 10 km the CNN largely reproduces the catalog:** 69% / 83% / 91% recall, with precision
  of 66–81% even against an incomplete catalog.
- **The cumulative SFD tracks the catalog within the error bars from 5 to ~30 km**
  (`outputs/phase2/sfd_T-all.png`). Per-crater agreement is ~55%, but misses and extras roughly
  balance (1,893 detections vs 1,979 catalog craters), so a *density* estimate, which is what most
  crater science uses, comes out close. Part of that balance comes from picking the F1-best cutoff,
  so treat it as encouraging, not proven.
- **5–10 km is the open question:** 587 of the 812 unmatched detections are there. The example
  tiles show some are real craters the catalog lacks (the degraded ~40 km crater in T1 is now found,
  and it's not in the catalog). Phase 4 adjudication will say how many.
- **Both detectors measure diameters ~5% smaller than the catalog.** Two independent methods agreeing
  suggests a systematic difference in where the rim is placed (JMARS three-point picks vs the
  detectors), rather than a detector bug.

**Known quirks / improvements for later:**
- The fine level never outputs craters above ~26 km, though it was trained on labels up to 40 km.
  The coarse level covers 30 km and up (recall 88% across 26–40 km), so nothing is lost, but coarse
  detections cluster just above their 30 km floor. An inference image pyramid would remove this
  handoff quirk.
- The training labels still treat uncataloged craters as "no crater". After Phase 4, the adjudicated
  craters can be added as labels and the model retrained.
- Checkpoint selection uses a noisy V1 metric; a larger validation area would make it steadier.

## 3.3b Retraining on registered labels (2026-11-03)

```
conda run -n mercury-craters python -m craters.phase4 register-labels   # ~25 min: both detectors over train + V1 tiles
conda run -n mercury-craters python -m craters.phase3 cnn               # ~40 min; model_name / use_registered_labels in config
```

**Why a separate registration:** the Phase 4 shift field is built from test-region detections, so it
had moved only 364 of 11,902 train craters and none in V1. `register-labels` runs the old CNN and the
tuned Hough over every train tile and every tile touching V1 (`data/detections_labels/`). It then
rebuilds the same north-south field (same `registration:` settings) into
`catalog_registered_labels.parquet`. That copy is used only for CNN training and V1 selection. Test
scoring still uses `catalog_registered.parquet`, unchanged.
- **Shift moved 10,350 train craters,** median −1.11 km, largest −6.7 km. In V1, 413 craters moved,
  median **+0.66 km**: the only region where the catalog sits *south* of the basemap.
- **The old CNN hadn't learned the shift:** before correction, its median train-area offset (−1.09 km)
  matches Hough's (−1.07 km).

**New model** `cnn_reg` (`data/models/cnn_reg_best.pt`; the old `cnn_best.pt` is kept):
- **Selected at epoch 8, score ≥ 0.2, V1 F1 0.600.** Its V1 F1 isn't comparable to the old 0.600:
  V1 labels moved too.
- **It makes fewer detections:** 2,723 on T1–T3, against 3,481.

**T1–T3, old `cnn` → new `cnn_reg`** (F1; precision / recall for "all" and 5–10 km):

| Catalog | Scoring | all | 5–10 | 10–20 | 20–50 | 50+ |
|---|---|---:|---:|---:|---:|---:|
| original | raw | 0.557 → **0.589** | 0.394 → 0.403 | 0.747 → 0.737 | 0.737 → 0.792 | 0.806 → 0.803 |
| registered | raw | 0.586 → **0.609** | 0.439 → 0.433 | 0.764 → 0.755 | 0.739 → 0.795 | 0.806 → 0.803 |
| registered | verdict-adjusted | 0.718 → **0.645** | 0.634 → 0.470 | 0.827 → 0.818 | 0.826 → 0.855 | 0.838 → 0.829 |

| Registered, raw | old P / R | new P / R |
|---|---|---|
| all | 0.600 / 0.572 | 0.755 / 0.510 |
| 5–10 km | 0.459 / 0.420 | 0.681 / 0.317 |

What it shows:
- **No 5–10 km gain.** Retraining traded recall for precision. Precision against the catalog rose
  from 46% to 68%, but recall fell from 42% to 32%. Raw F1 is flat at 5–10 km and up ~0.02 overall,
  mostly from 20–50 km.
- **The verdict-adjusted F1 dropped (0.718 → 0.645)** because the review shows many of the old model's
  extra 5–10 km detections were real craters, and the new model finds fewer of them.
  - The new model's adjusted numbers lean more on extrapolation: its own detections are mostly unreviewed.
  - Adding `cnn_reg` to the report changes the review item set, so the old CNN's adjusted F1
    reads 0.718 here, not 0.724.
- **The likely culprit is checkpoint selection, not the labels.** Epoch 8 is early, and V1 F1 is noisy
  (±0.05). Epoch 16 had V1 F1 0.589 at a much higher recall (0.77 vs 0.56). For a mapping tool, a
  recall-weighted or verdict-informed selection would probably serve better than raw V1 F1 against
  an incomplete catalog.
- `config/default.yaml` now points at `cnn_reg`. Set `model_name: cnn` and
  `use_registered_labels: false` to get back the original setup.

**For the global run:**
- Choose which checkpoint to use first. The old `cnn` has the best verdict-adjusted F1.
- Activate the env (or put `<env>\Library\bin` on PATH): plain `python.exe` without it crashes
  in matplotlib with Windows error 0xc06d007f (a DLL load failure).

## Next

- Phase 4 adjudication. Recommended next: it decides whether the 5–10 km "false positives" are real.
- 3.2 DeepMoon on the DEM (separate TensorFlow env and weights; DEM-limited to ≳ 10–15 km). Optional comparison.
- 3.4 Claude vision experiment (optional, exploratory, paid API calls).
