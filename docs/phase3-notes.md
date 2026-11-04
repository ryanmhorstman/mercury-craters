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
- `config/default.yaml` pointed at `cnn_reg` at this stage. Set `model_name: cnn` and
  `use_registered_labels: false` to get back the original setup.

**Recall-weighted selection: `cnn_reg_f2`.** The checkpoint and cutoff now maximise V1 F-beta
(`detectors.cnn.selection_beta: 2`; 1 = F1, which is how `cnn` and `cnn_reg` were chosen).
- **Same training run otherwise** (same seed and labels as `cnn_reg`). F2 picked **epoch 16, score ≥ 0.15**:
  V1 F2 0.691, F1 0.517, recall 0.89.
- **It makes 7,348 detections on T1–T3.**

T1–T3, all three models scored in one Phase 4 report (P / R / F1):

| Model | Raw, original catalog: all | Raw, registered: all | Raw, registered: 5–10 | Verdict-adjusted: all | Verdict-adjusted: 5–10 |
|---|---|---|---|---|---|
| `cnn` | 0.57 / 0.54 / 0.557 | 0.60 / 0.57 / 0.586 | 0.46 / 0.42 / 0.439 | 0.80 / 0.44 / 0.569 | 0.75 / 0.31 / 0.436 |
| `cnn_reg` | 0.73 / 0.49 / 0.589 | 0.76 / 0.51 / 0.609 | 0.68 / 0.32 / 0.433 | 0.88 / 0.35 / 0.499 | 0.84 / 0.19 / 0.303 |
| `cnn_reg_f2` | 0.33 / 0.76 / 0.458 | 0.35 / 0.82 / 0.494 | 0.24 / 0.79 / 0.370 | 0.70 / 0.88 / **0.780** | 0.65 / 0.89 / **0.755** |

What it shows:
- **Best after the verdicts: `cnn_reg_f2`,** so the config now points at it.
- **Against the raw catalog it scores worst:** its precision is 35%. Many of its extra detections are
  craters the catalog lacks, and they count as false positives there.
- **The adjusted numbers in this table can't be compared with the earlier tables.** Adding a
  high-recall detector adds many new review items, mostly uncatalogued craters it found. That grows
  the estimated set of real craters, so the other models' adjusted recall falls (the old `cnn` drops
  from 0.643 to 0.441).
- **Caveat: only 214 of 4,599 items are reviewed now.** Most of `cnn_reg_f2`'s extra detections are
  unreviewed, and their verdicts are extrapolated from the reviewed rate in the same kind × size
  bin. Before trusting 0.780, run `phase4 build` and review a few hundred more items, chiefly its
  5–10 km detections.

**For the global run:**
- Confirm the checkpoint choice first. `cnn_reg_f2` leads after the verdicts, pending more review.
- Activate the env (or put `<env>\Library\bin` on PATH): plain `python.exe` without it crashes
  in matplotlib with Windows error 0xc06d007f (a DLL load failure).

## 3.3c East-west twins: one crater, two circles (fixed 2026-11-03)

The map viewer showed `cnn_reg_f2` circles in side-by-side pairs: same size, centres 0.5–1 D apart,
nearly always offset east-west. The overlap merge only joins centres within 0.5·D, so both survive.

**How common, against the catalog's own neighbours** (D ≥ 5 km; "in a pair" = has a neighbour of
diameter ratio 0.75–1.33 whose centre is 0.5–1 D away):

| Set | Circles | In a pair | Pairs offset mostly E-W | Pairs from one tile |
|---|---:|---:|---:|---:|
| Thesis catalog, whole planet | 20,391 | 3.9% | 48% | — |
| Thesis catalog, T1–T3 | 1,982 | 3.7% | 38% | — |
| Thesis catalog, V1 | 487 | 3.3% | 1 of 8 | — |
| `cnn` (T1–T3) | 3,481 | 27% | 84% | 74% |
| `cnn_reg` (T1–T3) | 2,723 | 19% | 73% | 60% |
| `cnn_reg_f2` (T1–T3) | 7,348 | **60%** | 93% | 83% |
| Hough (T1–T3) | 2,105 | 0.9% | — | — |

Real overlapping same-size craters exist (~4%, no preferred direction). The CNN pairs are an
artefact of all three models. It's worst in `cnn_reg_f2` because its low cutoff (0.15) admits the
weaker member, and Hough doesn't show it.

**Cause: a second, weaker heatmap peak west of the real one.** Decoding the V1 tiles and looking
inside single tiles (1,091 pairs):
- **The second peak is 8–9 output cells (32–36 px) away:** median 8.1 cells, and 89% sit *west* of
  the stronger peak, with dy = 0 ± 1 cell. That is ~1.5 crater radii for typical 5–10 km craters,
  far outside the 3×3 window `decode` used to find local maxima.
- **It's the weaker peak:** score 0.19 against 0.23. The heat between the two peaks dips to ~60%
  of the weaker one, so they are genuinely two maxima, not one smeared blob.
- **The stronger peak sits on the crater.** Where a catalog crater lies under the pair, it is at
  the stronger peak (median position 0.0 along the pair axis, 0 = stronger, 1 = weaker).
- **Likely origin:** an almost fixed pixel offset in one direction points at the lighting. The
  low-incidence mosaic lights craters from one side, giving a bright rim on one side and a dark
  shadow on the other. The network probably fires partly on that rim/shadow pattern as if it were a
  centre. This wasn't tested further: the fix doesn't depend on it.
  The training targets aren't the cause: they put one Gaussian on each crater centre, and
  registration moved labels only north-south. So no retraining is needed.

**Fix: size-aware suppression in `decode`** (`detectors.cnn.suppress_twins`). After peak picking,
circles are visited best score first. Each one drops every weaker circle of similar radius
(ratio 0.75–1.33) whose centre lies within `radius_frac` radii. Settings are in
`detectors.cnn.twin_suppression`.

**Choosing `radius_frac` on V1 only** (each model at its own cutoff; "in a pair" as above, catalog
V1 3.3%):

| `radius_frac` | `cnn_reg_f2`: V1 F2 / F1 / in a pair | `cnn_reg`: F1 / in a pair | `cnn`: F1 / in a pair |
|---|---|---|---|
| off | 0.691 / 0.516 / 55% | 0.600 / 22% | 0.604 / 19% |
| 1.5 | 0.711 / 0.551 / 41% | 0.608 / 17% | 0.617 / 13% |
| 1.75 | 0.724 / 0.579 / 26% | 0.621 / 10% | 0.623 / 9% |
| **2.0** | **0.745 / 0.620 / 5.1%** | **0.628 / 3.0%** | **0.637 / 2.8%** |
| 2.25 | 0.752 / 0.646 / 3.9% | 0.621 / 2.8% | 0.638 / 2.2% |
| 2.5 | 0.756 / 0.659 / 3.3% | 0.620 / 2.8% | 0.645 / 2.0% |

- **Chosen: 2.0, i.e. centres within one diameter.** That brings the pair rate down to the
  catalog's level, and every model's best V1 cutoff is unchanged (0.15 / 0.2 / 0.15).
- Larger values still nudge F2 up, but they start removing separate craters whose rims don't even
  touch. The gain comes from deleting near neighbours, not from removing more twins.
- The cost of 2.0: of two genuinely overlapping same-size craters, the weaker one is dropped.
  By the catalog baseline that's at most ~2% of real craters.
- **Tile-overlap merging uses the same rule** (`detectors.cnn.overlap_rule`: centres within 1 D,
  ratio 0.75–1.33, CNN only; Hough keeps the 0.5·D rule).
  - With the decode fix alone, T1–T3 `cnn_reg_f2` still had 6.3% of circles in pairs. All of those
    came from two different tiles, 91% east-west: a twin whose partner was found in the
    neighbouring tile.
  - On V1, widening the merge is neutral: F2 0.745 → 0.743, F1 0.620 → 0.624. It takes the pair
    rate to ~0.5%, so it's used for consistency.

## 3.3d T1–T3 after the twin fix

```
python -m craters.phase3 cnn --skip-training --model cnn          # likewise cnn_reg, cnn_reg_f2
python -m craters.phase2 && python -m craters.phase4 build && python -m craters.phase4 report
```

Same weights and cutoffs; only decoding and merging changed. The pre-fix detection files are kept
in `data/detections_pre_twinfix/`.

| Model | T1–T3 detections | In a same-size pair (catalog 3.7%) |
|---|---:|---:|
| `cnn` | 3,481 → 2,999 | 27% → 0.7% |
| `cnn_reg` | 2,723 → 2,469 | 19% → 0.2% |
| `cnn_reg_f2` | 7,348 → **5,158** | 60% → 0.4% |

Before → after, registered catalog, P / R / F1 for "all", from one Phase 4 report each:

| Model | Raw: all | Raw: 5–10 F1 | Verdict-adjusted: all | Verdict-adjusted: 5–10 F1 |
|---|---|---|---|---|
| `cnn` | 0.60 / 0.57 / 0.586 → 0.70 / 0.56 / **0.621** | 0.439 → 0.479 | 0.83 / 0.41 / 0.548 → 0.87 / 0.47 / **0.607** | 0.419 → 0.465 |
| `cnn_reg` | 0.76 / 0.51 / 0.609 → 0.81 / 0.49 / **0.613** | 0.433 → 0.434 | 0.90 / 0.32 / 0.469 → 0.92 / 0.38 / **0.539** | 0.279 → 0.334 |
| `cnn_reg_f2` | 0.35 / 0.82 / 0.494 → 0.51 / 0.79 / **0.618** | 0.370 → 0.505 | 0.75 / 0.88 / 0.810 → 0.80 / 0.80 / **0.801** | 0.797 → 0.768 |

("Before" is the last pre-fix report, with 462 items reviewed; the §3.3b table above was made at
214.)

What it shows:
- **Raw precision rises sharply and recall barely moves.** For `cnn_reg_f2`, raw F1 goes from
  0.494 to 0.618, and it is now the best raw F1 of the three. Each twin was an unmatched
  detection, i.e. a false positive against the catalog.
- **Recall drops about 3 points** (`cnn_reg_f2` 0.819 → 0.789). Most of it is at 5–10 km, plus a few
  ≥ 20 km craters lost to the wider merge.
- **Verdict-adjusted F1 for `cnn_reg_f2` is about unchanged (0.810 → 0.801), but the balance moved:**
  adjusted precision went up (0.75 → 0.80) and adjusted recall down (0.88 → 0.80). Many twins had
  been reviewed as "real" (`docs/phase4-notes.md`), which counted them as found real craters.
- The older models gain in adjusted F1, because ghost items proposed only by `cnn_reg_f2` no longer
  count as craters they missed.

## 3.3e Large craters: what didn't work, and calibration (2026-11-04)

In the global run, 33 of the 235 catalog craters ≥ 90 km had no match. A contact sheet of all 33
showed two kinds of miss:
- **Found but drawn too small** (most). In peak-ring basins (Rodin, Scarlatti, Boethius) the CNN
  circles the inner ring. Elsewhere (Homer, for example) it circles a ring at ~0.7 D, which fails the
  0.75 ratio rule.
- **Not found at all.** These are the most degraded basins (Mendelssohn, several at 160–170 km).

Three attempts to close the gap:
1. **A lower coarse cutoff** (0.05 instead of 0.15). The missed basins are not sitting just below
   threshold: recall at 130–200 km went only from 54 to 58 of 66, while precision for ≥ 40 km
   detections fell from 0.61 to 0.45.
2. **A third, very coarse level** (2.66 km/px, 768 px tiles with 1,022 km overlap), using the same
   weights. With the 2× shrink, a 200 km crater is ~38 px, a size the model handles at other levels.
   Combined with the coarse level, it added only 4–6 craters ≥ 90 km and cost a lot of precision.
   The model doesn't recognize these basins at any scale. The experiment was dropped and not added
   to the tiling.
3. **A coarse-only fine-tune** (`phase3 finetune-coarse` → `cnn_reg_f2_coarse_ft`). This was 10
   epochs at lr 1e-4, with tiles holding craters ≥ 80 km sampled more often
   (`detectors.cnn.coarse_finetune`). It improved recall and precision *in the training area*
   (≥ 200 km 9/12 → 12/12, coarse precision 0.59 → 0.75). Everywhere else it was slightly *worse*:
   recall at 90–130 km went from 132/147 to 124/147. That is memorizing the training basins, not
   learning; the training area holds only ~40 craters ≥ 130 km. It is left switched off
   (`detectors.cnn.level_models: {}`). The plumbing stays, so a better-trained coarse model can be
   dropped in.

**What did help: calibration** (`python -m craters.calibrate`; details in `docs/global-run.md`).
- A per-level diameter correction, fitted on train + V1 matches. It recovers the "drawn ~7% small"
  share of the big misses.
- Per-size score cutoffs, chosen on V1 F1, bring the run closer to the thesis's strict defined-rim
  criterion.

T1–T3 F1 goes from 0.618 to 0.674, and catalog recall ≥ 50 km stays at ~0.92.

The remaining large-crater gap needs more big-crater training examples than the training area
holds. One route is labelling ≥ 50 km craters in the excluded areas: they were left out of training
for thin *small*-crater coverage, which shouldn't affect big craters.

## Next

- Phase 4 adjudication. Recommended next: it decides whether the 5–10 km "false positives" are real.
- 3.2 DeepMoon on the DEM (separate TensorFlow env and weights; DEM-limited to ≳ 10–15 km). Optional comparison.
- 3.4 Claude vision experiment (optional, exploratory, paid API calls).
