# Comparison with two published Mercury catalogs

    python -m craters.compare_catalogs     # ~5 min; writes outputs/compare/

The thesis catalog and the CNN run (`cnn_reg_f2`, calibrated) are compared here with:

- **Herrick et al. 2018**, *JGR Planets*,
  [doi:10.1029/2017JE005516](https://doi.org/10.1029/2017JE005516). Hand-mapped, D > 5 km, obvious
  secondaries left out. Supporting tables S1 (5–10 km, 14,574 craters) and S2 (> 10 km, 17,038).
- **La Grassa et al. 2025**, *Remote Sensing* 17:3287,
  [mdpi.com/2072-4292/17/19/3287](https://www.mdpi.com/2072-4292/17/19/3287). Catalog ME6M300TGT,
  made by deep learning (YOLOLens) on the image mosaic and the DEM: 6.3 million craters from
  0.4 km, [figshare doi:10.6084/m9.figshare.30187999](https://doi.org/10.6084/m9.figshare.30187999),
  CC BY 4.0. Its diameter is the mean of the box width and height.

Neither is in git. To rebuild, save Herrick's two crater tables (`.xlsx`) in
`data/raw/herrick_2018/` and the figshare zip as `data/raw/la_grassa_2025/ME6M300TGT.zip`.

All matching uses the project's rule: centres within half a diameter, diameters within 0.75–1.33×.
"The other lists x%" means x% of one catalog's craters have a match in the other.

## Positions

Herrick and La Grassa agree with each other to within 0.2 km. Both sit about 2 km west of the
craters on the USGS basemap the CNN ran on, mostly within ±30° latitude:

| latitude | Herrick: north / east (km) | La Grassa: north / east (km) |
|---|---|---|
| −90 to −30 | +0.5 / −0.8 | +0.3 / −0.2 |
| −30 to 30 | +0.1 / −2.1 | 0.0 / −2.2 |
| 30 to 90 | −0.2 / −1.1 | −0.3 / −2.8 |

These median offsets, from ≥ 10 km craters matched to the CNN run, are removed before comparing.
They matter only at 5–10 km, where 2 km is 20–40% of a diameter. Without the shift, Herrick lists
40% of the thesis's 5–10 km craters in T1–T3, against 44% with it; from 10 km up nothing changes.

The thesis was registered north–south only (Phase 4), and the same east–west shift shows up
there: the thesis sits 2.3 km east of both published catalogs, and 0.65 km east of the CNN.

## How many craters

| | 5–10 km | 10–20 km | 20–50 km | ≥ 50 km | all ≥ 5 km |
|---|---:|---:|---:|---:|---:|
| thesis | 9,304 | 6,710 | 3,685 | 692 | 20,391 |
| CNN, calibrated | 12,475 | 8,667 | 4,485 | 931 | 26,558 |
| Herrick 2018 | 14,540 | 9,346 | 5,245 | 2,440 | 31,571 |
| La Grassa 2025 | 231,816 | 32,388 | 7,436 | 1,561 | 273,201 |

- **La Grassa is a different kind of catalog.** It lists 25× more 5–10 km craters than the thesis.
  It aims at everything down to 0.4 km, small secondaries and very degraded craters included.
- **Herrick lists 3.5× more ≥ 50 km craters than the thesis.** These are the big, degraded
  craters the thesis left out by counting only well-defined rims (see below).

## Agreement, in the held-out test areas T1–T3

The CNN never trained on T1–T3, so only there is the comparison fair to it. Share of the row
catalog's craters that the column catalog lists:

| | thesis lists | CNN lists | Herrick lists | La Grassa lists |
|---|---|---|---|---|
| **thesis's craters** | — | 67% | 62% | 90% |
| **CNN's craters** | 67% | — | 66% | 89% |
| **Herrick's craters** | 68% | 72% | — | 87% |
| **La Grassa's craters** | 10% | 10% | 9% | — |

By size, for the thesis and Herrick:

| | 5–10 km | 10–20 km | 20–50 km | ≥ 50 km |
|---|---|---|---|---|
| Herrick lists, of the thesis's craters | 44% | 78% | 95% | 100% |
| CNN finds, of the thesis's craters | 55% | 82% | 86% | 82% |
| thesis lists, of Herrick's craters | 65% | 78% | 73% | 36% |
| CNN finds, of Herrick's craters | 67% | 83% | 80% | 39% |

- **The CNN agrees with the thesis about as well as the two hand-made catalogs agree with each
  other.** Herrick lists 62% of the thesis's craters, and the CNN finds 67%. The thesis lists 68% of
  Herrick's, and the CNN finds 72%.
- **Two experts disagree most at 5–10 km.** Only 44% of the thesis's 5–10 km craters are in
  Herrick, and 65% of Herrick's are in the thesis. That fits the ±45% spread between expert crater
  counters found by Robbins et al. 2014 ([doi:10.1016/j.icarus.2014.02.022](https://doi.org/10.1016/j.icarus.2014.02.022)).
  Part of it is definitional: Herrick left out obvious secondaries, and the thesis left out
  craters without well-defined rims.
- **From 20 km up, Herrick contains almost every thesis crater (95–100%), but the thesis has only
  36% of Herrick's ≥ 50 km craters.** The thesis's missing big craters are real, and Herrick
  counted them. The CNN, trained on the thesis, misses them too (39%).
- **La Grassa contains about 90% of everyone's craters, but only about 10% of its own craters are
  in any other catalog,** and 4–6% at 5–10 km. Scoring the CNN against it would say little.

Whole-planet numbers are in `outputs/compare/agreement.csv`. They are similar, except that the CNN
looks closer to the thesis there, because it trained on the thesis outside T1–T3.

## The CNN against the thesis and Herrick together

A CNN detection counts as correct when either hand catalog lists it (T1–T3, ≥ 5 km):

| D (km) | precision, thesis | precision, thesis + Herrick | precision after your review |
|---|---|---|---|
| all | 0.67 | 0.79 | 0.81 |
| 5–10 | 0.59 | 0.69 | 0.71 |
| 10–20 | 0.77 | 0.87 | 0.89 |
| 20–50 | 0.75 | 0.92 | 0.95 |
| ≥ 50 | 0.73 | 0.98 | 0.97 |

Two independent routes give nearly the same precision: Herrick's catalog, and the review of 302
detections by eye ([phase4-notes.md](phase4-notes.md)). Against both hand catalogs, recall falls from
0.67 to 0.61 overall and to 0.40 at ≥ 50 km, because Herrick's extra big craters count too.

## The supplement and the review, checked against Herrick

Share of each group listed in Herrick 2018 and in La Grassa 2025:

| group | n | in Herrick | in La Grassa |
|---|---:|---:|---:|
| DEM-confirmed CNN craters (`dem_confirmed`, ≥ 20 km) | 774 | **96%** | 88% |
| DEM scan + CNN (`dem_scan_cnn`, ≥ 40 km) | 386 | **88%** | 76% |
| your review: judged real, ≥ 10 km | 67 | 87% | 91% |
| your review: judged not real, ≥ 10 km | 34 | 24% | 71% |
| your review: judged real, 5–10 km | 57 | 39% | 91% |
| your review: judged not real, 5–10 km | 136 | 22% | 66% |

- **The 1,160 DEM-backed craters are real but not new.** 96% and 88% of them are in Herrick. They
  are missing from the thesis, not from the literature. That confirms the DEM checks, and it means
  the supplemented catalog is mostly the thesis plus Herrick's extra big craters.
- **Your verdicts agree with Herrick at ≥ 10 km:** 83% agreement, Cohen's κ = 0.63. At 5–10 km
  agreement is 66% (κ = 0.17). Herrick left out obvious secondaries, which you may well have judged
  real, so it is not a second reviewer there.
- **La Grassa doesn't separate them.** It lists 66–71% of the circles you judged not real.

## What this means

- The CNN, trained on one person's strict catalog, ends up as close to an independent expert
  catalog as that person's own catalog is.
- Most of what looked like CNN false positives from 10 km up are craters Herrick also mapped.
- What the thesis left out is now measured in two independent ways: DEM-backed craters 96% confirmed
  by Herrick, and the thesis listing 36% of Herrick's ≥ 50 km craters.
- The 5–10 km bin stays the hardest, for people as much as for the CNN.
