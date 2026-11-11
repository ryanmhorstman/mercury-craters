# mercury-craters

Automated crater detection on MESSENGER imagery of Mercury, compared against my
hand-built master's thesis crater catalog (Horstman 2016, NAU).

## Results so far

The published model is `cnn_reg_f2`, after calibration towards the thesis's definition of a
crater ([docs/global-run.md](docs/global-run.md)). It maps every crater of 5 km or more on Mercury
in about 7 minutes on one consumer GPU. In the three held-out test areas (T1–T3), craters ≥ 5 km:

| D (km) | Detections | Precision against the thesis | Precision after review (95%) |
|---|---:|---:|---|
| all | 1,980 | 0.67 | **0.81** (0.80–0.82) |
| 5–10 | 1,008 | 0.59 | 0.71 (0.70–0.73) |
| 10–20 | 598 | 0.77 | 0.89 (0.87–0.90) |
| 20–50 | 312 | 0.75 | 0.95 (0.93–0.96) |
| 50+ | 62 | 0.73 | 0.97 (0.92–0.98) |

- **Against the thesis alone**, the model finds 68% of the thesis's craters, and 67% of its
  detections are in the thesis.
- **Not every detection outside the thesis is wrong.** The thesis counted only craters with
  well-defined rims. 302 of the 638 detections it doesn't list were reviewed by eye, in random order
  ([docs/phase4-notes.md](docs/phase4-notes.md)). 42% of them are real craters, so about 1 in 5
  detections is false.
- **False detections are mostly small.** About 29% of 5–10 km detections are false, against 5% or
  fewer from 20 km up.
- The reviewer is the thesis author, and the queue held only detections the thesis lacks, so the
  review wasn't blind.

The elevation model independently confirms 1,160 craters ≥ 20 km that the thesis doesn't list
([docs/self-training.md](docs/self-training.md)). Training on them didn't make the model better.

**Against the published catalogs** ([docs/catalog-comparison.md](docs/catalog-comparison.md)), in
the test areas:

- The CNN agrees with the thesis about as well as Herrick et al. (2018), an independent hand-made
  catalog, does. Herrick lists 62% of the thesis's craters, and the CNN finds 67%.
- Counting a detection as correct when the thesis or Herrick lists it gives precision 0.79, close
  to the 0.81 from the review by eye.
- 96% of the DEM-confirmed craters are in Herrick's catalog: missing from the thesis, not from the
  literature.

## Setup

```
conda env create -f environment.yml
```

Data paths and every threshold live in `config/default.yaml`. Raw data goes in
`data/raw/` (gitignored; see `docs/phase0-inventory.md` for sources). The thesis catalog
workbook is not public; the pipeline expects a copy at `data/raw/Mercury Crater Catalogue 1.0.xlsx`.

## Phases

| Phase | Command | Notes |
|---|---|---|
| 1. Ground truth + tiling | `conda run -n mercury-craters python -m craters.phase1` | [docs/phase1-notes.md](docs/phase1-notes.md) |
| 2. Matching + metrics | `conda run -n mercury-craters python -m craters.phase2` | [docs/phase2-notes.md](docs/phase2-notes.md) |
| 3. Detectors | `conda run -n mercury-craters python -m craters.phase3 hough` / `cnn` | [docs/phase3-notes.md](docs/phase3-notes.md) |
| 4. Adjudication | `conda run -n mercury-craters python -m craters.phase4 review` / `report` | [docs/phase4-notes.md](docs/phase4-notes.md) |
| Global run | `conda run -n mercury-craters python -m craters.phase3 global`, then `python -m craters.compare_global` | [docs/global-run.md](docs/global-run.md) |
| Calibrated run | `python -m craters.calibrate`, then `python -m craters.compare_global --calibrated` | [docs/global-run.md](docs/global-run.md) |
| Self-training | `python -m craters.demcheck`, then `python -m craters.selftrain` | [docs/self-training.md](docs/self-training.md) |
| Supplemented catalog | `python -m craters.supplement`, then `python -m craters.supplement --score` | [docs/self-training.md](docs/self-training.md) |
| Published catalogs | `python -m craters.compare_catalogs` (Herrick 2018, La Grassa 2025) | [docs/catalog-comparison.md](docs/catalog-comparison.md) |
| Retrain comparison | `python -m craters.retrain` (6 trainings, about 4 hours of GPU) | [docs/self-training.md](docs/self-training.md) |

Tests: `conda run -n mercury-craters python -m pytest`

## Docs

- Plan: [docs/PLAN.md](docs/PLAN.md)
- Phase 0 inventory: [docs/phase0-inventory.md](docs/phase0-inventory.md)

## License

Code is released under the [MIT License](LICENSE). The thesis catalog and the USGS
basemap are not covered by it and are not distributed with this repository.
