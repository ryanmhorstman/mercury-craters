# mercury-craters

Automated crater detection on MESSENGER imagery of Mercury, compared against my
hand-built master's thesis crater catalog (Horstman 2016, NAU).

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

Tests: `conda run -n mercury-craters python -m pytest`

## Docs

- Plan: [docs/PLAN.md](docs/PLAN.md)
- Phase 0 inventory: [docs/phase0-inventory.md](docs/phase0-inventory.md)
