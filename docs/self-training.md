# Self-training with a DEM check

    python -m craters.demcheck      # fit the DEM bowl check
    python -m craters.selftrain     # rounds of pseudo-labelling + fine-tuning, gated on V1

The thesis catalog is the source of truth, but it was built from the image
mosaic alone and only counted clearly rimmed craters. Some real craters are
missing from it. Until now those taught the CNN "no crater here". This step
looks for them with evidence the catalog never used, the USGS 665 m DEM, and
feeds the confident ones back into training without anyone reviewing craters by eye.

## The DEM bowl check (`src/craters/demcheck.py`)

For a circle (lat, lon, D) the DEM is sampled on a polar grid: the floor (inside
0.4 radii), the highest point on each of 36 azimuths between 0.8 and 1.2 radii
(the rim), and the ground outside (1.6-2 radii). A circle passes when the rim
stands high enough above the floor on enough azimuths and the floor lies below
the surroundings.

The thresholds are fitted per diameter band on the train area. As many catalog
craters as possible should pass, while at most 1% of decoys may: decoys are
catalog circles moved 2-3 diameters onto ground the catalog says is empty.

| D (km) | catalog craters passing (train) | decoys passing (train) |
|---|---|---|
| 20-30 | 40% | 0.4% |
| 30-40 | 60% | 0.4% |
| 40-60 | 94% | 0.7% |
| 60-100 | 88% | 0.9% |
| 100-400 | 82% | 0% |

The check is strict by design. It misses many real craters, but when it says
yes it is almost never on empty ground. Below 20 km it can't separate catalog
craters from decoys at all (median depth/D of 10-20 km catalog craters is
0.001, the same as the decoys), so the DEM's effective resolution is far
coarser than its pixels, and no crater under 20 km is confirmed.

## DEM-confirmed craters missing from the catalog

A detection from the calibrated global run counts as a confirmed crater missing
from the catalog when it:

- matches no catalog crater;
- passes the bowl check;
- is not a copy, at a different size, of a catalog crater;
- is not centred inside a bigger catalog crater (a peak ring has a bowl too);
- is not drawn round a smaller catalog crater near its centre.

781 detections qualify (`outputs/selftrain/new_craters.csv`). 349 of them lie in
the areas excluded from training because the catalog looked incomplete there,
and 281 in the train area.

![A random sample of 20: basemap with the detection (yellow) and catalog craters (blue), then DEM hillshade](figures/dem_confirmed_sample.png)

All 20 craters in a random sample look like real craters. Several are fresh,
with central peaks, and the nearest catalog crater is often 25-60 km away and
much smaller.

| D (km) | 20-30 | 30-40 | 40-60 | 60-100 | 100-400 |
|---|---|---|---|---|---|
| confirmed, not in the catalog | 156 | 271 | 238 | 96 | 20 |

## The loop (`src/craters/selftrain.py`)

Each round runs these steps, starting from the best model so far:

1. **Pseudo-labels, train area only.** Confirmed craters become positives.
   Other non-catalog detections with score >= 0.5 become "ignore".
2. **Fine-tune** that model on the catalog plus the pseudo-labels.
3. **Global run and calibration.**
4. **Gate.** The round is kept only if V1 F1 against the catalog doesn't drop.

The catalog is never edited, and V1, T1-T3 and their buffers never get
pseudo-labels. T1-T3 are reported, never used to decide.

## First run: rejected

| round | model | new craters / ignored | V1 F1 | T1-T3 F1 | kept |
|---|---|---|---|---|---|
| 0 | cnn_reg_f2 | - | 0.678 | 0.675 | - |
| 1 | cnn_reg_f2_st1 | 281 / 38 | 0.635 | 0.642 | no |

The gate did its job, and `cnn_reg_f2` stays the published model. The drop
probably comes from the fine-tune itself rather than the labels. 281 positives
are about 2% of the training labels. The fine-tune restarted the learning rate
at 2e-4, and V1 F-beta during it peaked at 0.697, against 0.735 for the model it
started from. V1 F-beta also swings by +-0.1 between evaluations in the original
training (`data/models/cnn_reg_f2_history.csv`). A gentler fine-tune (lower
learning rate, more frequent checkpoints), with a control run without pseudo-labels, would
separate the two effects.
