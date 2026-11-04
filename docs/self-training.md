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

## Results: both rounds rejected, fine-tuning is the problem

All F1 values are for the calibrated global run, scored against the catalog.

| run | recipe | new craters / ignored | V1 F1 | T1-T3 F1 | kept |
|---|---|---|---|---|---|
| cnn_reg_f2 (start) | - | - | 0.678 | 0.675 | - |
| round 1, first try | 16 epochs, lr 2e-4, V1 every 4 | 281 / 38 | 0.635 | 0.642 | no |
| control | 12 epochs, lr 5e-5, V1 every 2, no pseudo-labels | 0 / 0 | 0.587 | 0.583 | (reference) |
| round 1, retry | same as control | 281 / 38 | 0.595 | 0.589 | no |

The gate did its job, and `cnn_reg_f2` stays the published model.

The control fine-tune has no pseudo-labels, and it does worse than the start
model: V1 F2 falls from 0.735 to 0.64 at its very first check, after 2 epochs.
So fine-tuning from `cnn_reg_f2` loses accuracy whatever the labels.
With the same recipe, the pseudo-labels then do slightly better than the
control (V1 +0.009, T1-T3 +0.006), but that is within the noise.

Most likely, `cnn_reg_f2` is a lucky checkpoint of a noisy training run. Its V1
F-beta went 0.66, 0.53, **0.735**, 0.59 at epochs 8-20
(`data/models/cnn_reg_f2_history.csv`), and T1-T3 confirms it is genuinely good,
not just good on V1. Any further training moves away from that point.

To test the pseudo-labels properly, the training itself needs to be steadier:
average the weights over time (EMA/SWA), then compare full retrains with and
without pseudo-labels over several seeds. That costs hours of GPU time per
comparison, and is left for later.

The confirmed craters stand on their own: `outputs/selftrain/new_craters.csv`
is a supplement to the catalog, confirmed by evidence the thesis didn't use,
whether or not they ever help train the CNN. The rejected models and runs are
kept, outside git, in `data/models/selftrain_st1_rejected/` and
`data/models/selftrain_retry_rejected/`.

## DEM scan: big craters neither the catalog nor the CNN has

    python -m craters.demscan       # ~11 min on one CPU core; writes outputs/selftrain/demscan_new.csv

The CNN misses many big, degraded craters with soft rims and smooth floors.
The thesis didn't count them, so in training the CNN was taught they are "no
crater", and self-training can't help with craters the CNN never detects.
`src/craters/demscan.py` therefore uses the bowl check as a detector of its own.

The scan tests circles on a grid over the whole planet: 40 to 400 km in steps
of 1.15x, with centres 0.2 D apart, 4.8 million circles in all. The best bowl
among overlapping circles wins, and inner rings are dropped.

The rule is tuned on the train area:

- **Recall:** the share of catalog craters >= 40 km the scan finds.
- **Known:** the share of scan craters that match a catalog crater or a CNN
  detection, or are a mis-sized copy of one.

A 28 km crater shows up in the scan as a 40 km circle around it, so those
copies count as known.

| rim on every azimuth, depth/D >= | train recall | train known |
|---|---|---|
| 0.01 (chosen: best recall with known >= 0.8) | 0.76 | 0.81 |
| 0.02 | 0.68 | 0.82 |
| 0.03 | 0.42 | 0.86 |

With rims as loose as 75% of azimuths, the scan finds 92% of the catalog's
>= 40 km craters, so the DEM does see them. The trade-off is precision.

The chosen rule gives **827 scan craters >= 40 km in neither set**
(`demscan_new.csv`). They are **candidates, not confirmed**. In a random sample
of 24 from the train area, about half are real degraded craters. The rest are
shadowed scarps, mosaic seams, rough terrain, or circles drawn round a smaller
crater.

![24 random scan craters (green in the viewer) in neither the catalog (blue) nor the CNN run (orange)](figures/demscan_sample.png)

The map viewer shows them as "DEM scan candidates", off by default. Turning
them into training labels would need a better filter first. For example, the
CNN could be run on these circles at a low score cutoff, and only circles both
methods support would be kept.
