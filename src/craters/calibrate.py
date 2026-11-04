"""Calibrate the global run towards the thesis catalog's own definition of a crater.

    python -m craters.phase3 global
    python -m craters.calibrate                 # writes global_<run>_calibrated.parquet
    python -m craters.compare_global --calibrated

The raw run is tuned for recall (V1 F2, one score cutoff for every size). The
thesis counted only craters with clearly defined rims, so to reproduce it the
run is adjusted in two steps, both fitted without looking at T1-T3:

1. Diameter. Matched pairs in the train and V1 areas give, per tiling level,
   the median catalog / detection diameter ratio in bands of detected diameter.
   Each detection is scaled by that ratio, interpolated in log diameter.
   (The fine level draws craters ~5-9% small; the coarse level draws >= 70 km
   craters ~6-7% small.)
2. Score cutoff by size. For each band of corrected diameter, the cutoff
   with the best V1 F1 inside that band. Bands are independent because a
   detection only matches a catalog crater within a 0.75-1.33 diameter ratio.
   Cutoffs below the run's own cutoff can't be chosen, and a band with too
   few V1 craters to choose on (the big ones) keeps the run's cutoff.

Writes data/detections/global_<run>_calibrated.parquet and
outputs/global/calibration.json.
"""
import json

import numpy as np
import pandas as pd

from .compare_global import AREA_ORDER, WHOLE_PLANET, area_of
from .config import load_config, run_name
from .metrics import label_outcomes, score


def fit_diameter_correction(catalog, detections, cfg):
    """Per level: band centres (km) and median log(catalog D / detected D) of train + V1 matches."""
    cal = cfg["calibration"]
    cat, det = label_outcomes(catalog, detections, [WHOLE_PLANET], cfg["match"], 0)
    det["area"] = area_of(det["lat"], det["lon"], cfg["regions"])
    fit = det[(det["outcome"] == "TP") & det["area"].isin(AREA_ORDER[1:3])]
    log_ratio = np.log(cat["diameter_km"].to_numpy()[fit["matched_cat"].astype(int)] / fit["diameter_km"].to_numpy())
    fit = fit.assign(log_ratio=log_ratio)
    correction = {}
    for level, edges in cal["diameter_bands_km"].items():
        rows = fit[fit["level"] == level]
        bands = pd.cut(rows["diameter_km"], edges)
        stats = rows.groupby(bands, observed=False)["log_ratio"].agg(["count", "median"])
        ok = stats["count"] >= cal["min_pairs"]
        centres = np.sqrt(np.array(edges[:-1]) * np.array(edges[1:]))
        correction[level] = {"centre_km": centres[ok.to_numpy()].round(2).tolist(),
                             "log_ratio": stats["median"][ok].round(4).tolist(),
                             "pairs": stats["count"][ok].astype(int).tolist()}
    return correction


def apply_diameter_correction(detections, correction):
    out = detections.copy()
    out["diameter_km"] = out["diameter_km"].astype(float)
    for level, c in correction.items():
        rows = out["level"] == level
        if not c["centre_km"] or not rows.any():
            continue
        log_ratio = np.interp(np.log(out.loc[rows, "diameter_km"]), np.log(c["centre_km"]), c["log_ratio"])
        out.loc[rows, "diameter_km"] *= np.exp(log_ratio)
    return out


def choose_size_cutoffs(catalog, detections, cfg, floor):
    """Per band of diameter: the score cutoff with the best V1 F1 in that band."""
    cal = cfg["calibration"]
    edges = cal["cutoff_bands_km"]
    v1 = list(cfg["regions"]["validation"].values())
    cutoffs = [c for c in cal["score_cutoffs"] if c >= floor]
    best = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        scores = []
        for cutoff in cutoffs:
            cat, det = label_outcomes(catalog, detections[detections["score"] >= cutoff], v1, cfg["match"], edges[0])
            in_cat = cat["counted"] & cat["diameter_km"].between(lo, hi, inclusive="left")
            in_det = det["counted"] & det["diameter_km"].between(lo, hi, inclusive="left")
            scores.append(score(cat[in_cat], det[in_det])["f1"])
        n = int(in_cat.sum())
        # Too few V1 craters to choose on (the big bands): keep the run's own cutoff.
        i = int(np.nanargmax(scores)) if n >= cal["min_v1_craters"] else 0
        best.append({"min_km": lo, "max_km": hi, "v1_craters": n, "score_cutoff": cutoffs[i],
                     "v1_f1": round(float(scores[i]), 4)})
    return best


def apply_size_cutoffs(detections, cutoffs):
    edges = [c["min_km"] for c in cutoffs] + [cutoffs[-1]["max_km"]]
    band = np.clip(np.searchsorted(edges, detections["diameter_km"].to_numpy(), side="right") - 1, 0, len(cutoffs) - 1)
    needed = np.array([c["score_cutoff"] for c in cutoffs])[band]
    return detections[detections["score"].to_numpy() >= needed].reset_index(drop=True)


def calibrate(cfg, record_path=None, quiet=False):
    """Calibrate the global run of detectors.cnn; returns the calibration record
    (written to outputs/global/calibration.json unless record_path is given)."""
    params = cfg["detectors"]["cnn"]
    name = run_name(params)
    detections = pd.read_parquet(cfg["paths"]["detections"] / f"global_{name}.parquet")
    catalog = pd.read_parquet(cfg["paths"]["catalog_registered_labels"])
    run_cutoff = json.loads((cfg["paths"]["models"] / f"{params['model_name']}_best.json").read_text())["score_cutoff"]

    correction = fit_diameter_correction(catalog, detections, cfg)
    corrected = apply_diameter_correction(detections, correction)
    cutoffs = choose_size_cutoffs(catalog, corrected, cfg, floor=run_cutoff)
    calibrated = apply_size_cutoffs(corrected, cutoffs)

    out = cfg["paths"]["detections"] / f"global_{name}_calibrated.parquet"
    calibrated.to_parquet(out, index=False)
    record = {"run": name, "diameter_correction": correction, "size_cutoffs": cutoffs,
              "detections_raw": len(detections), "detections_calibrated": len(calibrated)}
    record_path = record_path or cfg["paths"]["global"] / "calibration.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2))
    if quiet:
        return record
    for level, c in correction.items():
        print(f"{level}: diameter x " + ", ".join(f"{k:g} km {np.exp(r):.3f}" for k, r in zip(c["centre_km"], c["log_ratio"])))
    print(pd.DataFrame(cutoffs).to_string(index=False))
    print(f"{len(detections):,} -> {len(calibrated):,} detections; wrote {out}")
    return record


def main():
    calibrate(load_config())


if __name__ == "__main__":
    main()
