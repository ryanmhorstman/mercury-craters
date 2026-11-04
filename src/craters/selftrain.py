"""Self-training: teach the CNN DEM-confirmed craters the catalog doesn't list.

    python -m craters.demcheck      # once: fit the DEM bowl check (outputs/selftrain/dem_check.json)
    python -m craters.selftrain     # rounds until V1 stops improving or selftrain.max_rounds

Each round starts from the best model so far and its calibrated global run:

1. Pseudo-labels, from detections in the train area only that match no catalog crater:
     new crater  >= 20 km, passes the DEM bowl check for its size band, and isn't a
                 mis-sized copy of a catalog crater or a circle inside a bigger one
                 (a peak ring passes the bowl check too)
     ignore      any other with score >= ignore_min_score: likely real, so the
                 model isn't taught that it's "no crater", but not confirmed
2. Fine-tune the model from its own weights on catalog + pseudo-labels.
3. Global run, calibration (craters.calibrate), then the gate: the round is kept
   only if V1 F1 against the thesis catalog is at least the best so far minus
   gate_tolerance. A rejected round ends the loop, since the next round would be
   built from the same run and labels. T1-T3 F1 is reported, never used.

The thesis catalog stays the source of truth: it is never edited, every gate
scores against it, and pseudo-labels never fall in V1, T1-T3 or their buffers.
Writes outputs/selftrain/rounds.csv, state.json, new_craters.csv (DEM-confirmed
craters missing from the catalog, anywhere on the planet, from the best run) and
data/processed/selftrain/pseudo_<model>.parquet.
"""
import copy
import json

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .calibrate import calibrate
from .compare_global import WHOLE_PLANET, area_of
from .config import load_config, run_name
from .demcheck import Dem, bowl_features, load_rules, passes_banded
from .geo import km_to_chord, unit_vectors
from .metrics import label_outcomes, score
from .regions import assign_split


def calibrated_run(cfg, model):
    return pd.read_parquet(cfg["paths"]["detections"] / f"global_{model}_calibrated.parquet")


def near_known(detections, catalog):
    """True where a detection is a mis-sized copy of a catalog crater (centre within half its
    diameter, 0.5-2x its size), is centred in the inner 0.6 of a catalog crater over 1.5x bigger,
    or has a catalog crater of 0.25-0.5x its size within 0.3 diameters of its centre."""
    cat_points, cat_d = unit_vectors(catalog["lat"], catalog["lon"]), catalog["diameter_km"].to_numpy()
    det_points, det_d = unit_vectors(detections["lat"], detections["lon"]), detections["diameter_km"].to_numpy()
    out = np.zeros(len(detections), dtype=bool)
    if not len(detections):
        return out
    cat_tree, det_tree = cKDTree(cat_points), cKDTree(det_points)
    for i, (p, d) in enumerate(zip(det_points, det_d)):
        near = cat_tree.query_ball_point(p, km_to_chord(0.5 * d))
        out[i] = any(0.5 <= cat_d[j] / d <= 2.0 for j in near)
        # A circle drawn round a smaller catalog crater near its centre borrows that crater's bowl.
        near = cat_tree.query_ball_point(p, km_to_chord(0.3 * d))
        out[i] |= any(0.25 <= cat_d[j] / d < 0.5 for j in near)
    for p, big in zip(cat_points, cat_d):
        if big > 1.5 * det_d.min():
            near = np.array(det_tree.query_ball_point(p, km_to_chord(0.3 * big)), dtype=int)
            out[near[det_d[near] < big / 1.5]] = True
    return out


def unmatched_confirmed(cfg, detections, catalog, dem, rules):
    """Detections matching no catalog crater, with a DEM-confirmed flag (and the DEM features)."""
    _, det = label_outcomes(catalog, detections, [WHOLE_PLANET], cfg["match"], 0)
    det = det[det["outcome"] == "FP"].reset_index(drop=True)
    in_bands = det["diameter_km"].between(rules[0]["min_km"], rules[-1]["max_km"], inclusive="left").to_numpy()
    det["near_known"] = False
    det.loc[in_bands, "near_known"] = near_known(det[in_bands], catalog)
    check = in_bands & ~det["near_known"].to_numpy()
    features = bowl_features(dem, det.loc[check, "lat"], det.loc[check, "lon"], det.loc[check, "diameter_km"],
                             cfg["selftrain"]["dem_check"])
    for column in features:
        det.loc[check, column] = features[column].to_numpy()
    det["dem_confirmed"] = False
    det.loc[check, "dem_confirmed"] = passes_banded(features.reset_index(drop=True), det.loc[check, "diameter_km"], rules)
    return det


def pseudo_labels(cfg, detections, catalog, dem, rules):
    """New craters (ignore False) and ignore circles from one calibrated run, train area only."""
    p = cfg["selftrain"]
    det = unmatched_confirmed(cfg, detections, catalog, dem, rules)
    det = det[assign_split(det["lat"], det["lon"], cfg["regions"]) == "train"]
    new = det[det["dem_confirmed"]].assign(ignore=False)
    unsure = det[~det["dem_confirmed"] & (det["score"] >= p["ignore_min_score"])].assign(ignore=True)
    return pd.concat([new, unsure], ignore_index=True)[["lat", "lon", "diameter_km", "score", "level", "ignore"]]


def f1_scores(cfg, detections, catalog):
    """F1 against the catalog in V1 (the gate) and T1-T3 (reported only), craters >= the first bin edge."""
    min_km = cfg["evaluation"]["diameter_bins_km"][0]
    out = {}
    for name, boxes in (("v1_f1", list(cfg["regions"]["validation"].values())),
                        ("test_f1", list(cfg["regions"]["test"].values()))):
        cat, det = label_outcomes(catalog, detections, boxes, cfg["match"], min_km)
        out[name] = round(float(score(cat[cat["counted"]], det[det["counted"]])["f1"]), 4)
    return out


def train_round(cfg, base, name, labels_path):
    """Fine-tune `base` on catalog + pseudo-labels as model `name`, run it globally and calibrate."""
    from .phase3 import run_global
    from .train_cnn import train

    p = cfg["selftrain"]
    cfg = copy.deepcopy(cfg)
    cnn = cfg["detectors"]["cnn"]
    cnn.update(model_name=name, extra_labels=str(labels_path), level_models={},
               init_from=str(cfg["paths"]["models"] / f"{base}_best.pt"), epochs=p["epochs"], lr=p["lr"])
    train(cfg)
    run_global(cfg, timing_path=cfg["paths"]["selftrain"] / f"run_timing_{name}.json")
    calibrate(cfg, record_path=cfg["paths"]["selftrain"] / f"calibration_{name}.json", quiet=True)


def main():
    cfg = load_config()
    p = cfg["selftrain"]
    out_dir = cfg["paths"]["selftrain"]
    labels_dir = cfg["paths"]["catalog"].parent / "selftrain"
    labels_dir.mkdir(parents=True, exist_ok=True)
    catalog = pd.read_parquet(cfg["paths"]["catalog_registered_labels"])
    dem, rules = Dem(cfg["paths"]["dem_raw"]), load_rules(cfg)

    start = run_name(cfg["detectors"]["cnn"])
    best = {"model": start, **f1_scores(cfg, calibrated_run(cfg, start), catalog)}
    rounds = [{"round": 0, **best, "new_craters": 0, "ignored": 0, "kept": True}]
    print(f"round 0 ({start}): V1 F1 {best['v1_f1']}, T1-T3 F1 {best['test_f1']}", flush=True)

    for r in range(1, p["max_rounds"] + 1):
        labels = pseudo_labels(cfg, calibrated_run(cfg, best["model"]), catalog, dem, rules)
        name = f"{cfg['detectors']['cnn']['model_name']}_st{r}"
        labels_path = labels_dir / f"pseudo_{name}.parquet"
        labels.to_parquet(labels_path, index=False)
        print(f"round {r} ({name}): {int((~labels['ignore']).sum())} new craters, {int(labels['ignore'].sum())} ignored", flush=True)

        train_round(cfg, best["model"], name, labels_path)
        result = {"model": name, **f1_scores(cfg, calibrated_run(cfg, name), catalog)}
        kept = result["v1_f1"] >= best["v1_f1"] - p["gate_tolerance"]
        rounds.append({"round": r, **result, "new_craters": int((~labels["ignore"]).sum()),
                       "ignored": int(labels["ignore"].sum()), "kept": kept})
        pd.DataFrame(rounds).to_csv(out_dir / "rounds.csv", index=False)
        print(f"round {r}: V1 F1 {result['v1_f1']} (best {best['v1_f1']}), T1-T3 F1 {result['test_f1']} -> "
              f"{'kept' if kept else 'rejected, stopping'}", flush=True)
        if not kept:
            break
        best = result

    pd.DataFrame(rounds).to_csv(out_dir / "rounds.csv", index=False)
    (out_dir / "state.json").write_text(json.dumps({"best_model": best["model"], **best}, indent=2))

    found = unmatched_confirmed(cfg, calibrated_run(cfg, best["model"]), catalog, dem, rules)
    found = found[found["dem_confirmed"]].copy()
    found["area"] = area_of(found["lat"], found["lon"], cfg["regions"])
    columns = ["lat", "lon", "diameter_km", "score", "level", "depth_ratio", "rim_frac", "floor_drop", "area"]
    found[columns].sort_values("diameter_km", ascending=False).round(4).to_csv(out_dir / "new_craters.csv", index=False)
    print(f"best model {best['model']}; {len(found)} DEM-confirmed craters missing from the catalog -> {out_dir / 'new_craters.csv'}")
    if best["model"] != start:
        print(f"to publish it: set detectors.cnn.model_name to {best['model']} and rerun calibrate / compare_global")


if __name__ == "__main__":
    main()
