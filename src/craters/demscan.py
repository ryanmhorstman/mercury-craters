"""Find big craters straight from the DEM, including those the catalog and the CNN both miss.

    python -m craters.demscan           # scan, tune on the train area, write the results

The CNN learned from the thesis catalog, which counted only clearly rimmed
craters, so it learned to pass over big degraded craters with soft rims and
smooth floors. The DEM bowl check (craters.demcheck) is reliable from 40 km up,
so here it is used as a detector: circles of every size from 40 to 400 km are
tested on a grid over the whole planet.

1. Scan. Diameters step by `size_step`; centres are spaced `centre_step` x D
   apart. Each circle gets the bowl features of craters.demcheck; circles
   passing a loose pre-filter are kept as candidates.
2. Rule + merging. A rule (minimum rim_frac and depth_ratio, floor_drop > 0)
   picks candidates. Greedy merging then keeps the best bowl
   (rim_frac x depth_ratio) among circles describing the same crater, and drops
   circles centred inside a bigger kept one (inner rings).
3. Tuning, train area only. For each rule: recall = share of catalog craters
   >= 40 km the scan finds; "known" = share of scan craters that match a catalog
   crater or a CNN detection, or are a mis-sized copy of one (selftrain.near_known:
   the scan's smallest circle is 40 km, so a 28 km crater shows up as a 40 km circle round it). The rule with the best recall whose known share
   is >= min_known is chosen. Real craters that both missed count against
   "known", so it is a lower bound on precision.

Writes data/processed/selftrain/demscan_candidates.parquet (the scan),
outputs/selftrain/demscan.json (tuning) and outputs/selftrain/demscan_new.csv:
scan craters >= 40 km matching neither the catalog nor the CNN, anywhere on the planet.

    python -m craters.demscan --cnn     # also ask the CNN about each new scan crater (CPU, ~10 min)

4. CNN support. About half the new scan craters are real, so a second,
   independent opinion is asked for: the coarse level of the CNN is re-run over
   the planet at a very low score (cached in
   data/processed/selftrain/coarse_lowscore.parquet, diameters corrected as in
   craters.calibrate), and each circle gets the best score of a detection
   matching it. Decoys (train-area catalog circles >= 40 km moved onto empty
   ground, as in craters.demcheck) give the rate at which a circle on empty
   ground is supported by chance. A cutoff's precision is estimated as
   1 - decoy rate x (all new scan craters) / (supported ones): a lower bound,
   since it treats every new scan crater as false. The chosen cutoff supports
   the most scan craters with that estimate >= min_precision. The scan craters
   supported at that cutoff get cnn_ok = True in demscan_new.csv; the tuning is
   written to outputs/selftrain/demscan_cnn.json.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .calibrate import apply_diameter_correction
from .compare_global import area_of, load
from .config import load_config
from .demcheck import Dem, bowl_features, make_decoys
from .geo import MERCURY_RADIUS_M, km_to_chord, lon_360, unit_vectors
from .regions import assign_split
from .selftrain import near_known


def scan_centres(diameter_km, step_frac, max_abs_lat):
    """Centres spaced about step_frac x D apart over the planet (rows of latitude, longitude spread by 1/cos)."""
    step_deg = np.degrees(step_frac * diameter_km * 1000.0 / MERCURY_RADIUS_M)
    lats, lons = [], []
    for lat in np.arange(-max_abs_lat, max_abs_lat + 1e-9, step_deg):
        n = max(1, int(np.ceil(360.0 * np.cos(np.radians(lat)) / step_deg)))
        lats.append(np.full(n, lat))
        lons.append((np.arange(n) + 0.5) * 360.0 / n)
    return np.concatenate(lats), np.concatenate(lons)


def scan(cfg, dem):
    """Every circle on the scan grid that passes the loose pre-filter, with its bowl features."""
    p = cfg["selftrain"]["dem_scan"]
    sizes = p["min_km"] * p["size_step"] ** np.arange(int(np.log(p["max_km"] / p["min_km"]) / np.log(p["size_step"])) + 1)
    found = []
    for d in sizes:
        start = time.time()
        lat, lon = scan_centres(d, p["centre_step"], p["max_abs_lat"])
        f = bowl_features(dem, lat, lon, np.full(len(lat), d), p)
        keep = ((f["rim_frac"] >= p["prefilter"]["min_rim_frac"]) & (f["depth_ratio"] >= p["prefilter"]["min_depth_ratio"])
                & (f["floor_drop"] > 0)).to_numpy()
        found.append(f[keep].assign(lat=lat[keep], lon=lon[keep], diameter_km=d))
        print(f"  D {d:6.1f} km: {len(lat):>9,} circles, {keep.sum():>7,} candidates ({time.time() - start:.0f} s)", flush=True)
    return pd.concat(found, ignore_index=True)


def merge(candidates):
    """Greedy merging: the best bowl wins; circles describing the same crater (centre within half the
    larger diameter, sizes within 1.5x) or centred in the inner 0.6 of a bigger kept circle are dropped."""
    c = candidates.assign(quality=candidates["rim_frac"] * candidates["depth_ratio"])
    c = c.sort_values("quality", ascending=False).reset_index(drop=True)
    points, d = unit_vectors(c["lat"], c["lon"]), c["diameter_km"].to_numpy()
    tree = cKDTree(points)
    dropped = np.zeros(len(c), dtype=bool)
    keep = []
    for i in range(len(c)):
        if dropped[i]:
            continue
        keep.append(i)
        near = np.array(tree.query_ball_point(points[i], km_to_chord(0.75 * d[i])), dtype=int)
        dist = np.linalg.norm(points[near] - points[i], axis=1) * MERCURY_RADIUS_M / 1000.0
        ratio = d[near] / d[i]
        same = (dist <= 0.5 * np.maximum(d[near], d[i])) & (ratio >= 1 / 1.5) & (ratio <= 1.5)
        inner = (dist <= 0.3 * d[i]) & (ratio < 1 / 1.5)
        dropped[near[same | inner]] = True
    return c.iloc[keep].reset_index(drop=True)


def matched(a, b, ratio=(0.75, 1.33)):
    """True for each crater of `a` with a crater of `b` centred within half its diameter, at a similar size."""
    if not len(a) or not len(b):
        return np.zeros(len(a), dtype=bool)
    tree, b_d = cKDTree(unit_vectors(b["lat"], b["lon"])), b["diameter_km"].to_numpy()
    out = np.zeros(len(a), dtype=bool)
    for i, (p, d) in enumerate(zip(unit_vectors(a["lat"], a["lon"]), a["diameter_km"].to_numpy())):
        out[i] = any(ratio[0] <= b_d[j] / d <= ratio[1] for j in tree.query_ball_point(p, km_to_chord(0.5 * d)))
    return out


def best_match_score(a, b, ratio=(0.75, 1.33)):
    """For each crater of `a`: the highest score among `b` matching it as in matched(), or 0."""
    out = np.zeros(len(a))
    if not len(a) or not len(b):
        return out
    tree, b_d, b_s = cKDTree(unit_vectors(b["lat"], b["lon"])), b["diameter_km"].to_numpy(), b["score"].to_numpy()
    for i, (p, d) in enumerate(zip(unit_vectors(a["lat"], a["lon"]), a["diameter_km"].to_numpy())):
        scores = [b_s[j] for j in tree.query_ball_point(p, km_to_chord(0.5 * d)) if ratio[0] <= b_d[j] / d <= ratio[1]]
        out[i] = max(scores, default=0.0)
    return out


def is_known(found, known):
    """Matches a known crater, or is a mis-sized copy of one / drawn round or inside one."""
    return matched(found, known) | near_known(found, known)


def select(candidates, rule):
    c = candidates[(candidates["rim_frac"] >= rule["min_rim_frac"]) & (candidates["depth_ratio"] >= rule["min_depth_ratio"])]
    return merge(c)


def tune(cfg, candidates, catalog, cnn):
    """Train-area recall and "known" share for every rule on the grid; the chosen rule."""
    p = cfg["selftrain"]["dem_scan"]
    regions = cfg["regions"]
    in_train = lambda f: assign_split(f["lat"], f["lon"], regions) == "train"  # noqa: E731
    cat = catalog[catalog["diameter_km"].between(p["min_km"], p["max_km"]) & in_train(catalog)]
    known = pd.concat([catalog, cnn], ignore_index=True)[["lat", "lon", "diameter_km"]]
    rows = []
    for rim in p["rim_frac_grid"]:
        for depth in p["depth_ratio_grid"]:
            rule = {"min_rim_frac": rim, "min_depth_ratio": depth}
            found = select(candidates, rule)
            found = found[in_train(found)]
            rows.append({**rule, "scan_craters": len(found),
                         "recall": round(float(matched(cat, found).mean()), 4),
                         "known": round(float(is_known(found, known).mean()), 4) if len(found) else np.nan})
            print(f"  {rows[-1]}", flush=True)
    table = pd.DataFrame(rows)
    ok = table[table["known"] >= p["min_known"]]
    best = ok.loc[ok["recall"].idxmax()] if len(ok) else table.loc[table["known"].idxmax()]
    return {"min_rim_frac": float(best["min_rim_frac"]), "min_depth_ratio": float(best["min_depth_ratio"])}, table


def low_score_run(cfg):
    """The coarse level of the published model over the whole planet at a very low score, on the CPU."""
    from .detect import detect_on_tiles
    from .detectors.cnn import Detector, overlap_rule

    cache = cfg["paths"]["catalog"].parent / "selftrain" / "coarse_lowscore.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    params = cfg["detectors"]["cnn"]
    tiles = pd.read_parquet(cfg["paths"]["tiles_global"])
    tiles = tiles[tiles["level"] == "coarse"]
    detector = Detector(cfg["paths"]["models"] / f"{params['model_name']}_best.pt", params["level_downsample"],
                        device="cpu", min_score=cfg["selftrain"]["dem_scan"]["cnn"]["min_score"],
                        twins=params["twin_suppression"])
    print(f"re-running the coarse level on {len(tiles)} tiles (CPU)", flush=True)
    found = detect_on_tiles(detector, tiles, cfg, progress=True, merge=overlap_rule(cfg["match"], params["twin_suppression"]))
    correction = json.loads((cfg["paths"]["global"] / "calibration.json").read_text())["diameter_correction"]
    found = apply_diameter_correction(found, correction)[["lat", "lon", "diameter_km", "score", "level"]]
    found.to_parquet(cache, index=False)
    return found


def cnn_check(cfg, new, catalog):
    """cnn_score for each new scan crater, and the decoy-tuned cutoff (see the module docstring, step 4)."""
    p = cfg["selftrain"]["dem_scan"]
    low = low_score_run(cfg)
    cat = catalog[catalog["diameter_km"].between(p["min_km"], p["max_km"])
                  & (assign_split(catalog["lat"], catalog["lon"], cfg["regions"]) == "train")].reset_index(drop=True)
    decoys = make_decoys(cat, np.random.default_rng(cfg["seed"]))
    cat_score, decoy_score = best_match_score(cat, low), best_match_score(decoys, low)
    new_score = best_match_score(new, low)
    rows = [{"score_cutoff": c, "catalog_pass": round(float((cat_score >= c).mean()), 4),
             "decoy_pass": round(float((decoy_score >= c).mean()), 4), "new_pass": int((new_score >= c).sum())}
            for c in p["cnn"]["score_grid"]]
    for r in rows:
        r["precision_min"] = round(1 - r["decoy_pass"] * len(new) / r["new_pass"], 4) if r["new_pass"] else np.nan
    ok = [r for r in rows if r["precision_min"] >= p["cnn"]["min_precision"]]
    chosen = max(ok, key=lambda r: r["new_pass"]) if ok else {"score_cutoff": np.inf}
    record = {"catalog_craters": len(cat), "decoys": len(decoys), "grid": rows, "chosen": chosen}
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    return new_score, record


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cnn", action="store_true", help="also ask the CNN about each new scan crater")
    args = parser.parse_args()
    cfg = load_config()
    out_dir = cfg["paths"]["selftrain"]
    cache = cfg["paths"]["catalog"].parent / "selftrain" / "demscan_candidates.parquet"
    _, catalog, cnn = load(cfg, calibrated=True)

    if cache.exists():
        candidates = pd.read_parquet(cache)
        print(f"{len(candidates):,} scan candidates from {cache}", flush=True)
    else:
        print("scanning the DEM", flush=True)
        candidates = scan(cfg, Dem(cfg["paths"]["dem_raw"]))
        cache.parent.mkdir(parents=True, exist_ok=True)
        candidates.to_parquet(cache, index=False)

    print("tuning on the train area", flush=True)
    rule, table = tune(cfg, candidates, catalog, cnn)
    found = select(candidates, rule)
    found["lon"] = lon_360(found["lon"])
    found["in_catalog"] = is_known(found, catalog)
    found["in_cnn"] = is_known(found, cnn)
    found["area"] = area_of(found["lat"], found["lon"], cfg["regions"])
    new = found[~found["in_catalog"] & ~found["in_cnn"]].reset_index(drop=True)
    columns = ["lat", "lon", "diameter_km", "depth_ratio", "rim_frac", "floor_drop", "area"]
    if args.cnn:
        print("asking the CNN", flush=True)
        new["cnn_score"], cnn_record = cnn_check(cfg, new, catalog)
        new["cnn_ok"] = new["cnn_score"] >= cnn_record["chosen"]["score_cutoff"]
        (out_dir / "demscan_cnn.json").write_text(json.dumps(cnn_record, indent=2))
        columns += ["cnn_score", "cnn_ok"]
        print(f"CNN cutoff {cnn_record['chosen']}: {int(new['cnn_ok'].sum())} of {len(new)} new scan craters supported", flush=True)

    record = {"rule": rule, "grid": table.to_dict("records"), "scan_craters": len(found), "new": len(new),
              "by_area": new["area"].value_counts().to_dict()}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "demscan.json").write_text(json.dumps(record, indent=2))
    new[columns].sort_values("diameter_km", ascending=False).round(4).to_csv(out_dir / "demscan_new.csv", index=False)
    print(f"rule {rule}: {len(found):,} scan craters, {len(new):,} in neither the catalog nor the CNN run -> "
          f"{out_dir / 'demscan_new.csv'}")


if __name__ == "__main__":
    main()
