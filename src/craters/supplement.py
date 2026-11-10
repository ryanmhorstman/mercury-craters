"""The thesis catalog plus the craters this project found and confirmed with the DEM.

    python -m craters.supplement                 # build the supplemented catalog
    python -m craters.supplement --score         # score the calibrated global run against both catalogs
    python -m craters.supplement --score NAME    # ... or any run: data/detections/NAME.parquet

The supplemented catalog has three kinds of rows, told apart by `source`:

  thesis         the registered thesis catalog (catalog_registered_labels), unchanged
  dem_confirmed  CNN detections >= 20 km matching no catalog crater that pass the DEM
                 bowl check (outputs/selftrain/new_craters.csv, from craters.selftrain)
  dem_scan_cnn   DEM-scan craters >= 40 km that the CNN also sees at a low score
                 (cnn_ok rows of outputs/selftrain/demscan_new.csv, from craters.demscan)

A new crater is dropped when it is the same crater (the `match` rule) as a thesis
crater or as a new crater already added, so nothing is listed twice. The thesis
files are only read. The thesis sheet name, called `source` in the catalog, is
kept as `sheet`.

Writes data/processed/catalog_supplemented.parquet and
outputs/selftrain/catalog_supplemented.csv. Scoring writes
outputs/selftrain/scores_<run>.csv: precision, recall and F1 in V1 and T1-T3,
per diameter bin, against the thesis catalog and against the supplemented one.

The dem_confirmed craters are detections of cnn_reg_f2 itself, so that model
finds every one of them by construction; keep that in mind when comparing models
on the supplemented catalog.
"""
import argparse

import numpy as np
import pandas as pd

from .config import load_config, run_name
from .dedupe import dedupe_circles
from .match import candidate_pairs
from .metrics import evaluate
from .regions import assign_split

THESIS, DEM_CONFIRMED, DEM_SCAN_CNN = "thesis", "dem_confirmed", "dem_scan_cnn"
ID_PREFIX = {DEM_CONFIRMED: "DC", DEM_SCAN_CNN: "DS"}
COLUMNS = ["id", "lat", "lon", "diameter_km", "source", "split", "sheet", "morph_type", "name",
           "cnn_score", "depth_ratio", "rim_frac"]


def drop_known(known, new, match):
    """Rows of `new` that are not the same crater as any row of `known`, nor as a bigger row of `new`."""
    new = new.sort_values("diameter_km", ascending=False).reset_index(drop=True)
    repeated = dedupe_circles(new["lat"], new["lon"], new["diameter_km"], np.arange(len(new)),
                              match["center_frac"], match["ratio_min"], match["ratio_max"]) >= 0
    in_known = np.zeros(len(new), dtype=bool)
    in_known[candidate_pairs(known, new, match)["det"]] = True
    return new[~repeated & ~in_known].reset_index(drop=True)


def build_supplemented(catalog, dem_confirmed, dem_scan_cnn, match, regions):
    """Thesis rows first, then the new craters of each source that aren't already listed."""
    parts = [catalog.rename(columns={"source": "sheet"}).assign(source=THESIS)]
    for source, new in ((DEM_CONFIRMED, dem_confirmed), (DEM_SCAN_CNN, dem_scan_cnn)):
        listed = pd.concat(parts, ignore_index=True)[["lat", "lon", "diameter_km"]]
        new = drop_known(listed, new, match)
        new["id"] = [f"{ID_PREFIX[source]}{i:05d}" for i in range(1, len(new) + 1)]
        new["split"] = assign_split(new["lat"], new["lon"], regions)
        parts.append(new.assign(source=source))
    return pd.concat(parts, ignore_index=True).reindex(columns=COLUMNS)


def read_new_craters(cfg):
    """The two sets of DEM-backed craters, with the CNN's score for each in `cnn_score`."""
    folder = cfg["paths"]["selftrain"]
    confirmed = pd.read_csv(folder / "new_craters.csv").rename(columns={"score": "cnn_score"})
    scan = pd.read_csv(folder / "demscan_new.csv")
    return confirmed, scan[scan["cnn_ok"]]


def build(cfg):
    catalog = pd.read_parquet(cfg["paths"]["catalog_registered_labels"])
    confirmed, scan = read_new_craters(cfg)
    supplemented = build_supplemented(catalog, confirmed, scan, cfg["match"], cfg["regions"])

    out = cfg["paths"]["catalog_supplemented"]
    supplemented.to_parquet(out, index=False)
    csv = cfg["paths"]["selftrain"] / "catalog_supplemented.csv"
    supplemented.round(5).to_csv(csv, index=False)

    counts = supplemented["source"].value_counts()
    for source, offered in ((DEM_CONFIRMED, len(confirmed)), (DEM_SCAN_CNN, len(scan))):
        print(f"{source}: {counts.get(source, 0)} added of {offered} ({offered - counts.get(source, 0)} already listed)")
    print(f"{len(supplemented):,} craters ({counts[THESIS]:,} thesis) -> {out} and {csv}")
    new = supplemented[supplemented["source"] != THESIS]
    print("\nnew craters by split:\n" + pd.crosstab(new["split"], new["source"], margins=True).to_string())
    return supplemented


def scoring_catalogs(cfg):
    """The two catalogs a run is scored against, by name."""
    supplemented = cfg["paths"]["catalog_supplemented"]
    if not supplemented.exists():
        raise FileNotFoundError(f"{supplemented} is missing: run `python -m craters.supplement` first")
    return {THESIS: pd.read_parquet(cfg["paths"]["catalog_registered_labels"]),
            "supplemented": pd.read_parquet(supplemented)}


def score_regions(cfg):
    """Region name -> boxes: V1, each test region, and the test regions together."""
    regions = cfg["regions"]
    out = {name: [box] for name, box in {**regions["validation"], **regions["test"]}.items()}
    out["-".join([min(regions["test"]), max(regions["test"])])] = list(regions["test"].values())
    return out


def score_run(cfg, detections, catalogs):
    """One row per catalog, region and diameter bin (plus "all"), with the columns of metrics.score."""
    bins = cfg["evaluation"]["diameter_bins_km"]
    tables = []
    for catalog_name, catalog in catalogs.items():
        for region, boxes in score_regions(cfg).items():
            table = evaluate(catalog, detections, boxes, cfg["match"], bins)[0]
            table.insert(0, "region", region)
            table.insert(0, "catalog", catalog_name)
            tables.append(table)
    return pd.concat(tables, ignore_index=True)


def f1_summary(scores):
    """F1 per region and bin, one column per catalog."""
    wide = scores.pivot(index=["region", "bin"], columns="catalog", values="f1")
    order = pd.MultiIndex.from_frame(scores[["region", "bin"]].drop_duplicates())
    return wide.loc[order, list(scores["catalog"].unique())]


def score(cfg, run=None):
    run = run or f"global_{run_name(cfg['detectors']['cnn'])}_calibrated"
    detections = pd.read_parquet(cfg["paths"]["detections"] / f"{run}.parquet")
    scores = score_run(cfg, detections, scoring_catalogs(cfg))
    out = cfg["paths"]["selftrain"] / f"scores_{run}.csv"
    scores.round(4).to_csv(out, index=False)
    columns = ["catalog", "region", "bin", "n_catalog", "n_detections", "TP", "FP", "FN", "precision", "recall", "f1"]
    print(f"{run}:\n" + scores[columns].round(3).to_string(index=False))
    print("\nF1:\n" + f1_summary(scores).round(3).to_string())
    print(f"\nwrote {out}")
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--score", nargs="?", const="", metavar="RUN",
                        help="score data/detections/RUN.parquet (default: the calibrated global run) against both catalogs")
    args = parser.parse_args()
    cfg = load_config()
    if args.score is None:
        build(cfg)
    else:
        score(cfg, args.score or None)


if __name__ == "__main__":
    main()
