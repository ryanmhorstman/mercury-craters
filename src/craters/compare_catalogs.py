"""Compare the thesis catalog and the CNN run with two published Mercury catalogs.

    python -m craters.compare_catalogs

The published catalogs (data/raw, not in git; see docs/catalog-comparison.md):

  herrick     Herrick et al. 2018, JGR Planets, doi:10.1029/2017JE005516: hand-mapped, D > 5 km,
              obvious secondaries left out. Supporting tables S1 (5-10 km) and S2 (> 10 km).
  la_grassa   La Grassa et al. 2025, Remote Sensing 17:3287, catalog ME6M300TGT
              (doi:10.6084/m9.figshare.30187999): deep learning (YOLOLens), D > 0.4 km, 6.3M craters.
              Only craters >= compare.la_grassa_min_km are kept, in data/interim.

Both are in a slightly different frame from the USGS basemap the CNN ran on. For
each, the median north and east offset of its >= offset_min_km craters from the
matching CNN detections is measured per latitude band and removed, so that the
0.5 D matching rule isn't spent on a known shift.

Writes to outputs/compare/:
  offsets.csv     the shift removed from each published catalog
  counts.csv      craters per diameter bin, whole planet and T1-T3
  agreement.csv   for every ordered pair (reference, other): the share of the reference's
                  craters the other also lists, per area and bin
  supplement.csv  the supplemented catalog's new craters: share in each published catalog
  review.csv      the reviewed CNN detections missing from the thesis: share in each
                  published catalog, by verdict
  cnn_vs_union.csv  CNN scores against the thesis and Herrick together (T1-T3)
"""
import zipfile

import numpy as np
import pandas as pd

from .adjudicate import verdicts_for
from .catalog import load_eval_catalog
from .compare_global import WHOLE_PLANET
from .config import load_config, run_name
from .geo import MERCURY_RADIUS_M
from .match import candidate_pairs, match_one_to_one
from .metrics import bin_labels, evaluate
from .supplement import THESIS, drop_known

KM_PER_DEG = np.radians(1.0) * MERCURY_RADIUS_M / 1000.0
LA_GRASSA_COLUMNS = {"Latitude": "lat", "Longitude": "lon", "Diameter_mean": "diameter_km", "Confidence": "score"}


# ---------------------------------------------------------------- loading

def load_herrick(cfg):
    """Herrick 2018 tables S1 and S2 as one catalog: lat, lon (0-360 E), diameter_km."""
    folder = cfg["paths"]["herrick_2018"]
    tables = [pd.read_excel(path) for path in sorted(folder.glob("*.xlsx")) if "pairs" not in path.name.lower()]
    if len(tables) != 2:
        raise FileNotFoundError(f"expected tables S1 and S2 (two .xlsx files) in {folder}")
    craters = pd.concat(tables, ignore_index=True)
    return pd.DataFrame({"lat": craters["lat"], "lon": craters["lon"] % 360, "diameter_km": craters["diam"]})


def load_la_grassa(cfg):
    """La Grassa 2025 craters >= la_grassa_min_km: lat, lon (0-360 E), diameter_km (mean of width and
    height), score (the model's confidence). The subset is cut from the 1.8 GB CSV once and cached."""
    subset = cfg["paths"]["la_grassa_2025"]
    if not subset.exists():
        min_km = cfg["compare"]["la_grassa_min_km"]
        archive = zipfile.ZipFile(cfg["paths"]["la_grassa_2025_zip"])
        parts = []
        with archive.open(archive.infolist()[0]) as f:
            for chunk in pd.read_csv(f, usecols=list(LA_GRASSA_COLUMNS), chunksize=1_000_000):
                parts.append(chunk[chunk["Diameter_mean"] >= min_km])
        subset.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(parts, ignore_index=True).to_parquet(subset, index=False)
    craters = pd.read_parquet(subset).rename(columns=LA_GRASSA_COLUMNS)
    craters["lon"] = craters["lon"] % 360
    return craters[["lat", "lon", "diameter_km", "score"]]


# ---------------------------------------------------------------- frame

def offsets_from(reference, other, match, min_km):
    """Median north and east offset (km) of `other` from `reference`, over matched pairs >= min_km."""
    reference = reference[reference["diameter_km"] >= min_km].reset_index(drop=True)
    other = other[other["diameter_km"] >= min_km * match["ratio_min"]].reset_index(drop=True)
    pairs = match_one_to_one(reference, other, match)
    if pairs.empty:                                     # nothing to measure: leave this band where it is
        return {"pairs": 0, "north_km": 0.0, "east_km": 0.0}
    lat = reference["lat"].to_numpy()[pairs["cat"]]
    north = (other["lat"].to_numpy()[pairs["det"]] - lat) * KM_PER_DEG
    d_lon = (other["lon"].to_numpy()[pairs["det"]] - reference["lon"].to_numpy()[pairs["cat"]] + 180) % 360 - 180
    east = d_lon * KM_PER_DEG * np.cos(np.radians(lat))
    return {"pairs": len(pairs), "north_km": float(np.median(north)), "east_km": float(np.median(east))}


def shift_onto(reference, other, cfg):
    """`other` moved by its median offset from `reference`, per latitude band. Returns (moved copy, offsets)."""
    p = cfg["compare"]
    moved, rows = other.copy(), []
    for lo, hi in p["offset_lat_bands"]:
        band = lambda frame: frame[(frame["lat"] >= lo) & (frame["lat"] < hi + (hi == 90))]  # noqa: E731
        offset = offsets_from(band(reference), band(other), cfg["match"], p["offset_min_km"])
        rows.append({"lat_band": f"{lo}..{hi}", **offset})
        rows_in = (moved["lat"] >= lo) & (moved["lat"] < hi + (hi == 90))
        moved.loc[rows_in, "lat"] -= offset["north_km"] / KM_PER_DEG
        cos_lat = np.maximum(np.cos(np.radians(moved.loc[rows_in, "lat"])), 0.05)
        moved.loc[rows_in, "lon"] = (moved.loc[rows_in, "lon"] - offset["east_km"] / (KM_PER_DEG * cos_lat)) % 360
    return moved, pd.DataFrame(rows)


# ---------------------------------------------------------------- comparisons

def areas(cfg):
    return {"whole planet": [WHOLE_PLANET], "T1-T3": list(cfg["regions"]["test"].values())}


def counts(catalogs, cfg):
    """Craters >= the first bin edge per catalog, area and diameter bin."""
    bins = cfg["evaluation"]["diameter_bins_km"]
    rows = []
    for area, boxes in areas(cfg).items():
        for name, catalog in catalogs.items():
            table = evaluate(catalog, catalog.iloc[:0], boxes, cfg["match"], bins)[0]
            rows.append({"area": area, "catalog": name, **table.set_index("bin")["n_catalog"].to_dict()})
    return pd.DataFrame(rows)


def agreement(catalogs, cfg):
    """For each ordered pair: the share of the reference catalog's craters the other also lists
    (the other's recall of it), per area and diameter bin, by the project's matching rule."""
    bins = cfg["evaluation"]["diameter_bins_km"]
    rows = []
    for area, boxes in areas(cfg).items():
        for ref_name, reference in catalogs.items():
            for other_name, other in catalogs.items():
                if other_name == ref_name:
                    continue
                table = evaluate(reference, other, boxes, cfg["match"], bins)[0]
                for _, row in table.iterrows():
                    rows.append({"area": area, "reference": ref_name, "other": other_name, "bin": row["bin"],
                                 "n_reference": int(row["n_catalog"]), "found": int(row["TP"]), "share": row["recall"]})
    return pd.DataFrame(rows)


def listed_in(circles, catalog, match):
    """True for each circle that is the same crater as some row of `catalog`."""
    found = np.zeros(len(circles), dtype=bool)
    found[candidate_pairs(catalog, circles.reset_index(drop=True), match)["det"].unique()] = True
    return found


def share_listed(groups, published, match):
    """For each named group of circles: how many, and the share listed in each published catalog."""
    rows = []
    for name, circles in groups.items():
        row = {"group": name, "n": len(circles)}
        for catalog_name, catalog in published.items():
            row[f"in_{catalog_name}"] = listed_in(circles, catalog, match).mean() if len(circles) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def supplement_groups(cfg):
    supplemented = pd.read_parquet(cfg["paths"]["catalog_supplemented"])
    return {source: supplemented[supplemented["source"] == source].reset_index(drop=True)
            for source in supplemented["source"].unique() if source != THESIS}


def review_groups(cfg):
    """The reviewed CNN detections missing from the thesis (phase4 --calibrated), by verdict and size."""
    items = pd.read_csv(cfg["paths"]["review_items_calibrated"])
    verdicts, _ = verdicts_for(items, cfg["paths"]["verdicts"])
    items["verdict"] = items["item_id"].map(verdicts)
    items = items[items["verdict"].isin(["real", "not_real"])]
    groups = {}
    for verdict in ("real", "not_real"):
        for label, (lo, hi) in (("5-10 km", (5, 10)), (">= 10 km", (10, np.inf))):
            rows = items[(items["verdict"] == verdict) & items["diameter_km"].between(lo, hi, inclusive="left")]
            groups[f"judged {verdict}, {label}"] = rows.reset_index(drop=True)
    return groups


def union_catalog(first, second, match):
    """`first` plus the craters of `second` it doesn't already list."""
    extra = drop_known(first[["lat", "lon", "diameter_km"]], second[["lat", "lon", "diameter_km"]], match)
    return pd.concat([first[["lat", "lon", "diameter_km"]], extra], ignore_index=True)


def main():
    cfg = load_config()
    out = cfg["paths"]["compare"]
    out.mkdir(parents=True, exist_ok=True)
    match, bins = cfg["match"], cfg["evaluation"]["diameter_bins_km"]

    thesis = load_eval_catalog(cfg)
    cnn = pd.read_parquet(cfg["paths"]["detections"] / f"global_{run_name(cfg['detectors']['cnn'])}_calibrated.parquet")
    published, offsets = {}, []
    for name, loader in (("herrick", load_herrick), ("la_grassa", load_la_grassa)):
        published[name], shift = shift_onto(cnn, loader(cfg), cfg)
        offsets.append(shift.assign(catalog=name))
    offsets = pd.concat(offsets, ignore_index=True)
    offsets.round(3).to_csv(out / "offsets.csv", index=False)
    print("shift removed (median offset from the CNN run, craters >= "
          f"{cfg['compare']['offset_min_km']} km):\n" + offsets.round(2).to_string(index=False))

    catalogs = {"thesis": thesis, "cnn": cnn, **published}
    table = counts(catalogs, cfg)
    table.to_csv(out / "counts.csv", index=False)
    print("\ncraters per diameter bin:\n" + table.to_string(index=False))

    shares = agreement(catalogs, cfg)
    shares.round(4).to_csv(out / "agreement.csv", index=False)
    wide = shares.pivot_table(index=["area", "reference", "bin"], columns="other", values="share", sort=False)
    print("\nshare of the reference's craters the other lists:\n" + wide.round(2).to_string())

    groups = {**supplement_groups(cfg), **review_groups(cfg)}
    table = share_listed(groups, published, match)
    for path, names in (("supplement.csv", supplement_groups(cfg)), ("review.csv", review_groups(cfg))):
        table[table["group"].isin(names)].round(4).to_csv(out / path, index=False)
    print("\nlisted in the published catalogs:\n" + table.round(3).to_string(index=False))

    union = union_catalog(thesis, published["herrick"], match)
    test = list(cfg["regions"]["test"].values())
    rows = []
    for name, reference in (("thesis", thesis), ("thesis + herrick", union)):
        rows.append(evaluate(reference, cnn, test, match, bins)[0].assign(reference=name))
    scores = pd.concat(rows, ignore_index=True)
    scores.round(4).to_csv(out / "cnn_vs_union.csv", index=False)
    print("\nCNN in T1-T3 against the thesis, and against the thesis and Herrick together:\n"
          + scores[["reference", "bin", "n_catalog", "n_detections", "precision", "recall", "f1"]].round(3).to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
