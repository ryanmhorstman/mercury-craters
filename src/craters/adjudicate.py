"""Phase 4: turn detector/catalog disagreements into a review queue, and verdicts into adjusted metrics.

Review items (test regions only, craters >= the minimum diameter):
  catalog    a catalog crater that at least one detector missed
             question: is this catalogued crater real?
  detection  an unmatched detection (merged across detectors)
             question: is this a real crater the catalog lacks?
The reviewer sees both kinds the same way (a circle on the image), in a fixed
random order, without being told which kind it is.

Verdicts: real / not_real / unsure. Each item gets p_real:
  1 or 0 when reviewed; otherwise (unreviewed or unsure) the real-rate of the
  reviewed items in its stratum (kind x diameter bin). Because the queue is in
  random order, any prefix of it is a random sample, so partial review gives
  unbiased estimates; full review gives exact numbers.
"""
import csv
import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .dedupe import dedupe_circles
from .match import candidate_pairs
from .metrics import bin_labels, evaluate, wilson_interval

VERDICTS = ("real", "not_real", "unsure")
VERDICT_COLUMNS = ["item_id", "verdict", "lat", "lon", "diameter_km", "kind", "reviewed_at"]


def item_bin(diameter_km, bins):
    return pd.cut(pd.Series(diameter_km), bins, right=False, labels=bin_labels(bins)).astype(str).to_numpy()


def build_items(catalog, detectors, boxes, match, bins, seed):
    """Review queue for the given detectors. Returns (items, per-detector original tables)."""
    tables, missed_by, unmatched = {}, {}, []
    for name, detections in detectors.items():
        table, cat, det = evaluate(catalog, detections, boxes, match, bins)
        tables[name] = table
        for crater_id in cat.loc[cat["outcome"] == "FN", "id"]:
            missed_by.setdefault(crater_id, []).append(name)
        fp = det[det["outcome"] == "FP"][["lat", "lon", "diameter_km", "score"]].assign(detector=name)
        unmatched.append(fp)

    cat_items = catalog[catalog["id"].isin(missed_by)].copy()
    cat_items = pd.DataFrame({
        "item_id": "cat:" + cat_items["id"], "kind": "catalog",
        "lat": cat_items["lat"], "lon": cat_items["lon"], "diameter_km": cat_items["diameter_km"],
        "detectors": cat_items["id"].map(lambda i: ";".join(sorted(missed_by[i]))),   # detectors that MISSED it
        "auto_verdict": "",
    })

    det_items = merge_unmatched(pd.concat(unmatched, ignore_index=True), list(detectors), match)
    # A detection left over by one-to-one matching can sit on a crater that's already
    # catalogued (a second circle on it). It is not a new crater; mark it without asking.
    on_catalogued = set(candidate_pairs(catalog, det_items, match)["det"].tolist())
    det_items["auto_verdict"] = ["duplicate" if i in on_catalogued else "" for i in range(len(det_items))]

    items = pd.concat([cat_items, det_items], ignore_index=True)
    items["bin"] = item_bin(items["diameter_km"], bins)
    order = np.random.default_rng(seed).permutation(len(items))
    items = items.iloc[order].reset_index(drop=True)
    items.insert(0, "position", np.arange(1, len(items) + 1))
    return items, tables


def merge_unmatched(unmatched, detector_order, match):
    """One item per distinct unmatched crater; `detectors` lists every detector that proposed it."""
    if unmatched.empty:
        return pd.DataFrame(columns=["item_id", "kind", "lat", "lon", "diameter_km", "detectors"])
    rank = unmatched["detector"].map({name: i for i, name in enumerate(detector_order)})
    order = np.lexsort((-unmatched["score"].to_numpy(), rank.to_numpy()))
    duplicate_of = dedupe_circles(unmatched["lat"], unmatched["lon"], unmatched["diameter_km"], order,
                                  match["center_frac"], match["ratio_min"], match["ratio_max"])
    group = np.where(duplicate_of < 0, np.arange(len(unmatched)), duplicate_of)
    proposers = unmatched.groupby(group)["detector"].agg(lambda s: ";".join(sorted(set(s))))
    kept = unmatched[duplicate_of < 0]
    return pd.DataFrame({
        "item_id": [stable_id(r.lat, r.lon, r.diameter_km) for r in kept.itertuples()],
        "kind": "detection", "lat": kept["lat"].to_numpy(), "lon": kept["lon"].to_numpy(),
        "diameter_km": kept["diameter_km"].to_numpy(), "detectors": proposers.loc[kept.index].to_numpy(),
    })


def stable_id(lat, lon, diameter_km):
    key = f"{lat:.4f},{lon:.4f},{diameter_km:.2f}"
    return "det:" + hashlib.sha1(key.encode()).hexdigest()[:10]


# ---------------------------------------------------------------- verdicts

def append_verdict(path, item, verdict):
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {VERDICTS}")
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=VERDICT_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow({"item_id": item["item_id"], "verdict": verdict, "lat": round(float(item["lat"]), 5),
                         "lon": round(float(item["lon"]), 5), "diameter_km": round(float(item["diameter_km"]), 3),
                         "kind": item["kind"], "reviewed_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})


def latest_verdicts(path):
    """item_id -> latest verdict (changing your mind just appends a new row)."""
    if not path.exists():
        return {}
    verdicts = pd.read_csv(path)
    return verdicts.groupby("item_id")["verdict"].last().to_dict()


def verdicts_for(items, path, tolerance_km=0.25):
    """Latest verdict per item, but only if it was given on the circle as it is now.

    A verdict records where its circle was. If an item has since moved (the
    catalog was re-registered) or changed size, its old verdict no longer
    answers the question and the item goes back in the queue.
    Returns (verdicts dict, number of stale verdicts dropped).
    """
    if not path.exists():
        return {}, 0
    rows = pd.read_csv(path).groupby("item_id").last()
    current = items.set_index("item_id")[["lat", "lon", "diameter_km"]]
    rows = rows.join(current, rsuffix="_now", how="inner")
    km_per_deg = np.radians(1.0) * 2439.4
    d_north = (rows["lat"] - rows["lat_now"]) * km_per_deg
    d_east = ((rows["lon"] - rows["lon_now"] + 180) % 360 - 180) * km_per_deg * np.cos(np.radians(rows["lat_now"]))
    same_place = np.hypot(d_north, d_east) <= tolerance_km
    same_size = (rows["diameter_km"] - rows["diameter_km_now"]).abs() <= 0.05 * rows["diameter_km_now"]
    valid = rows[same_place & same_size]
    return valid["verdict"].to_dict(), int(len(rows) - len(valid))


def p_real(items, verdicts):
    """Probability each item is a real crater: reviewed -> 1/0, else its stratum's reviewed rate."""
    auto = items["auto_verdict"].fillna("") != ""
    verdict = items["item_id"].map(verdicts).fillna("")
    verdict = verdict.where(~auto, "not_real")                             # duplicates are not new craters
    known = verdict.isin(["real", "not_real"])
    p = pd.Series(np.nan, index=items.index)
    p[known] = (verdict[known] == "real").astype(float)
    # Only human verdicts set the rates used to fill in unreviewed items.
    human = known & ~auto
    strata = items["kind"] + "|" + items["bin"]
    rates = p[human].groupby(strata[human]).mean()
    overall = p[human].groupby(items["kind"][human]).mean()
    fallback = strata.map(rates).fillna(items["kind"].map(overall))
    return p.fillna(fallback), verdict


# ---------------------------------------------------------------- adjusted metrics

def adjusted_metrics(items, verdicts, tables, bins):
    """Original vs adjudication-adjusted metrics per detector and bin.

    For detector X, with p = p_real per item:
      TP' = TP(original) + sum p over detection items X proposed
      FP' = sum (1 - p) over detection items X proposed
      FN' = sum p over catalog items X missed
            + sum p over detection items X did NOT propose (new craters others found)
    Catalogue craters every detector matched aren't reviewed and are assumed real.
    """
    p = p_real(items, verdicts)[0].to_numpy()      # NaN while a stratum has no verdicts yet
    is_det = (items["kind"] == "detection").to_numpy()
    is_cat = (items["kind"] == "catalog").to_numpy()
    rows = []
    for name, table in tables.items():
        # `detectors` lists who proposed a detection item, and who missed a catalog item.
        listed = items["detectors"].str.split(";").apply(lambda ds: name in ds).to_numpy()
        for _, orig in table.iterrows():
            in_bin = np.ones(len(items), bool) if orig["bin"] == "all" else (items["bin"] == orig["bin"]).to_numpy()
            tp = orig["TP"] + p[is_det & listed & in_bin].sum()
            fp = (1 - p[is_det & listed & in_bin]).sum()
            fn = p[is_cat & listed & in_bin].sum() + p[is_det & ~listed & in_bin].sum()
            precision = tp / (tp + fp) if tp + fp else np.nan
            recall = tp / (tp + fn) if tp + fn else np.nan
            rows.append({
                "detector": name, "bin": orig["bin"],
                "precision": orig["precision"], "recall": orig["recall"], "f1": orig["f1"],
                "precision_adj": precision, "recall_adj": recall,
                "f1_adj": 2 * precision * recall / (precision + recall) if precision + recall else np.nan,
                "TP_adj": tp, "FP_adj": fp, "FN_adj": fn,
            })
    return pd.DataFrame(rows)


def catalog_summary(items, verdicts, n_catalog_counted, bins):
    """How complete and how clean the catalog looks, by bin (estimates until everything is reviewed)."""
    p, verdict = p_real(items, verdicts)
    p = p.to_numpy()
    rows = []
    for label in ["all", *bin_labels(bins)]:
        in_bin = np.ones(len(items), bool) if label == "all" else (items["bin"] == label).to_numpy()
        cat = ((items["kind"] == "catalog").to_numpy()) & in_bin
        det = ((items["kind"] == "detection").to_numpy()) & in_bin
        n_cat = n_catalog_counted[label]
        not_real = (1 - p[cat]).sum()
        new_real = p[det].sum()
        real_in_catalog = n_cat - not_real
        rows.append({
            "bin": label, "catalog": n_cat,
            "reviewed": int((verdict[in_bin].isin(["real", "not_real"]) & (items.loc[in_bin, "auto_verdict"] == "")).sum()),
            "to_review": int((in_bin & (items["auto_verdict"] == "")).sum()),
            "catalog_not_real_est": not_real, "new_real_craters_est": new_real,
            "catalog_completeness_est": real_in_catalog / (real_in_catalog + new_real) if real_in_catalog + new_real else np.nan,
        })
    return pd.DataFrame(rows)


def false_positive_summary(items, verdicts, table, bins):
    """How many of one run's detections are real craters, from a review of its unmatched ones.

    items: the run's unmatched detections (kind "detection"); table: its metrics table.
    Matched detections are taken as real, auto-marked duplicates as false. Of the rest,
    the reviewed share that is real (with its 95% Wilson interval) stands in for the
    unreviewed ones of the same diameter bin. Returns one row per bin, plus "all".
    """
    p, verdict = p_real(items, verdicts)
    auto = items["auto_verdict"].fillna("") != ""
    rows = []
    for label in ["all", *bin_labels(bins)]:
        in_bin = np.ones(len(items), bool) if label == "all" else (items["bin"] == label).to_numpy()
        orig = table.set_index("bin").loc[label]
        human = in_bin & ~auto.to_numpy()
        reviewed = human & verdict.isin(["real", "not_real"]).to_numpy()
        n_reviewed, n_real = int(reviewed.sum()), int((verdict[reviewed] == "real").sum())
        unreviewed = int(human.sum()) - n_reviewed
        rate_lo, rate_hi = wilson_interval(n_real, n_reviewed)
        # "all" sums the per-bin estimates; a bin's own rate fills in its unreviewed items.
        # Until every bin in it has a verdict, the estimate is unknown (NaN), not "all false".
        real = orig["TP"] + p.to_numpy()[in_bin].sum()
        n = orig["TP"] + orig["FP"]
        rows.append({
            "bin": label, "detections": int(n), "matched_thesis": int(orig["TP"]), "unmatched": int(orig["FP"]),
            "auto_duplicates": int((in_bin & auto.to_numpy()).sum()), "reviewed": n_reviewed, "reviewed_real": n_real,
            "real_rate": n_real / n_reviewed if n_reviewed else np.nan, "real_rate_lo": rate_lo, "real_rate_hi": rate_hi,
            "false_est": n - real, "precision_est": real / n if n else np.nan,
            "precision_lo": (orig["TP"] + n_real + rate_lo * unreviewed) / n if n and n_reviewed else np.nan,
            "precision_hi": (orig["TP"] + n_real + rate_hi * unreviewed) / n if n and n_reviewed else np.nan,
        })
    return pd.DataFrame(rows)
