"""Precision, recall, F1 and diameter error, overall and by diameter bin.

Matching runs on the full catalog and detection sets; counting is then scoped
to a region and a minimum diameter:
  - TP: a matched pair whose catalog crater is counted (in region, >= min size)
  - FN: a counted catalog crater with no match
  - FP: a counted detection (in region, >= min size) with no match
A detection matched to an uncounted crater (just outside the region, or just
under the size cut) is neither TP nor FP, so edge effects don't penalise it.

By bin, TP and FN go by catalog diameter and FP by detection diameter.
"""
import numpy as np
import pandas as pd

from .geo import in_box
from .match import match_one_to_one


def in_boxes(lat, lon, boxes):
    inside = np.zeros(len(lat), dtype=bool)
    for box in boxes:
        inside |= in_box(lat, lon, box)
    return inside


def bin_labels(bins):
    return [f"{lo:g}-{hi:g}" if np.isfinite(hi) else f"{lo:g}+" for lo, hi in zip(bins[:-1], bins[1:])]


def wilson_interval(successes, trials, z=1.96):
    """95% Wilson score interval for a proportion; (nan, nan) when trials == 0."""
    if trials == 0:
        return np.nan, np.nan
    p = successes / trials
    denom = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denom
    half = z * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denom
    return centre - half, centre + half


def label_outcomes(catalog, detections, boxes, match, min_diameter_km):
    """Match, then mark each catalog crater and detection as TP / FN / FP / not counted.

    Returns (catalog, detections) copies with columns `counted`, `outcome` and,
    for matched rows, the partner's index and diameter.
    """
    catalog = catalog.reset_index(drop=True).copy()
    detections = detections.reset_index(drop=True).copy()
    pairs = match_one_to_one(catalog, detections, match)

    catalog["counted"] = in_boxes(catalog["lat"], catalog["lon"], boxes) & (catalog["diameter_km"] >= min_diameter_km)
    detections["counted"] = in_boxes(detections["lat"], detections["lon"], boxes) & (detections["diameter_km"] >= min_diameter_km)

    catalog["matched_det"] = -1
    detections["matched_cat"] = -1
    catalog.loc[pairs["cat"], "matched_det"] = pairs["det"].to_numpy()
    detections.loc[pairs["det"], "matched_cat"] = pairs["cat"].to_numpy()
    catalog["det_diameter_km"] = np.nan
    catalog.loc[pairs["cat"], "det_diameter_km"] = detections["diameter_km"].to_numpy()[pairs["det"]]
    catalog["center_offset_frac"] = np.nan
    catalog.loc[pairs["cat"], "center_offset_frac"] = (pairs["distance_km"] / catalog["diameter_km"].to_numpy()[pairs["cat"]]).to_numpy()

    catalog["outcome"] = np.where(~catalog["counted"], "", np.where(catalog["matched_det"] >= 0, "TP", "FN"))

    partner = detections["matched_cat"].to_numpy()
    has_partner = partner >= 0
    partner_counted = np.zeros(len(detections), dtype=bool)
    partner_counted[has_partner] = catalog["counted"].to_numpy()[partner[has_partner]]
    detections["outcome"] = np.where(has_partner & partner_counted, "TP",
                                     np.where(detections["counted"] & ~has_partner, "FP", ""))
    return catalog, detections


def score(catalog_rows, detection_rows):
    """Metrics for already-labelled rows (one bin, or everything)."""
    tp = int((catalog_rows["outcome"] == "TP").sum())
    fn = int((catalog_rows["outcome"] == "FN").sum())
    fp = int((detection_rows["outcome"] == "FP").sum())
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * precision * recall / (precision + recall) if tp else (0.0 if tp + fp + fn else np.nan)
    matched = catalog_rows[catalog_rows["outcome"] == "TP"]
    rel_err = (matched["det_diameter_km"] - matched["diameter_km"]) / matched["diameter_km"]
    p_lo, p_hi = wilson_interval(tp, tp + fp)
    r_lo, r_hi = wilson_interval(tp, tp + fn)
    return {
        "n_catalog": tp + fn, "n_detections": tp + fp, "TP": tp, "FP": fp, "FN": fn,
        "precision": precision, "precision_lo": p_lo, "precision_hi": p_hi,
        "recall": recall, "recall_lo": r_lo, "recall_hi": r_hi, "f1": f1,
        "diam_err_median": rel_err.median() if len(rel_err) else np.nan,       # signed: + means detector says bigger
        "diam_err_abs_median": rel_err.abs().median() if len(rel_err) else np.nan,
        "center_offset_median": matched["center_offset_frac"].median() if len(matched) else np.nan,
    }


def evaluate(catalog, detections, boxes, match, bins):
    """Metrics table: one row for all sizes, then one per diameter bin."""
    min_d = bins[0]
    cat, det = label_outcomes(catalog, detections, boxes, match, min_d)
    rows = [{"bin": "all", **score(cat, det)}]
    cat_bin = pd.cut(cat["diameter_km"], bins, right=False, labels=bin_labels(bins))
    det_bin = pd.cut(det["diameter_km"], bins, right=False, labels=bin_labels(bins))
    for label in bin_labels(bins):
        rows.append({"bin": label, **score(cat[cat_bin == label], det[det_bin == label])})
    return pd.DataFrame(rows), cat, det
