"""Correct the catalog's north-south registration against the current basemap.

The thesis catalog was measured in JMARS on a 2015 MDIS mosaic. On the USGS
basemap used here, the same craters sit up to several km further south, and
by different amounts in different places (individual source images were
registered differently). Both detectors see the same north-south offset,
while their east-west offsets differ by a steady ~2 km (a lighting bias of
the detectors), so only the north-south component is corrected.

For each catalog crater, the shift is the median north offset (detection
minus catalog) of matched craters >= min_diameter_km within radius_deg,
pooled across detectors. Large shifts push some craters past the matching
limit, so the estimate is repeated after applying the first correction.
The original coordinates are kept in lat_catalog / lon_catalog.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .geo import MERCURY_RADIUS_M, km_to_chord, unit_vectors
from .match import match_one_to_one

KM_PER_DEG = np.radians(1.0) * MERCURY_RADIUS_M / 1000.0


def matched_offsets(catalog, detectors, match, min_diameter_km):
    """North offset (km) of every matched pair with a catalog crater >= min_diameter_km."""
    rows = []
    for name, detections in detectors.items():
        pairs = match_one_to_one(catalog, detections, match)
        cat = catalog.iloc[pairs["cat"].to_numpy()]
        det = detections.iloc[pairs["det"].to_numpy()]
        rows.append(pd.DataFrame({
            "detector": name, "lat": cat["lat"].to_numpy(), "lon": cat["lon"].to_numpy(),
            "diameter_km": cat["diameter_km"].to_numpy(),
            "north_km": (det["lat"].to_numpy() - cat["lat"].to_numpy()) * KM_PER_DEG,
        }))
    offsets = pd.concat(rows, ignore_index=True)
    return offsets[offsets["diameter_km"] >= min_diameter_km].reset_index(drop=True)


def local_north_shift(catalog, offsets, radius_deg, min_pairs):
    """Median north offset of the pairs within radius_deg of each catalog crater (NaN if too few)."""
    tree = cKDTree(unit_vectors(offsets["lat"], offsets["lon"]))
    radius = km_to_chord(radius_deg * KM_PER_DEG)
    north = offsets["north_km"].to_numpy()
    shift = np.full(len(catalog), np.nan)
    counts = np.zeros(len(catalog), dtype=int)
    for i, near in enumerate(tree.query_ball_point(unit_vectors(catalog["lat"], catalog["lon"]), radius)):
        counts[i] = len(near)
        if len(near) >= min_pairs:
            shift[i] = np.median(north[near])
    return shift, counts


def register(catalog, detectors, match, params):
    """Returns (corrected catalog, per-iteration summary)."""
    corrected = catalog.copy()
    corrected["lat_catalog"], corrected["lon_catalog"] = catalog["lat"], catalog["lon"]
    corrected["north_shift_km"] = 0.0
    history = []
    for iteration in range(1, params["iterations"] + 1):
        offsets = matched_offsets(corrected, detectors, match, params["min_diameter_km"])
        residual, counts = local_north_shift(corrected, offsets, params["radius_deg"], params["min_pairs"])
        known = ~np.isnan(residual)
        corrected.loc[known, "north_shift_km"] += residual[known]
        corrected["lat"] = corrected["lat_catalog"] + corrected["north_shift_km"] / KM_PER_DEG
        history.append({"iteration": iteration, "pairs_used": len(offsets),
                        "median_residual_km": float(np.median(offsets["north_km"])) if len(offsets) else np.nan,
                        "craters_corrected": int(known.sum())})
    corrected["registered"] = corrected["north_shift_km"] != 0.0
    return corrected, pd.DataFrame(history)
