"""Merge circles that describe the same crater.

Used for duplicate entries in the catalog and for detections repeated across
overlapping tiles. Two circles are "the same crater" when, measured against the
circle being kept:
  - their centres are within center_frac x its diameter, and
  - their diameter ratio is within [ratio_min, ratio_max].
"""
import numpy as np
from scipy.spatial import cKDTree

from .geo import MERCURY_RADIUS_M, great_circle_km, km_to_chord, unit_vectors


def same_crater(lat_a, lon_a, d_a, lat_b, lon_b, d_b, center_frac, ratio_min, ratio_max,
                radius_m=MERCURY_RADIUS_M):
    """Vectorised test of the matching rule, with a as the reference circle."""
    distance = great_circle_km(lat_a, lon_a, lat_b, lon_b, radius_m)
    ratio = np.asarray(d_b) / np.asarray(d_a)
    return (distance <= center_frac * np.asarray(d_a)) & (ratio >= ratio_min) & (ratio <= ratio_max)


def dedupe_circles(lat, lon, diameter_km, order, center_frac, ratio_min, ratio_max,
                   radius_m=MERCURY_RADIUS_M):
    """Greedy de-duplication.

    Circles are visited in `order` (best first: highest score, or highest
    priority). Each circle not yet claimed is kept and claims every later
    circle that matches it.

    Returns duplicate_of: an int array, -1 for kept circles, otherwise the
    index of the kept circle that absorbed it.
    """
    lat, lon, diameter_km = (np.asarray(a, dtype=float) for a in (lat, lon, diameter_km))
    xyz = unit_vectors(lat, lon)
    tree = cKDTree(xyz)
    duplicate_of = np.full(len(lat), -1)
    claimed = np.zeros(len(lat), dtype=bool)

    for i in order:
        if claimed[i]:
            continue
        claimed[i] = True
        search_radius = km_to_chord(center_frac * diameter_km[i], radius_m)
        candidates = np.array([j for j in tree.query_ball_point(xyz[i], search_radius) if not claimed[j]], dtype=int)
        if candidates.size == 0:
            continue
        n = candidates.size
        hits = candidates[same_crater(
            np.full(n, lat[i]), np.full(n, lon[i]), np.full(n, diameter_km[i]),
            lat[candidates], lon[candidates], diameter_km[candidates],
            center_frac, ratio_min, ratio_max, radius_m,
        )]
        claimed[hits] = True
        duplicate_of[hits] = i
    return duplicate_of


def merge_overlap_detections(detections, match):
    """Collapse detections repeated across overlapping tiles, keeping the highest score.

    detections: DataFrame with lat, lon, diameter_km, score (one row per tile detection).
    """
    order = np.argsort(-detections["score"].to_numpy(), kind="stable")
    duplicate_of = dedupe_circles(
        detections["lat"], detections["lon"], detections["diameter_km"], order,
        match["center_frac"], match["ratio_min"], match["ratio_max"],
    )
    merged = detections[duplicate_of < 0].copy()
    merged["n_tiles"] = 1 + np.bincount(duplicate_of[duplicate_of >= 0], minlength=len(detections))[duplicate_of < 0]
    return merged.reset_index(drop=True)
