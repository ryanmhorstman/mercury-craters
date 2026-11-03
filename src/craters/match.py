"""One-to-one matching of detections to catalog craters.

A detection can match a catalog crater when
  - the centre distance is <= center_frac x the catalog diameter, and
  - detection diameter / catalog diameter is within [ratio_min, ratio_max].
Among the allowed pairs, the Hungarian algorithm picks the one-to-one
assignment with the lowest total cost, where
  cost = centre distance / catalog diameter + |ln(diameter ratio)|.

The assignment is solved separately for each connected group of candidate
pairs, so the cost matrices stay tiny even for tens of thousands of craters.
"""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .geo import great_circle_km, km_to_chord, unit_vectors

NOT_ALLOWED = 1e9


def candidate_pairs(catalog, detections, match):
    """All (catalog, detection) pairs that pass the matching rule, with their cost."""
    if len(catalog) == 0 or len(detections) == 0:
        return pd.DataFrame(columns=["cat", "det", "distance_km", "ratio", "cost"])
    cat_xyz = unit_vectors(catalog["lat"], catalog["lon"])
    det_tree = cKDTree(unit_vectors(detections["lat"], detections["lon"]))
    cat_d = catalog["diameter_km"].to_numpy()
    radii = km_to_chord(match["center_frac"] * cat_d)

    cat_idx, det_idx = [], []
    for i, near in enumerate(det_tree.query_ball_point(cat_xyz, radii)):
        cat_idx.extend([i] * len(near))
        det_idx.extend(near)
    cat_idx, det_idx = np.array(cat_idx, dtype=int), np.array(det_idx, dtype=int)

    det_d = detections["diameter_km"].to_numpy()
    distance = great_circle_km(catalog["lat"].to_numpy()[cat_idx], catalog["lon"].to_numpy()[cat_idx],
                               detections["lat"].to_numpy()[det_idx], detections["lon"].to_numpy()[det_idx])
    ratio = det_d[det_idx] / cat_d[cat_idx]
    ok = ((distance <= match["center_frac"] * cat_d[cat_idx])
          & (ratio >= match["ratio_min"]) & (ratio <= match["ratio_max"]))
    pairs = pd.DataFrame({"cat": cat_idx[ok], "det": det_idx[ok], "distance_km": distance[ok], "ratio": ratio[ok]})
    pairs["cost"] = pairs["distance_km"] / cat_d[pairs["cat"]] + np.abs(np.log(pairs["ratio"]))
    return pairs


def match_one_to_one(catalog, detections, match):
    """Returns the matched pairs: positional indices into catalog and detections."""
    pairs = candidate_pairs(catalog, detections, match)
    if pairs.empty:
        return pairs

    # Group pairs into connected components of the bipartite candidate graph.
    n_cat, n_det = len(catalog), len(detections)
    graph = coo_matrix((np.ones(len(pairs)), (pairs["cat"], n_cat + pairs["det"])), shape=(n_cat + n_det,) * 2)
    _, component = connected_components(graph, directed=False)
    pairs["component"] = component[pairs["cat"].to_numpy()]

    chosen = []
    for _, group in pairs.groupby("component"):
        cats, cat_pos = np.unique(group["cat"], return_inverse=True)
        dets, det_pos = np.unique(group["det"], return_inverse=True)
        cost = np.full((len(cats), len(dets)), NOT_ALLOWED)
        cost[cat_pos, det_pos] = group["cost"].to_numpy()
        rows, cols = linear_sum_assignment(cost)
        keep = cost[rows, cols] < NOT_ALLOWED
        chosen.append(pd.DataFrame({"cat": cats[rows[keep]], "det": dets[cols[keep]]}))

    chosen = pd.concat(chosen, ignore_index=True)
    return chosen.merge(pairs.drop(columns="component"), on=["cat", "det"])
