import numpy as np
import pandas as pd

from craters.dedupe import dedupe_circles, merge_overlap_detections


def test_duplicates_merge_and_distinct_craters_survive(match):
    # 0 and 1: same crater (2 km apart, 10 vs 11 km). 2: same place, very different size.
    # 3: 10 km crater 20 km away, a separate crater.
    lat = np.array([10.0, 10.0, 10.0, 10.0])
    lon = np.array([100.0, 100.047, 100.0, 100.47])
    diameter = np.array([10.0, 11.0, 30.0, 10.0])
    duplicate_of = dedupe_circles(lat, lon, diameter, np.arange(4), **match)
    assert duplicate_of.tolist() == [-1, 0, -1, -1]


def test_priority_order_decides_which_record_is_kept(match):
    lat, lon, diameter = [0.0, 0.0], [50.0, 50.0], [20.0, 20.0]
    assert dedupe_circles(lat, lon, diameter, [1, 0], **match).tolist() == [1, -1]


def test_overlap_merge_keeps_best_score(match):
    detections = pd.DataFrame({
        "lat": [5.0, 5.0, 5.0], "lon": [10.0, 10.01, 30.0],
        "diameter_km": [12.0, 12.5, 12.0], "score": [0.6, 0.9, 0.7], "tile_id": ["a", "b", "b"],
    })
    merged = merge_overlap_detections(detections, match)
    assert sorted(merged["score"]) == [0.7, 0.9]
    assert merged.set_index("score").loc[0.9, "n_tiles"] == 2
