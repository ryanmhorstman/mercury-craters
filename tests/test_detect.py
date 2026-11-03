import cv2
import numpy as np
import pandas as pd
import pytest

from craters.detect import to_lonlat
from craters.detectors import hough
from craters.tiling import Tile

LEVEL = {"detect_km": [5, 40]}
TILE = Tile("t", "fine", 20.0, 285.0, 512, 166.31)


def ring_image(circles, size=512):
    """Mid-grey tile with a bright sunlit arc and a dark shadowed arc per crater."""
    image = np.full((size, size), 110, np.uint8)
    for col, row, r in circles:
        centre = (int(round(col - 0.5)), int(round(row - 0.5)))
        cv2.ellipse(image, centre, (int(r), int(r)), 0, 200, 340, 220, 3)
        cv2.ellipse(image, centre, (int(r), int(r)), 0, 20, 160, 20, 3)
    return image


def test_hough_finds_drawn_craters_at_the_right_place_and_size():
    truth = [(150.5, 150.5, 40.0), (360.5, 300.5, 25.0), (260.5, 420.5, 60.0)]
    found = hough.detect(ring_image(truth), TILE, LEVEL)
    for col, row, r in truth:
        # Neighbouring radius bands can both fire on one crater; the overlap
        # merge keeps the higher score, so judge the best-scoring nearby circle.
        near = found[np.hypot(found["col"] - col, found["row"] - row) < 0.15 * r]
        assert not near.empty
        best = near.loc[near["score"].idxmax()]
        assert best["radius_px"] == pytest.approx(r, rel=0.15)
        assert best["score"] > 0.5


def test_circles_cut_by_the_tile_edge_or_out_of_range_are_dropped():
    circles = pd.DataFrame({
        "col": [256.0, 10.0, 256.0, 256.0],
        "row": [256.0, 256.0, 256.0, 256.0],
        "radius_px": [60.0, 30.0, 10.0, 200.0],   # ok / cut by edge / 3.3 km (too small) / 66.5 km (too big)
        "score": [0.9, 0.9, 0.9, 0.9],
    })
    out = to_lonlat(TILE, circles, LEVEL)
    assert len(out) == 1
    assert out["diameter_km"].iloc[0] == pytest.approx(120 * 166.31 / 1000)
    assert out["lat"].iloc[0] == pytest.approx(20.0, abs=0.01)
    assert out["lon"].iloc[0] == pytest.approx(285.0, abs=0.01)
