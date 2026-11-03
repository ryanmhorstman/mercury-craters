import numpy as np
import pandas as pd
import pytest

from craters.registration import KM_PER_DEG, register

PARAMS = {"min_diameter_km": 10.0, "radius_deg": 5.0, "min_pairs": 10, "iterations": 2}


def grid_catalog(seed=0, n=400):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "id": [f"RH{i:05d}" for i in range(n)],
        "lat": rng.uniform(0, 30, n), "lon": rng.uniform(270, 300, n),
        "diameter_km": rng.uniform(6, 40, n),
    })


def test_recovers_a_shift_that_varies_across_the_region(match):
    catalog = grid_catalog()
    # True craters sit south of the catalog: 1 km in the south half, 4 km in the north half.
    true_shift = np.where(catalog["lat"] < 15, -1.0, -4.0)
    truth = catalog.assign(lat=catalog["lat"] + true_shift / KM_PER_DEG)
    rng = np.random.default_rng(1)
    det_a = truth.assign(lon=truth["lon"] + 1.0 / KM_PER_DEG, score=1.0)      # detector A: biased 1 km east
    det_b = truth.assign(lon=truth["lon"] - 1.0 / KM_PER_DEG, score=1.0)      # detector B: biased 1 km west
    det_a["lat"] += rng.normal(0, 0.3, len(det_a)) / KM_PER_DEG

    corrected, history = register(catalog, {"a": det_a, "b": det_b}, match, PARAMS)

    # Away from the boundary between the two zones, the north shift is recovered to ~0.3 km.
    clear = (catalog["lat"] - 15).abs() > 6
    error = corrected.loc[clear, "north_shift_km"] - true_shift[clear]
    assert np.abs(error).max() < 0.5
    assert (corrected["lon"] == catalog["lon"]).all()                       # east-west untouched
    assert (corrected["lat_catalog"] == catalog["lat"]).all()               # originals kept
    assert len(history) == 2


def test_too_few_pairs_leaves_craters_alone(match):
    catalog = grid_catalog(n=5)
    detections = catalog.assign(lat=catalog["lat"] - 2 / KM_PER_DEG, score=1.0)
    corrected, _ = register(catalog, {"a": detections}, match, PARAMS)
    assert not corrected["registered"].any()
    assert (corrected["lat"] == catalog["lat"]).all()
