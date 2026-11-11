import numpy as np
import pandas as pd
import pytest

from craters.compare_catalogs import KM_PER_DEG, listed_in, offsets_from, shift_onto, union_catalog

CFG_COMPARE = {"offset_min_km": 10, "offset_lat_bands": [[-90, -30], [-30, 30], [30, 90]]}


def craters(n=60, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"lat": rng.uniform(-25, 25, n), "lon": rng.uniform(10, 350, n), "diameter_km": rng.uniform(20, 40, n)})


def moved(frame, north_km, east_km):
    out = frame.copy()
    out["lat"] += north_km / KM_PER_DEG
    out["lon"] = (out["lon"] + east_km / (KM_PER_DEG * np.cos(np.radians(frame["lat"])))) % 360
    return out


def test_offsets_are_measured_and_removed_per_band(match):
    reference = craters()
    other = moved(reference, 1.0, -2.0)
    offset = offsets_from(reference, other, match, 10)
    assert offset["pairs"] == 60
    assert offset["north_km"] == pytest.approx(1.0, abs=0.02) and offset["east_km"] == pytest.approx(-2.0, abs=0.02)

    back, table = shift_onto(reference, other, {"match": match, "compare": CFG_COMPARE})
    assert table["pairs"].tolist() == [0, 60, 0]                    # empty bands are left alone
    assert np.allclose(back["lat"], reference["lat"], atol=1e-3)
    assert np.allclose(back["lon"], reference["lon"], atol=1e-3)


def test_listed_in_and_union(match):
    catalog = pd.DataFrame({"lat": [0.0, 10.0], "lon": [20.0, 20.0], "diameter_km": [10.0, 30.0]})
    other = pd.DataFrame({"lat": [0.02, 40.0, 10.0], "lon": [20.0, 20.0, 20.0], "diameter_km": [10.5, 15.0, 80.0]})
    assert listed_in(other, catalog, match).tolist() == [True, False, False]   # same crater; elsewhere; far bigger
    union = union_catalog(catalog, other, match)
    assert len(union) == 4 and union["diameter_km"].tolist()[:2] == [10.0, 30.0]
