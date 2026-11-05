import numpy as np
import pandas as pd

from craters.demcheck import Dem, bowl_features, choose_rule, passes_banded, polar_grid
from craters.geo import great_circle_km
from craters.selftrain import near_known

PARAMS = {"azimuths": 36, "rim_min_frac": 0.01, "depth_ratio_grid": [0.0, 0.01, 0.02],
          "rim_frac_grid": [0.5, 0.8], "max_decoy_pass": 0.01}


def synthetic_dem(lat0, lon0, diameter_km, depth_m=1000.0, rim_m=300.0):
    """A whole-planet DEM at 0.05 deg with one bowl-shaped crater, rim and all."""
    dem = Dem.__new__(Dem)
    dem.height, dem.width = 3600, 7200
    rows, cols = np.mgrid[0:dem.height, 0:dem.width]
    lat, lon = 90.0 - (rows + 0.5) * 0.05, (cols + 0.5) * 0.05
    near = (np.abs(lat - lat0) < 5) & (np.abs(lon - lon0) < 5)
    r = np.zeros(lat.shape)
    r[near] = great_circle_km(lat[near], lon[near], np.full(near.sum(), lat0), np.full(near.sum(), lon0)) / (diameter_km / 2)
    z = np.where(r < 1, -depth_m * (1 - r ** 2), rim_m * np.exp(-((r - 1) / 0.25) ** 2))
    z[~near] = 0.0
    dem.data = np.round(z * 2).astype(np.int16)          # half-metres, like the USGS DEM
    dem.nodata, dem.scale, dem.offset = -32768, 0.5, 0.0
    return dem


def test_polar_grid_distances():
    lat, lon = polar_grid(30.0, 100.0, 20.0, np.array([0.5, 1.0]), np.linspace(0, 2 * np.pi, 8, endpoint=False))
    d = great_circle_km(np.full(lat.size, 30.0), np.full(lat.size, 100.0), lat.ravel(), lon.ravel()).reshape(lat.shape)
    assert np.allclose(d[0], 10.0, atol=0.01) and np.allclose(d[1], 20.0, atol=0.01)


def test_bowl_features_see_the_crater_and_not_flat_ground():
    dem = synthetic_dem(10.0, 50.0, 60.0)
    f = bowl_features(dem, [10.0, 10.0], [50.0, 54.0], [60.0, 60.0], PARAMS)
    crater, flat = f.iloc[0], f.iloc[1]
    assert crater["depth_ratio"] > 0.015 and crater["rim_frac"] == 1.0 and crater["floor_drop"] > 0
    assert abs(flat["depth_ratio"]) < 1e-3 and flat["rim_frac"] == 0.0


def test_choose_rule_keeps_decoys_out():
    real = pd.DataFrame({"depth_ratio": [0.03] * 9 + [0.005], "rim_frac": [1.0] * 10, "floor_drop": [0.01] * 10})
    decoy = pd.DataFrame({"depth_ratio": [0.015, 0.015] + [0.0] * 98, "rim_frac": [0.9] * 100, "floor_drop": [0.01] * 100})
    rule = choose_rule(real, decoy, PARAMS)
    assert rule["decoy_pass"] <= 0.01 and rule["min_depth_ratio"] == 0.02 and rule["real_pass"] == 0.9


def test_passes_banded_uses_each_bands_rule():
    features = pd.DataFrame({"depth_ratio": [0.01, 0.01, 0.01], "rim_frac": [1.0] * 3, "floor_drop": [0.01] * 3})
    rules = [{"min_km": 20, "max_km": 40, "min_depth_ratio": 0.0, "min_rim_frac": 0.5},
             {"min_km": 40, "max_km": 400, "min_depth_ratio": 0.02, "min_rim_frac": 0.5}]
    assert passes_banded(features, [25.0, 50.0, 10.0], rules).tolist() == [True, False, False]


def test_near_known():
    catalog = pd.DataFrame({"lat": [0.0, 20.0, 40.0], "lon": [10.0, 10.0, 10.0], "diameter_km": [30.0, 200.0, 12.0]})
    detections = pd.DataFrame({
        "lat": [0.1, 20.5, 40.0, 60.0],
        "lon": [10.0, 10.0, 10.05, 10.0],
        "diameter_km": [50.0, 40.0, 40.0, 40.0],   # mis-sized copy, inside a basin, round a small crater, new
    })
    assert near_known(detections, catalog).tolist() == [True, True, True, False]


def test_scan_centres_spacing():
    from craters.demscan import scan_centres

    lat, lon = scan_centres(100.0, 0.2, 60)
    on_equator = np.isclose(lat, lat[np.argmin(np.abs(lat))])
    spacing_km = 360.0 / on_equator.sum() * np.radians(1) * 2439.4
    assert 19.0 < spacing_km <= 20.0 and lat.min() == -60


def test_merge_keeps_the_best_bowl_and_drops_inner_rings():
    from craters.demscan import merge

    candidates = pd.DataFrame({
        "lat": [0.0, 0.1, 0.0, 30.0], "lon": [10.0, 10.0, 10.05, 10.0],
        "diameter_km": [100.0, 115.0, 40.0, 50.0],          # a crater, a worse copy of it, its inner ring, another crater
        "rim_frac": [1.0, 0.9, 1.0, 1.0], "depth_ratio": [0.03, 0.02, 0.02, 0.02], "floor_drop": [0.01] * 4,
    })
    kept = merge(candidates)
    assert sorted(kept["diameter_km"]) == [50.0, 100.0]


def test_best_match_score_takes_the_best_matching_detection():
    from craters.demscan import best_match_score

    circles = pd.DataFrame({"lat": [0.0, 30.0], "lon": [10.0, 10.0], "diameter_km": [100.0, 100.0]})
    detections = pd.DataFrame({
        "lat": [0.2, 0.0, 0.0, 30.0], "lon": [10.0, 10.0, 10.0, 10.0],
        "diameter_km": [110.0, 100.0, 40.0, 300.0],          # match, better match, too small, too big
        "score": [0.1, 0.3, 0.9, 0.9],
    })
    assert best_match_score(circles, detections).tolist() == [0.3, 0.0]
