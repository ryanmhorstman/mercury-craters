import numpy as np
import pandas as pd
import pytest

from craters.match import match_one_to_one
from craters.metrics import evaluate, label_outcomes
from craters.sfd import box_area_km2, cumulative_sfd, sqrt2_diameters
from craters.synthetic import noisy_detections

BOX = [270, 300, 0, 30]
BINS = [5, 10, 20, 50, np.inf]


def synthetic_catalog(n=600, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "lat": np.degrees(np.arcsin(rng.uniform(0, np.sin(np.radians(30)), n))),
        "lon": rng.uniform(270, 300, n),
        "diameter_km": 5 * (1 - rng.random(n)) ** (-1 / 1.8),        # power law above 5 km
    })


def crater(lat, lon, d):
    return {"lat": lat, "lon": lon, "diameter_km": d}


def test_catalog_against_itself_is_perfect(match):
    catalog = synthetic_catalog()
    table, _, _ = evaluate(catalog, catalog.copy(), [BOX], match, BINS)
    for _, row in table.iterrows():
        if row["n_catalog"] == 0:
            continue
        assert row["precision"] == 1.0 and row["recall"] == 1.0 and row["f1"] == 1.0, row["bin"]
        assert row["diam_err_abs_median"] == 0.0
        assert row["center_offset_median"] == pytest.approx(0.0, abs=1e-6)
    assert table.loc[0, "TP"] == len(catalog)


def test_thresholds_are_inclusive_and_exclusive_where_they_should_be(match):
    # Four 10 km craters at 10N. One degree of longitude there is 42.574 km x cos(10 deg).
    catalog = pd.DataFrame([crater(10.0, 285.0, 10.0)] * 4).assign(lon=[280.0, 285.0, 290.0, 295.0])
    per_km_lon = 1 / (42.574 * np.cos(np.radians(10.0)))
    detections = pd.DataFrame([
        crater(10.0, 280.0 + 4.9 * per_km_lon, 10.0),   # 4.9 km off: inside 0.5 D -> match
        crater(10.0, 285.0 + 5.2 * per_km_lon, 10.0),   # 5.2 km off: outside -> no match
        crater(10.0, 290.0, 13.2),                      # ratio 1.32 -> match
        crater(10.0, 295.0, 7.4),                       # ratio 0.74 -> no match
    ])
    pairs = match_one_to_one(catalog, detections, match)
    assert sorted(pairs["cat"].tolist()) == [0, 2]


def test_one_to_one_two_detections_on_one_crater(match):
    catalog = pd.DataFrame([crater(15.0, 285.0, 20.0)])
    detections = pd.DataFrame([crater(15.0, 285.0, 20.5), crater(15.0, 285.01, 19.0)])
    table, _, det = evaluate(catalog, detections, [BOX], match, BINS)
    assert table.loc[0, ["TP", "FP", "FN"]].tolist() == [1, 1, 0]
    assert det.loc[0, "outcome"] == "TP"       # the closer, better-sized one wins


def test_hungarian_beats_greedy(match):
    # Greedy would give detection 0 to crater 0 (its best fit) and leave crater 1 unmatched.
    catalog = pd.DataFrame([crater(15.0, 285.0, 20.0), crater(15.0, 285.25, 20.0)])
    detections = pd.DataFrame([crater(15.0, 285.12, 20.0), crater(15.0, 285.0 - 0.10, 20.0)])
    pairs = match_one_to_one(catalog, detections, match)
    assert len(pairs) == 2


def test_edge_and_size_cut_matches_are_not_false_positives(match):
    # Catalog crater is 4.8 km (below the cut); its detection is 5.2 km. Neither TP nor FP.
    # Catalog crater just outside the box; detection just inside. Neither TP nor FP.
    catalog = pd.DataFrame([crater(15.0, 285.0, 4.8), crater(-0.01, 285.0, 10.0)])
    detections = pd.DataFrame([crater(15.0, 285.0, 5.2), crater(0.02, 285.0, 10.0)])
    cat, det = label_outcomes(catalog, detections, [BOX], match, 5.0)
    assert cat["outcome"].tolist() == ["", ""]
    assert det["outcome"].tolist() == ["", ""]


def test_noisy_detector_metrics_match_what_was_injected(match):
    catalog = synthetic_catalog(n=3000, seed=2)
    rng = np.random.default_rng(3)
    detections, found = noisy_detections(catalog, [BOX], rng, center_sigma_frac=0.05, diameter_sigma=0.04)
    table, cat, det = evaluate(catalog, detections, [BOX], match, BINS)
    overall = table.iloc[0]
    true_recall = found[cat["counted"].to_numpy()].mean()
    assert overall["recall"] == pytest.approx(true_recall, abs=0.02)
    injected_fp = (detections["fake"] & det["counted"]).sum()
    assert overall["FP"] == pytest.approx(injected_fp, rel=0.15)   # a few fakes land on real craters
    assert abs(overall["diam_err_median"]) < 0.02


def test_sfd_counts_and_poisson_errors():
    grid = np.array([5.0, 10.0, 20.0])
    density, err, counts = cumulative_sfd([5, 6, 12, 25], 100.0, grid)
    assert counts.tolist() == [4, 2, 1]
    assert np.allclose(density, [0.04, 0.02, 0.01])
    assert np.allclose(err, np.sqrt([4, 2, 1]) / 100.0)


def test_sqrt2_grid_and_area():
    assert np.allclose(sqrt2_diameters(5.0, 12.0), [5.657, 8.0, 11.314], atol=1e-3)
    whole_planet = box_area_km2([0, 360, -90, 90])
    assert whole_planet == pytest.approx(4 * np.pi * 2439.4**2)
