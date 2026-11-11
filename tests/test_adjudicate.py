import numpy as np
import pandas as pd
import pytest

from craters.adjudicate import adjusted_metrics, append_verdict, build_items, catalog_summary, latest_verdicts
from craters.metrics import evaluate

BOX = [270, 300, 0, 30]
BINS = [5, 10, 20, 50, np.inf]


def crater(i, lat, lon, d):
    return {"id": f"RH{i:05d}", "lat": lat, "lon": lon, "diameter_km": d}


@pytest.fixture
def scene():
    # Catalogue: 0,1 real and found by both; 2 real, missed by A; 3 not a crater (catalogue error), missed by both.
    catalog = pd.DataFrame([crater(0, 5, 280, 12), crater(1, 10, 285, 25), crater(2, 15, 290, 8), crater(3, 20, 295, 6)])
    a = pd.DataFrame({"lat": [5, 10, 25.0, 12.0], "lon": [280, 285, 275.0, 278.0], "diameter_km": [12, 25, 9, 7], "score": 1.0})
    b = pd.DataFrame({"lat": [5, 10, 15, 25.0], "lon": [280, 285, 290, 275.0], "diameter_km": [12, 25, 8, 9], "score": 1.0})
    # Unmatched: (25N, 275E, 9 km) found by both -> one item, real. (12N, 278E, 7 km) found by A only -> not real.
    truth = {"cat:RH00002": "real", "cat:RH00003": "not_real"}
    return catalog, {"A": a, "B": b}, truth


def test_queue_has_each_disagreement_once(scene, match):
    catalog, detectors, _ = scene
    items, _ = build_items(catalog, detectors, [BOX], match, BINS, seed=0)
    assert sorted(items["kind"]) == ["catalog", "catalog", "detection", "detection"]
    shared = items[(items["kind"] == "detection") & (items["lat"] == 25.0)].iloc[0]
    assert shared["detectors"] == "A;B"
    assert items.loc[items["item_id"] == "cat:RH00002", "detectors"].iloc[0] == "A"


def test_fully_reviewed_adjustment_equals_rescoring_against_the_corrected_catalog(scene, match, tmp_path):
    catalog, detectors, truth = scene
    items, tables = build_items(catalog, detectors, [BOX], match, BINS, seed=0)
    verdict_file = tmp_path / "verdicts.csv"
    for _, item in items.iterrows():
        if item["kind"] == "catalog":
            verdict = truth[item["item_id"]]
        else:
            verdict = "real" if item["lat"] == 25.0 else "not_real"
        append_verdict(verdict_file, item, verdict)
    verdicts = latest_verdicts(verdict_file)

    adjusted = adjusted_metrics(items, verdicts, tables, BINS).set_index(["detector", "bin"])

    corrected = pd.concat([catalog[catalog["id"] != "RH00003"],
                           pd.DataFrame([crater(9, 25.0, 275.0, 9.0)])], ignore_index=True)
    for name, detections in detectors.items():
        expected = evaluate(corrected, detections, [BOX], match, BINS)[0].set_index("bin").loc["all"]
        got = adjusted.loc[(name, "all")]
        assert got["precision_adj"] == pytest.approx(expected["precision"])
        assert got["recall_adj"] == pytest.approx(expected["recall"])

    summary = catalog_summary(items, verdicts, {"all": 4, "5-10": 2, "10-20": 1, "20-50": 1, "50+": 0}, BINS).set_index("bin")
    assert summary.loc["all", "catalog_not_real_est"] == pytest.approx(1)
    assert summary.loc["all", "new_real_craters_est"] == pytest.approx(1)


def test_changing_your_mind_keeps_the_latest_verdict(tmp_path):
    item = {"item_id": "det:x", "lat": 1.0, "lon": 2.0, "diameter_km": 6.0, "kind": "detection"}
    path = tmp_path / "v.csv"
    append_verdict(path, item, "real")
    append_verdict(path, item, "not_real")
    assert latest_verdicts(path) == {"det:x": "not_real"}
    with pytest.raises(ValueError):
        append_verdict(path, item, "maybe")


def test_a_verdict_lapses_when_its_circle_moves(tmp_path):
    from craters.adjudicate import verdicts_for

    path = tmp_path / "v.csv"
    item = {"item_id": "cat:RH00001", "lat": 20.0, "lon": 300.0, "diameter_km": 8.0, "kind": "catalog"}
    append_verdict(path, item, "not_real")
    same = pd.DataFrame([item])
    moved = pd.DataFrame([{**item, "lat": 20.0 - 2.0 / 42.574}])      # re-registered 2 km south
    assert verdicts_for(same, path) == ({"cat:RH00001": "not_real"}, 0)
    assert verdicts_for(moved, path) == ({}, 1)


def test_auto_duplicates_do_not_set_the_rate_for_unreviewed_items():
    from craters.adjudicate import p_real

    items = pd.DataFrame({
        "item_id": ["d1", "d2", "d3", "d4"], "kind": "detection", "bin": "5-10",
        "auto_verdict": ["duplicate", "", "", ""],
    })
    p, _ = p_real(items, {})
    assert p.iloc[0] == 0.0 and p.iloc[1:].isna().all()          # unknown, not "0% real"
    p, _ = p_real(items, {"d2": "real"})
    assert p.tolist() == [0.0, 1.0, 1.0, 1.0]                     # the one human verdict sets the rate


def test_second_circle_on_a_catalogued_crater_is_auto_marked_duplicate(match):
    catalog = pd.DataFrame([crater(0, 10, 285, 20)])
    # Two detections on the same 20 km crater: one matches, the leftover is a duplicate, not a new crater.
    det = pd.DataFrame({"lat": [10.0, 10.02], "lon": [285.0, 285.02], "diameter_km": [20.0, 17.0], "score": [0.9, 0.5]})
    items, _ = build_items(catalog, {"A": det}, [BOX], match, BINS, seed=0)
    assert items["auto_verdict"].tolist() == ["duplicate"]


def test_false_positive_summary_from_a_partial_review(match):
    from craters.adjudicate import false_positive_summary

    catalog = pd.DataFrame([crater(0, 5, 280, 12)])
    detections = pd.DataFrame({"lat": [5.0, 10.0, 15.0, 20.0, 25.0], "lon": [280.0] * 5,
                               "diameter_km": [12, 6, 6, 6, 6], "score": 1.0})   # one matched, four unmatched 5-10 km
    items, tables = build_items(catalog, {"run": detections}, [BOX], match, BINS, seed=0)
    table = tables["run"]
    nothing = false_positive_summary(items, {}, table, BINS).set_index("bin")
    assert np.isnan(nothing.loc["5-10", "precision_est"])          # no verdicts yet: unknown, not "all false"

    by_lat = items.set_index("lat")["item_id"]
    verdicts = {by_lat[10.0]: "real", by_lat[15.0]: "not_real"}  # half of the reviewed ones are real
    out = false_positive_summary(items, verdicts, table, BINS).set_index("bin")
    row = out.loc["5-10"]
    assert (row["detections"], row["unmatched"], row["reviewed"], row["reviewed_real"]) == (4, 4, 2, 1)
    assert row["precision_est"] == pytest.approx(0.5) and row["false_est"] == pytest.approx(2.0)
    assert row["precision_lo"] < 0.5 < row["precision_hi"]
    assert out.loc["all", "precision_est"] == pytest.approx(3 / 5)  # + the matched 12 km crater
