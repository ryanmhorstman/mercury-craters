import pandas as pd

from craters.supplement import build_supplemented, drop_known, f1_summary, score_run


def circles(rows):
    return pd.DataFrame(rows, columns=["lat", "lon", "diameter_km"])


def thesis_catalog():
    catalog = circles([(15.0, 285.0, 30.0), (-15.0, 165.0, 50.0)])       # one in T1, one in V1
    return catalog.assign(id=["RH00001", "RH00002"], source=["Central Peaks", "Simple Craters"], split=["T1", "V1"])


def test_drop_known_removes_listed_craters_and_repeats(match):
    known = circles([(0.0, 10.0, 40.0)])
    new = circles([(0.1, 10.0, 44.0),      # same crater as the known one
                   (0.0, 10.0, 100.0),     # same centre, far bigger: a different crater
                   (30.0, 10.0, 60.0),     # new
                   (30.1, 10.0, 55.0)])    # the same new crater again, smaller
    assert drop_known(known, new, match)["diameter_km"].tolist() == [100.0, 60.0]


def test_build_supplemented_labels_sources_and_keeps_the_thesis_rows(match, regions):
    catalog = thesis_catalog()
    confirmed = circles([(15.05, 285.0, 31.0), (20.0, 290.0, 25.0)]).assign(cnn_score=[0.4, 0.3])   # first is RH00001
    scan = circles([(20.0, 290.05, 26.0), (40.0, 100.0, 80.0)]).assign(cnn_score=[0.05, 0.03])      # first is the confirmed one
    out = build_supplemented(catalog, confirmed, scan, match, regions)

    assert out["source"].tolist() == ["thesis", "thesis", "dem_confirmed", "dem_scan_cnn"]
    assert out["id"].tolist() == ["RH00001", "RH00002", "DC00001", "DS00001"]
    assert out["split"].tolist() == ["T1", "V1", "T1", "train"]
    assert out["sheet"].tolist()[:2] == ["Central Peaks", "Simple Craters"]      # the thesis sheet survives as `sheet`
    thesis = out[out["source"] == "thesis"]
    assert thesis[["lat", "lon", "diameter_km"]].equals(catalog[["lat", "lon", "diameter_km"]])
    assert catalog["source"].tolist() == ["Central Peaks", "Simple Craters"]     # the input isn't modified


def test_score_run_counts_a_new_crater_as_found_only_on_the_supplemented_catalog(match, regions):
    cfg = {"regions": regions, "match": match, "evaluation": {"diameter_bins_km": [5, 20, float("inf")]}}
    catalog = thesis_catalog()
    supplemented = pd.concat([catalog, circles([(20.0, 290.0, 25.0)])], ignore_index=True)
    detections = circles([(15.0, 285.0, 30.0), (20.0, 290.0, 25.0)])            # the T1 thesis crater + the new one

    scores = score_run(cfg, detections, {"thesis": catalog, "supplemented": supplemented})
    row = scores.set_index(["catalog", "region", "bin"])
    assert row.loc[("thesis", "T1", "all"), ["TP", "FP", "FN"]].tolist() == [1, 1, 0]
    assert row.loc[("supplemented", "T1", "all"), ["TP", "FP", "FN"]].tolist() == [2, 0, 0]
    assert row.loc[("thesis", "V1", "all"), ["TP", "FP", "FN"]].tolist() == [0, 0, 1]
    assert set(scores["region"]) == {"V1", "T1", "T1-T1"}                       # the fixture has one test region

    f1 = f1_summary(scores)
    assert f1.columns.tolist() == ["thesis", "supplemented"]
    assert f1.loc[("T1", "all"), "supplemented"] == 1.0
