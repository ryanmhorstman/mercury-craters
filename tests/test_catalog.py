from openpyxl import Workbook

from craters.catalog import build_catalog, load_workbook_table


def write_workbook(path):
    wb = Workbook()
    peaks = wb.active
    peaks.title = "Central Peaks"
    peaks.append(["longitude E", "latitude N", "diameter (km)", "Dpk (km)", "Type", "Dpk/Dc"])
    peaks.append([285.0, 15.0, 30.0, 6.0, "S", 0.2])
    simple = wb.create_sheet("Simple Craters")
    simple.append(["longitude E", "latitude N", "Diameter (meters)"])
    simple.append([285.01, 15.0, 29500.0])    # same crater as the central peak
    simple.append([100.0, 40.0, 7000.0])
    simple.append([100.0, 40.0, 7000.0])      # exact duplicate row
    wb.save(path)


def test_units_dedupe_priority_and_split(tmp_path, regions, match):
    path = tmp_path / "catalog.xlsx"
    write_workbook(path)
    raw = load_workbook_table(path, ["Central Peaks", "Simple Craters"])
    assert raw["diameter_km"].round(2).tolist() == [30.0, 29.5, 7.0, 7.0]   # meters -> km

    catalog, duplicates = build_catalog(raw, match, regions)
    assert len(catalog) == 2 and len(duplicates) == 2
    peak = catalog[catalog["source"] == "Central Peaks"].iloc[0]
    assert peak["also_in"] == "Simple Craters"
    assert peak["morph_type"] == "S"
    assert peak["split"] == "T1"
    assert list(catalog["id"]) == ["RH00001", "RH00002"]
    assert set(catalog.columns) >= {"id", "lat", "lon", "diameter_km", "source"}
