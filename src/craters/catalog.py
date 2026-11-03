"""Turn the thesis workbook into one clean crater table.

Output columns: id, lat, lon, diameter_km, source, plus provenance
(sheet_row, also_in, morph_type, name) and the evaluation split.
The workbook is opened read-only and never written.
"""
import numpy as np
import pandas as pd
from openpyxl import load_workbook

from .dedupe import dedupe_circles
from .geo import lon_360
from .regions import assign_split


def read_sheet(workbook, sheet_name):
    """One sheet -> DataFrame with lat, lon, diameter_km and any type/name columns."""
    rows = list(workbook[sheet_name].iter_rows(values_only=True))
    header = [str(h).strip().lower() if h is not None else "" for h in rows[0]]

    def column(*names):
        return next((i for i, h in enumerate(header) if h in names), None)

    lon_col = column("longitude e")
    lat_col = column("latitude n")
    d_col = next(i for i, h in enumerate(header) if h.startswith("diameter"))
    to_km = 0.001 if "(meters)" in header[d_col] else 1.0
    type_col, name_col = column("type"), column("name")

    records = []
    for row_number, row in enumerate(rows[1:], start=2):  # Excel row numbers
        if row is None or row[lat_col] is None:
            continue
        records.append({
            "lat": float(row[lat_col]),
            "lon": float(lon_360(float(row[lon_col]))),
            "diameter_km": float(row[d_col]) * to_km,
            "source": sheet_name,
            "sheet_row": row_number,
            "morph_type": row[type_col] if type_col is not None and type_col < len(row) else None,
            "name": row[name_col] if name_col is not None and name_col < len(row) else None,
        })
    return pd.DataFrame.from_records(records)


def load_workbook_table(xlsx_path, sheets):
    workbook = load_workbook(xlsx_path, read_only=True, data_only=True)
    try:
        return pd.concat([read_sheet(workbook, name) for name in sheets], ignore_index=True)
    finally:
        workbook.close()


def build_catalog(raw, match, regions):
    """De-duplicate and label. Returns (catalog, duplicates).

    Entries are visited in sheet priority order (raw is already in that order),
    so a crater listed as both a central peak and in the general list keeps its
    central-peak record. The absorbed entry's sheet is kept in `also_in`.
    """
    duplicate_of = dedupe_circles(
        raw["lat"], raw["lon"], raw["diameter_km"], order=np.arange(len(raw)),
        center_frac=match["center_frac"], ratio_min=match["ratio_min"], ratio_max=match["ratio_max"],
    )
    raw = raw.assign(duplicate_of=duplicate_of)

    kept = raw[raw["duplicate_of"] < 0].copy()
    dups = raw[raw["duplicate_of"] >= 0].copy()
    also_in = dups.groupby("duplicate_of")["source"].agg(lambda s: ";".join(sorted(set(s))))
    kept["also_in"] = kept.index.map(also_in).fillna("")

    kept.insert(0, "id", [f"RH{i:05d}" for i in range(1, len(kept) + 1)])
    kept["split"] = assign_split(kept["lat"], kept["lon"], regions)

    dups["kept_id"] = kept.loc[dups["duplicate_of"], "id"].to_numpy()
    dups["kept_source"] = kept.loc[dups["duplicate_of"], "source"].to_numpy()
    kept = kept.drop(columns="duplicate_of").reset_index(drop=True)
    dups = dups.drop(columns="duplicate_of").reset_index(drop=True)
    return kept, dups


def summarize(catalog, duplicates, bins):
    labels = [f"{lo:g}-{hi:g}" if np.isfinite(hi) else f"{lo:g}+" for lo, hi in zip(bins[:-1], bins[1:])]
    binned = pd.cut(catalog["diameter_km"], bins, right=False, labels=labels)
    lines = [
        f"craters kept: {len(catalog)}  (duplicates merged: {len(duplicates)})",
        "duplicates by sheet pair:",
        duplicates.groupby(["kept_source", "source"]).size().to_string(),
        "",
        f"craters >= {bins[0]:g} km by split and diameter bin:",
        pd.crosstab(catalog["split"], binned, margins=True).to_string(),
    ]
    return "\n".join(lines)


def load_eval_catalog(cfg):
    """The catalog to score against: the registered copy when enabled and built, else the original."""
    registered = cfg["paths"]["catalog_registered"]
    if cfg["evaluation"].get("use_registered_catalog") and registered.exists():
        return pd.read_parquet(registered)
    return pd.read_parquet(cfg["paths"]["catalog"])


def main(cfg):
    paths = cfg["paths"]
    raw = load_workbook_table(paths["catalog_xlsx"], cfg["catalog"]["sheets"])
    catalog, duplicates = build_catalog(raw, cfg["match"], cfg["regions"])
    paths["catalog"].parent.mkdir(parents=True, exist_ok=True)
    catalog.to_parquet(paths["catalog"], index=False)
    duplicates.to_csv(paths["catalog_duplicates"], index=False)
    print(f"read {len(raw)} rows from {paths['catalog_xlsx'].name}")
    print(summarize(catalog, duplicates, cfg["evaluation"]["diameter_bins_km"]))
    return catalog


if __name__ == "__main__":
    from .config import load_config

    main(load_config())
