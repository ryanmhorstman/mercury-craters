"""Phase 2: score every detector against the catalog in the test regions.

    python -m craters.phase2

Reads data/detections/<detector>.parquet (lat, lon, diameter_km, score). Two
sanity runs are always included so the evaluation itself can be checked:
  sanity:catalog_self  the catalog scored against itself (must be perfect)
  sanity:noisy         a fake detector with known misses, jitter and false positives
"""
import numpy as np
import pandas as pd

from .catalog import load_eval_catalog
from .config import load_config
from .detect import load_tile_image
from .metrics import evaluate
from .sfd import box_area_km2, plot_sfd, sqrt2_diameters
from .synthetic import noisy_detections
from .tiling import tile_from_row
from .viewer import plot_outcomes

TABLE_COLUMNS = ["detector", "region", "bin", "n_catalog", "n_detections", "TP", "FP", "FN",
                 "precision", "recall", "f1", "diam_err_median", "diam_err_abs_median", "center_offset_median",
                 "precision_lo", "precision_hi", "recall_lo", "recall_hi"]


def load_detectors(cfg, catalog):
    detectors = {}
    det_dir = cfg["paths"]["detections"]
    if det_dir.exists():
        for path in sorted(det_dir.glob("*.parquet")):
            detectors[path.stem] = pd.read_parquet(path)

    test_boxes = list(cfg["regions"]["test"].values())
    detectors["sanity:catalog_self"] = catalog[["lat", "lon", "diameter_km"]].assign(score=1.0)
    rng = np.random.default_rng(cfg["seed"])
    noisy, _ = noisy_detections(catalog, test_boxes, rng, **cfg["evaluation"]["sanity"]["noisy"])
    detectors["sanity:noisy"] = noisy
    return detectors


def region_sets(cfg):
    test = cfg["regions"]["test"]
    return {"T-all": list(test.values()), **{name: [box] for name, box in test.items()}}


def markdown_table(table):
    def pct(x):
        return "" if pd.isna(x) else f"{100 * x:.0f}%"

    lines = ["| detector | region | bin | catalog | TP | FP | FN | precision | recall | F1 | median diam. err |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in table.iterrows():
        recall = f"{pct(r['recall'])} ({pct(r['recall_lo'])}–{pct(r['recall_hi'])})" if not pd.isna(r["recall"]) else ""
        derr = "" if pd.isna(r["diam_err_median"]) else f"{100 * r['diam_err_median']:+.1f}%"
        lines.append(f"| {r['detector']} | {r['region']} | {r['bin']} | {r['n_catalog']} | {r['TP']} | {r['FP']} | "
                     f"{r['FN']} | {pct(r['precision'])} | {recall} | {pct(r['f1'])} | {derr} |")
    return "\n".join(lines)


def example_tiles(cfg, name, cat, det, region, tiles, rng):
    """One fine tile per test region showing hits, misses and unmatched detections."""
    candidates = tiles[(tiles["level"] == "fine") & (tiles["split"] == region)]
    row = candidates.iloc[rng.integers(len(candidates))]
    tile = tile_from_row(row)
    safe_name = name.replace(":", "_")
    return plot_outcomes(tile, load_tile_image(tile, cfg), cat, det, f"{name} | {region} | {tile.tile_id}",
                         cfg["paths"]["phase2"] / "examples" / f"{safe_name}_{region}.png")


def main():
    cfg = load_config()
    out_dir = cfg["paths"]["phase2"]
    catalog = load_eval_catalog(cfg)
    print(f"catalog: {'registered' if 'north_shift_km' in catalog else 'original'} positions")
    tiles = pd.read_parquet(cfg["paths"]["tiles"])
    detectors = load_detectors(cfg, catalog)
    bins = cfg["evaluation"]["diameter_bins_km"]
    grid = sqrt2_diameters(*cfg["evaluation"]["sfd_range_km"])

    tables = []
    for region, boxes in region_sets(cfg).items():
        area = sum(box_area_km2(b) for b in boxes)
        sfd_series = {}
        for name, detections in detectors.items():
            table, cat, det = evaluate(catalog, detections, boxes, cfg["match"], bins)
            tables.append(table.assign(detector=name, region=region))
            if region in cfg["regions"]["test"] and not name.startswith("sanity:"):
                # Same seed per region, so every detector is shown on the same tile.
                example_tiles(cfg, name, cat, det, region, tiles, np.random.default_rng(cfg["seed"]))
            if "catalog" not in sfd_series:
                sfd_series["catalog"] = cat.loc[cat["counted"], "diameter_km"].to_numpy()
            if name != "sanity:catalog_self":
                sfd_series[name] = det.loc[det["counted"], "diameter_km"].to_numpy()
        plot_sfd(sfd_series, area, grid, f"Cumulative SFD, {region} ({area / 1e6:.2f} million km²)",
                 out_dir / f"sfd_{region}.png")

    table = pd.concat(tables, ignore_index=True)[TABLE_COLUMNS]
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "metrics.csv", index=False)
    (out_dir / "metrics.md").write_text(markdown_table(table) + "\n", encoding="utf-8")

    print(f"detectors: {', '.join(detectors)}")
    print(markdown_table(table[table["region"] == "T-all"]))
    print(f"\nwrote {out_dir / 'metrics.csv'}, metrics.md and sfd_*.png")


if __name__ == "__main__":
    main()
