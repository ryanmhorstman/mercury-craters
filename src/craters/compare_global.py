"""Compare the global CNN run with the full thesis catalog.

    python -m craters.phase3 global          # first: data/detections/global_<model>.parquet
    python -m craters.compare_global

Writes to outputs/global/:
  counts_by_bin.csv        craters >= 5 km per diameter bin, globally and per latitude band
  sfd_ratio.csv            cumulative N(>=D) of both, and their ratio, at root-2 diameters
  match_by_area.csv        Hungarian-matched P / R / F1 per area (test, V1, train, other, polar)
  sfd_global.png, sfd_lat_bands.png, density_maps.png, counts_by_bin.png

The catalog is the de-duplicated one with north-south corrected positions
where Phase 4 had enough matched pairs to estimate a shift
(catalog_registered_labels; identical to catalog_registered inside T1-T3).
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .config import load_config, run_name  # noqa: E402
from .geo import MERCURY_RADIUS_M  # noqa: E402
from .metrics import bin_labels, label_outcomes, score  # noqa: E402
from .regions import assign_split  # noqa: E402
from .sfd import box_area_km2, cumulative_sfd, sqrt2_diameters  # noqa: E402

# Validated categorical pair (slots 1 and 2 of the reference palette): catalog blue, detector orange.
CATALOG_COLOR, DETECTOR_COLOR = "#2a78d6", "#eb6834"
NEUTRAL = "#f0efec"
RATIO_CMAP = LinearSegmentedColormap.from_list("catalog_detector", [CATALOG_COLOR, NEUTRAL, DETECTOR_COLOR])
WHOLE_PLANET = [0.0, 360.0, -90.0, 90.001]

# Areas for match statistics, from how much the model's training and selection saw of each.
AREA_ORDER = ["T1-T3 (held out)", "V1 (checkpoint selection)", "train (labels seen)",
              "other |lat|<60 (not trained)", "polar |lat|>=60 (not trained)"]


def area_of(lat, lon, regions):
    split = assign_split(lat, lon, regions)
    area = np.where(np.isin(split, list(regions["test"])), AREA_ORDER[0],
                    np.where(split == "V1", AREA_ORDER[1],
                             np.where(split == "train", AREA_ORDER[2], AREA_ORDER[3])))
    return np.where(np.abs(np.asarray(lat)) >= regions["train"]["max_abs_lat"], AREA_ORDER[4], area)


def band_label(band):
    lo, hi = band
    fmt = lambda v: f"{abs(v):g}°{'N' if v > 0 else 'S' if v < 0 else ''}"  # noqa: E731
    return f"{fmt(lo)}–{fmt(hi)}"


def load(cfg, calibrated=False):
    """calibrated: the run after `python -m craters.calibrate` (diameter correction + per-size cutoffs)."""
    name = run_name(cfg["detectors"]["cnn"]) + ("_calibrated" if calibrated else "")
    detections = pd.read_parquet(cfg["paths"]["detections"] / f"global_{name}.parquet")
    catalog = pd.read_parquet(cfg["paths"]["catalog_registered_labels"])
    return name, catalog, detections


# ---------------------------------------------------------------- counts and SFDs

def counts_by_bin(catalog, detections, cfg):
    bins = cfg["evaluation"]["diameter_bins_km"]
    labels = bin_labels(bins)
    rows = []
    bands = [[-90, 90]] + cfg["global"]["lat_bands"]
    for band in bands:
        for who, frame in (("catalog", catalog), ("detector", detections)):
            inside = (frame["lat"] >= band[0]) & (frame["lat"] < band[1] + (band[1] == 90) * 1e-3)
            counts = pd.cut(frame.loc[inside, "diameter_km"], bins, right=False, labels=labels).value_counts()
            rows.append({"band": "global" if band == [-90, 90] else band_label(band), "source": who,
                         **{label: int(counts.get(label, 0)) for label in labels},
                         f">={bins[0]:g} km": int(counts.sum())})
    table = pd.DataFrame(rows)
    wide = table.pivot(index="band", columns="source").swaplevel(axis=1).sort_index(axis=1)
    order = ["global"] + [band_label(b) for b in cfg["global"]["lat_bands"]]
    return table, wide.loc[order]


def sfd_table(catalog, detections, cfg):
    lo, hi = cfg["evaluation"]["sfd_range_km"]
    grid = sqrt2_diameters(lo, hi)
    area = 4 * np.pi * (MERCURY_RADIUS_M / 1000.0) ** 2
    cat_n, _, cat_count = cumulative_sfd(catalog["diameter_km"], area, grid)
    det_n, _, det_count = cumulative_sfd(detections["diameter_km"], area, grid)
    return pd.DataFrame({"D_km": grid.round(2), "catalog_N": cat_count, "detector_N": det_count,
                         "catalog_per_km2": cat_n, "detector_per_km2": det_n,
                         "ratio": np.where(cat_count > 0, det_count / np.maximum(cat_count, 1), np.nan)})


def sfd_axes(ax, ratio_ax, series, area_km2, grid):
    """Cumulative SFD of each series on ax, detector/catalog ratio on ratio_ax."""
    curves = {}
    for i, (label, (diameters, color)) in enumerate(series.items()):
        density, err, counts = cumulative_sfd(diameters, area_km2, grid)
        keep = counts > 0
        x = grid[keep] * (1 + 0.025 * (i - 0.5))       # nudge apart so error bars stay readable
        ax.errorbar(x, density[keep], yerr=err[keep], color=color, marker="o", ms=3.5, lw=1.5, capsize=2,
                    label=f"{label} (n={len(diameters):,})")
        curves[label] = counts
    (cat_label, cat_counts), (det_label, det_counts) = curves.items()
    keep = (cat_counts > 0) & (det_counts > 0)
    ratio = det_counts[keep] / cat_counts[keep]
    ratio_err = ratio * np.sqrt(1 / det_counts[keep] + 1 / cat_counts[keep])
    ratio_ax.errorbar(grid[keep], ratio, yerr=ratio_err, color="#52514e", marker="o", ms=3, lw=1.2, capsize=2)
    ratio_ax.axhline(1.0, color="#52514e", lw=0.8, ls="--")
    for a in (ax, ratio_ax):
        a.set_xscale("log")
        a.grid(True, which="both", alpha=0.25)
    ax.set_yscale("log")
    ratio_ax.set_yscale("log")
    ratio_ax.set_ylabel("detector / catalog", fontsize=8)
    ratio_ax.set_xlabel("Diameter D (km)")


def plot_sfd_global(catalog, detections, cfg, name, out_path):
    lo, hi = cfg["evaluation"]["sfd_range_km"]
    grid = sqrt2_diameters(lo, hi)
    area = 4 * np.pi * (MERCURY_RADIUS_M / 1000.0) ** 2
    fig, (ax, rax) = plt.subplots(2, 1, figsize=(6.5, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    sfd_axes(ax, rax, {"thesis catalog": (catalog["diameter_km"], CATALOG_COLOR),
                       f"CNN ({name})": (detections["diameter_km"], DETECTOR_COLOR)}, area, grid)
    ax.set_ylabel("Cumulative N(≥D) per km²")
    ax.set_title(f"Mercury, whole planet, D ≥ {lo:g} km (√2 steps, 1σ Poisson)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_sfd_bands(catalog, detections, cfg, out_path):
    lo, hi = cfg["evaluation"]["sfd_range_km"]
    grid = sqrt2_diameters(lo, hi)
    bands = cfg["global"]["lat_bands"]
    fig = plt.figure(figsize=(12, 7.5))
    outer = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.28)
    for k, band in enumerate(sorted(bands, key=lambda b: -b[0])):          # north at top-left
        inner = outer[k // 3, k % 3].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
        ax, rax = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
        in_band = lambda f: f[(f["lat"] >= band[0]) & (f["lat"] < band[1] + (band[1] == 90) * 1e-3)]  # noqa: E731
        sfd_axes(ax, rax, {"catalog": (in_band(catalog)["diameter_km"], CATALOG_COLOR),
                           "CNN": (in_band(detections)["diameter_km"], DETECTOR_COLOR)},
                 box_area_km2([0, 360, *band]), grid)
        ax.tick_params(labelbottom=False)
        ax.set_title(band_label(band), fontsize=10)
        ax.legend(fontsize=7)
        rax.set_ylim(0.3, 10)
    fig.suptitle("Cumulative SFD by latitude band, N(≥D) per km²; lower panels: detector / catalog", fontsize=11)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_counts(wide, cfg, out_path):
    labels = bin_labels(cfg["evaluation"]["diameter_bins_km"])
    g = wide.loc["global"]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for offset, who, color, text in ((-0.2, "catalog", CATALOG_COLOR, "thesis catalog"),
                                     (0.2, "detector", DETECTOR_COLOR, "CNN")):
        values = [g[(who, label)] for label in labels]
        bars = ax.bar(x + offset, values, width=0.38, color=color, label=text)
        ax.bar_label(bars, labels=[f"{v:,}" for v in values], fontsize=8, padding=2, color="#0b0b0b")
    ax.set_xticks(x, [f"{label} km" for label in labels])
    ax.set_ylabel("Craters, whole planet")
    ax.set_title("Craters per diameter bin", fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- density maps

def cell_counts(frame, cell_deg):
    lon_edges = np.arange(0, 360 + cell_deg, cell_deg)
    lat_edges = np.arange(-90, 90 + cell_deg, cell_deg)
    counts, _, _ = np.histogram2d(frame["lat"], frame["lon"] % 360, bins=[lat_edges, lon_edges])
    r_km = MERCURY_RADIUS_M / 1000.0
    cell_area = r_km**2 * np.radians(cell_deg) * np.diff(np.sin(np.radians(lat_edges)))
    return counts, counts / cell_area[:, None] * 1e4, lon_edges, lat_edges   # per 10^4 km^2


def outline_regions(ax, regions):
    for name, box in {**regions["test"], **regions["validation"]}.items():
        lon_min, lon_max, lat_min, lat_max = box
        ax.add_patch(plt.Rectangle((lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
                                   fill=False, ec="#0b0b0b", lw=1.2))
        ax.text(lon_min + 1, lat_max - 4, name, fontsize=7, color="#0b0b0b", va="top")
    for box in regions["train"]["exclude"]:
        lon_min, lon_max, lat_min, lat_max = box
        ax.add_patch(plt.Rectangle((lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
                                   fill=False, ec="#52514e", lw=0.8, ls=":"))


def plot_density(catalog, detections, cfg, out_path):
    cell = cfg["global"]["density_cell_deg"]
    cat_counts, cat_density, lon_edges, lat_edges = cell_counts(catalog, cell)
    det_counts, det_density, _, _ = cell_counts(detections, cell)
    vmax = np.percentile(np.concatenate([cat_density.ravel(), det_density.ravel()]), 99)
    enough = (cat_counts >= cfg["global"]["min_cell_count"]) & (det_counts >= cfg["global"]["min_cell_count"])
    ratio = np.where(enough, np.log2(np.maximum(det_counts, 1) / np.maximum(cat_counts, 1)), np.nan)

    fig, axes = plt.subplots(3, 1, figsize=(9, 12.5))
    panels = [(cat_density, "Blues", 0, vmax, "Thesis catalog, D ≥ 5 km", "craters per 10⁴ km²"),
              (det_density, "Blues", 0, vmax, "CNN, D ≥ 5 km", "craters per 10⁴ km²"),
              (ratio, RATIO_CMAP, -3, 3, "log₂(CNN / catalog); orange = CNN finds more", "log₂ ratio")]
    for ax, (values, cmap, lo, hi, title, unit) in zip(axes, panels):
        mesh = ax.pcolormesh(lon_edges, lat_edges, values, cmap=cmap, vmin=lo, vmax=hi, shading="flat")
        outline_regions(ax, cfg["regions"])
        ax.set_xlim(0, 360)
        ax.set_ylim(-90, 90)
        ax.set_aspect("equal")
        ax.set_xticks(range(0, 361, 60))
        ax.set_yticks(range(-90, 91, 30))
        ax.set_xlabel("East longitude (°)")
        ax.set_ylabel("Latitude (°)")
        ax.set_title(title, fontsize=10)
        fig.colorbar(mesh, ax=ax, fraction=0.025, pad=0.02, label=unit)
    fig.text(0.5, 0.005, f"{cell}° cells. Solid boxes: test T1–T3 and validation V1; dotted: areas left out of "
             f"training because the catalog looked incomplete. Ratio cells need ≥ {cfg['global']['min_cell_count']} "
             "of each.", ha="center", fontsize=8, color="#52514e")
    fig.tight_layout(rect=[0, 0.015, 1, 1])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return pd.DataFrame({"lat_lo": np.repeat(lat_edges[:-1], len(lon_edges) - 1),
                         "lon_lo": np.tile(lon_edges[:-1], len(lat_edges) - 1),
                         "catalog": cat_counts.ravel().astype(int), "detector": det_counts.ravel().astype(int)})


# ---------------------------------------------------------------- matching

def match_by_area(catalog, detections, cfg):
    """Match once over the whole planet, then count TP / FN / FP per area and diameter bin."""
    bins = cfg["evaluation"]["diameter_bins_km"]
    cat, det = label_outcomes(catalog, detections, [WHOLE_PLANET], cfg["match"], bins[0])
    cat["area"] = area_of(cat["lat"], cat["lon"], cfg["regions"])
    det["area"] = area_of(det["lat"], det["lon"], cfg["regions"])
    cat["bin"] = pd.cut(cat["diameter_km"], bins, right=False, labels=bin_labels(bins))
    det["bin"] = pd.cut(det["diameter_km"], bins, right=False, labels=bin_labels(bins))
    rows = []
    for area in ["whole planet"] + AREA_ORDER:
        in_cat = cat if area == "whole planet" else cat[cat["area"] == area]
        in_det = det if area == "whole planet" else det[det["area"] == area]
        for label in ["all"] + bin_labels(bins):
            c = in_cat if label == "all" else in_cat[in_cat["bin"] == label]
            d = in_det if label == "all" else in_det[in_det["bin"] == label]
            rows.append({"area": area, "bin": label, **score(c, d)})
    return pd.DataFrame(rows)


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calibrated", action="store_true",
                        help="compare the calibrated run (python -m craters.calibrate); outputs go to outputs/global/calibrated/")
    args = parser.parse_args()
    cfg = load_config()
    out = cfg["paths"]["global"] / ("calibrated" if args.calibrated else "")
    out.mkdir(parents=True, exist_ok=True)
    name, catalog, detections = load(cfg, args.calibrated)
    min_d = cfg["evaluation"]["diameter_bins_km"][0]
    catalog = catalog[catalog["diameter_km"] >= min_d].reset_index(drop=True)
    detections = detections[detections["diameter_km"] >= min_d].reset_index(drop=True)
    print(f"catalog: {len(catalog):,} craters >= {min_d:g} km; detector {name}: {len(detections):,}")

    _, wide = counts_by_bin(catalog, detections, cfg)
    wide.to_csv(out / "counts_by_bin.csv")
    print("\ncounts by diameter bin:\n" + wide.to_string())

    sfd = sfd_table(catalog, detections, cfg)
    sfd.to_csv(out / "sfd_ratio.csv", index=False)
    print("\ncumulative counts N(>=D), whole planet:\n" + sfd[["D_km", "catalog_N", "detector_N", "ratio"]].round(2).to_string(index=False))

    matches = match_by_area(catalog, detections, cfg)
    matches.to_csv(out / "match_by_area.csv", index=False)
    cols = ["area", "bin", "n_catalog", "n_detections", "TP", "precision", "recall", "f1"]
    print("\nmatch statistics (raw, against the catalog):\n" + matches[cols].round(3).to_string(index=False))

    plot_sfd_global(catalog, detections, cfg, name, out / "sfd_global.png")
    plot_sfd_bands(catalog, detections, cfg, out / "sfd_lat_bands.png")
    plot_counts(wide, cfg, out / "counts_by_bin.png")
    plot_density(catalog, detections, cfg, out / "density_maps.png").to_csv(out / "density_cells.csv", index=False)

    if args.calibrated:
        print(f"figures in {out}")
        return
    # Same tiles and cutoff as the Phase 3 test run, so T1-T3 should reproduce it.
    test = pd.read_parquet(cfg["paths"]["detections"] / f"{cfg['detectors']['cnn']['model_name']}.parquet")
    in_test = np.isin(assign_split(detections["lat"], detections["lon"], cfg["regions"]), list(cfg["regions"]["test"]))
    test_in = np.isin(assign_split(test["lat"], test["lon"], cfg["regions"]), list(cfg["regions"]["test"]))
    check = {"global_run_T1_T3": int(in_test.sum()), "phase3_run_T1_T3": int(test_in.sum())}
    (out / "consistency.json").write_text(json.dumps(check, indent=2))
    print(f"\nT1-T3 consistency (>= {min_d:g} km, centres inside the boxes): {check}")
    print(f"figures in {out}")


if __name__ == "__main__":
    main()
