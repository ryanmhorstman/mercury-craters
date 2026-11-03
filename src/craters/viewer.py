"""Draw catalog craters as circles on a reprojected tile.

    python -m craters.viewer --lat 27.77 --lon 57.08 --level coarse
    python -m craters.viewer --tile-id fine_r010_c0300 --source dem
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from .config import load_config  # noqa: E402
from .geo import lon_360  # noqa: E402
from .tiling import Tile, catalog_in_tile, render_tile, tile_from_row  # noqa: E402

SOURCE_COLORS = {
    "Simple Craters": "#4cc9f0",
    "Central Peaks": "#f9c74f",
    "Central Pits": "#f94144",
    "Peak Ring Basins": "#90be6d",
    "ProtoBasins": "#c77dff",
}


def hillshade(elevation, pixel_m, azimuth_deg=315.0, altitude_deg=45.0):
    dy, dx = np.gradient(elevation, pixel_m)
    slope = np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    az, alt = np.radians(azimuth_deg), np.radians(altitude_deg)
    shaded = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(shaded, 0, 1)


def stretch(image, low=1, high=99):
    lo, hi = np.nanpercentile(image, [low, high])
    return np.clip((image - lo) / max(hi - lo, 1e-9), 0, 1)


def draw_tile(ax, tile, image, craters, source_label, min_diameter_km=0.0):
    shown = hillshade(image, tile.pixel_m) if source_label == "dem" else stretch(image)
    ax.imshow(shown, cmap="gray", interpolation="nearest")
    craters = craters[craters["diameter_km"] >= min_diameter_km]
    for _, c in craters.iterrows():
        ax.add_patch(Circle((c["col"] - 0.5, c["row"] - 0.5), c["radius_px"], fill=False,
                            lw=0.9, color=SOURCE_COLORS.get(c["source"], "white")))
    ax.set_xlim(-0.5, tile.size_px - 0.5)
    ax.set_ylim(tile.size_px - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{source_label} | {tile.tile_id}\n{tile.lat0:.2f}N {tile.lon0:.2f}E | "
                 f"{tile.size_km:.0f} km | {len(craters)} catalog craters", fontsize=8)


OUTCOME_STYLES = {
    "TP": ("#52b788", "hit (matched catalog crater)"),
    "FN": ("#e63946", "missed catalog crater"),
    "FP": ("#f4a261", "detection not in catalog"),
}


def plot_outcomes(tile, image, catalog, detections, title, out_path):
    """Hits, misses and unmatched detections on one tile.

    catalog / detections are the labelled frames from metrics.evaluate.
    Hits are drawn with the detection's circle; misses with the catalog's.
    """
    fig, ax = plt.subplots(figsize=(7.5, 8))
    ax.imshow(stretch(image), cmap="gray", interpolation="nearest")
    layers = [
        ("TP", detections[detections["outcome"] == "TP"]),
        ("FN", catalog[catalog["outcome"] == "FN"]),
        ("FP", detections[detections["outcome"] == "FP"]),
    ]
    counts = {}
    for outcome, rows in layers:
        shown = catalog_in_tile(tile, rows) if len(rows) else rows
        counts[outcome] = len(shown)
        color = OUTCOME_STYLES[outcome][0]
        for _, c in shown.iterrows():
            ax.add_patch(Circle((c["col"] - 0.5, c["row"] - 0.5), c["radius_px"], fill=False, lw=1.3, color=color))
    ax.set_xlim(-0.5, tile.size_px - 0.5)
    ax.set_ylim(tile.size_px - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{title}\n{tile.lat0:.2f}N {tile.lon0:.2f}E | {tile.size_km:.0f} km | "
                 f"hits {counts['TP']}, missed {counts['FN']}, not in catalog {counts['FP']}", fontsize=9)
    handles = [plt.Line2D([], [], color=c, lw=2, label=label) for c, label in OUTCOME_STYLES.values()]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def legend(fig):
    handles = [plt.Line2D([], [], color=c, lw=2, label=s) for s, c in SOURCE_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=7, frameon=False)


def plot(tile, catalog, cfg, sources, out_path, min_diameter_km=0.0):
    craters = catalog_in_tile(tile, catalog)
    fig, axes = plt.subplots(1, len(sources), figsize=(6 * len(sources), 6.6), squeeze=False)
    for ax, source in zip(axes[0], sources):
        raster = cfg["paths"]["basemap"] if source == "basemap" else cfg["paths"]["dem_raw"]
        if source == "basemap" and not raster.exists():
            raster = cfg["paths"]["basemap_raw"]
        draw_tile(ax, tile, render_tile(tile, raster), craters, source, min_diameter_km)
    legend(fig)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def tile_at(lat, lon, level_name, cfg):
    level = cfg["tiling"]["levels"][level_name]
    return Tile(tile_id=f"{level_name}_at_{lat:.2f}_{lon_360(lon):.2f}", level=level_name,
                lat0=lat, lon0=float(lon_360(lon)), size_px=level["size_px"], pixel_m=level["pixel_m"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tile-id")
    parser.add_argument("--lat", type=float)
    parser.add_argument("--lon", type=float, help="east longitude")
    parser.add_argument("--level", default="fine")
    parser.add_argument("--source", choices=["basemap", "dem", "both"], default="both")
    parser.add_argument("--out")
    args = parser.parse_args()

    cfg = load_config()
    catalog = pd.read_parquet(cfg["paths"]["catalog"])
    if args.tile_id:
        tiles = pd.read_parquet(cfg["paths"]["tiles"])
        tile = tile_from_row(tiles.set_index("tile_id", drop=False).loc[args.tile_id])
    else:
        tile = tile_at(args.lat, args.lon, args.level, cfg)
    sources = ["basemap", "dem"] if args.source == "both" else [args.source]
    out = args.out or cfg["paths"]["figures"] / f"{tile.tile_id}.png"
    print(plot(tile, catalog, cfg, sources, out))


if __name__ == "__main__":
    main()
