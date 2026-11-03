"""Phase 1: clean catalog, tiled basemap, tile index, and alignment previews.

    python -m craters.phase1
"""
import numpy as np
import pandas as pd

from . import catalog as catalog_step
from .config import load_config
from .rasters import build_cog
from .tiling import make_tiles, tile_from_row, tiles_frame
from .viewer import plot, tile_at


def build_tiles(cfg):
    frames = []
    for name, level in cfg["tiling"]["levels"].items():
        tiles = make_tiles(name, level, cfg["tiling"]["lat_limit"], cfg["regions"])
        frames.append(tiles_frame(tiles))
    tiles = pd.concat(frames, ignore_index=True)
    tiles.to_parquet(cfg["paths"]["tiles"], index=False)
    print("\ntiles by level and split:")
    print(pd.crosstab(tiles["level"], tiles["split"], margins=True).to_string())
    return tiles


def previews(cfg, catalog, tiles):
    rng = np.random.default_rng(cfg["seed"])
    out_dir = cfg["paths"]["figures"]
    written = []
    for name, (lon, lat) in cfg["viewer"]["alignment_craters"].items():
        tile = tile_at(lat, lon, "coarse", cfg)
        written.append(plot(tile, catalog, cfg, ["basemap", "dem"], out_dir / f"align_{name}.png"))
    for region in list(cfg["regions"]["test"]) + list(cfg["regions"]["validation"]):
        fine = tiles[(tiles["level"] == "fine") & (tiles["split"] == region)]
        for _, row in fine.iloc[rng.choice(len(fine), cfg["viewer"]["samples_per_region"], replace=False)].iterrows():
            written.append(plot(tile_from_row(row), catalog, cfg, ["basemap", "dem"],
                                out_dir / f"sample_{region}_{row['tile_id']}.png"))
    print("\npreviews:")
    for path in written:
        print(" ", path)


def main():
    cfg = load_config()
    paths = cfg["paths"]
    catalog = catalog_step.main(cfg)
    if not paths["basemap"].exists():
        print(f"\nbuilding {paths['basemap']} (one-off, several minutes)")
        build_cog(paths["basemap_raw"], paths["basemap"])
    tiles = build_tiles(cfg)
    previews(cfg, catalog, tiles)


if __name__ == "__main__":
    main()
