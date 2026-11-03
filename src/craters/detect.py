"""Run any tile-level detector over regions and return lat/lon detections.

A detector is a function  detect(image, tile, level_cfg) -> DataFrame with
columns col, row, radius_px, score  (tile pixel coordinates, pixel centres at +0.5).
This module handles everything around it: choosing tiles, rendering and
caching them, dropping circles cut by the tile edge, converting to lat/lon,
and merging repeats across overlapping tiles and between levels.
"""
import numpy as np
import pandas as pd

from .dedupe import merge_overlap_detections
from .geo import MERCURY_RADIUS_M, in_box
from .tiling import render_tile, tile_from_row


def tiles_covering(tiles, boxes, level_name):
    """Tiles of one level that overlap any of the boxes (not just those centred inside)."""
    level_tiles = tiles[tiles["level"] == level_name]
    half_km = level_tiles["size_px"].iloc[0] * level_tiles["pixel_m"].iloc[0] / 2000.0
    margin_lat = np.degrees(half_km * 1000.0 * np.sqrt(2) / MERCURY_RADIUS_M)
    keep = np.zeros(len(level_tiles), dtype=bool)
    for lon_min, lon_max, lat_min, lat_max in boxes:
        widest = np.cos(np.radians(min(max(abs(lat_min), abs(lat_max)) + margin_lat, 89.0)))
        margin_lon = margin_lat / widest
        grown = [(lon_min - margin_lon) % 360, (lon_max + margin_lon) % 360, lat_min - margin_lat, lat_max + margin_lat]
        keep |= in_box(level_tiles["lat0"].to_numpy(), level_tiles["lon0"].to_numpy(), grown)
    return level_tiles[keep]


def load_tile_image(tile, cfg):
    """Basemap tile as uint8, cached under data/interim/tiles/<level>/."""
    cache = cfg["paths"]["basemap"].parent / "tiles" / tile.level / f"{tile.tile_id}.npy"
    if cache.exists():
        return np.load(cache)
    image = render_tile(tile, cfg["paths"]["basemap"])
    image = np.clip(np.nan_to_num(image, nan=0.0), 0, 255).round().astype(np.uint8)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, image)
    return image


def to_lonlat(tile, circles, level_cfg):
    """Tile-pixel circles -> lat/lon rows, keeping whole circles in the level's size range."""
    if circles.empty:
        return pd.DataFrame(columns=["lat", "lon", "diameter_km", "score", "tile_id", "level"])
    r = circles["radius_px"].to_numpy()
    col, row = circles["col"].to_numpy(), circles["row"].to_numpy()
    whole = (col - r >= 0) & (col + r <= tile.size_px) & (row - r >= 0) & (row + r <= tile.size_px)
    diameter_km = 2 * r * tile.pixel_m / 1000.0
    lo, hi = level_cfg["detect_km"]
    keep = whole & (diameter_km >= lo) & (diameter_km <= hi)
    lon, lat = tile.px_to_lonlat(col[keep], row[keep])
    return pd.DataFrame({"lat": lat, "lon": lon, "diameter_km": diameter_km[keep],
                         "score": circles["score"].to_numpy()[keep], "tile_id": tile.tile_id, "level": tile.level})


def run_detector(detect_fn, tiles, boxes, cfg, levels=None, images=None, progress=True):
    """Detect over every tile touching the boxes, then merge repeats.

    images: optional dict tile_id -> uint8 array, to skip disk reads when the
    same tiles are scored many times (parameter tuning).
    """
    chosen = pd.concat([tiles_covering(tiles, boxes, name) for name in cfg["tiling"]["levels"]
                        if not levels or name in levels])
    return detect_on_tiles(detect_fn, chosen, cfg, images, progress)


def detect_on_tiles(detect_fn, tiles, cfg, images=None, progress=True):
    """Detect over exactly the given tiles (any mix of levels), then merge repeats."""
    found = []
    for level_name, level_cfg in cfg["tiling"]["levels"].items():
        chosen = tiles[tiles["level"] == level_name]
        for i, (_, row) in enumerate(chosen.iterrows()):
            tile = tile_from_row(row)
            image = images[tile.tile_id] if images is not None and tile.tile_id in images else load_tile_image(tile, cfg)
            found.append(to_lonlat(tile, detect_fn(image, tile, level_cfg), level_cfg))
            if progress and (i + 1) % 50 == 0:
                print(f"  {level_name}: {i + 1}/{len(chosen)} tiles", flush=True)
    found = [f for f in found if not f.empty]
    if not found:
        return pd.DataFrame(columns=["lat", "lon", "diameter_km", "score", "tile_id", "level"])
    detections = pd.concat(found, ignore_index=True)
    return merge_overlap_detections(detections, cfg["match"])
