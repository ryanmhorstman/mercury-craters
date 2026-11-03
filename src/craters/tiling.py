"""Overlapping tiles, each reprojected to a stereographic grid centred on the tile.

Raw equirectangular pixels stretch east-west by 1/cos(lat), so a crater at
60 deg latitude is twice as wide as it is tall. Each tile is resampled onto a
square grid in a stereographic projection centred on the tile, where craters
are circles at every latitude. Detections in tile pixels are converted back to
lat/lon with the same projection.
"""
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.windows import Window
from scipy.ndimage import map_coordinates

from .geo import MERCURY_RADIUS_M, geographic_crs, local_transformers, lon_360
from .regions import assign_split


@dataclass(frozen=True)
class Tile:
    tile_id: str
    level: str
    lat0: float
    lon0: float          # east, 0-360
    size_px: int
    pixel_m: float
    split: str = ""

    @property
    def half_m(self):
        return self.size_px * self.pixel_m / 2.0

    @property
    def size_km(self):
        return self.size_px * self.pixel_m / 1000.0

    def lonlat_to_px(self, lon, lat):
        """Lon/lat -> (col, row) in tile pixels; pixel centres are at +0.5."""
        forward, _ = local_transformers(self.lat0, self.lon0)
        x, y = forward.transform(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
        return (np.asarray(x) + self.half_m) / self.pixel_m, (self.half_m - np.asarray(y)) / self.pixel_m

    def px_to_lonlat(self, col, row):
        _, inverse = local_transformers(self.lat0, self.lon0)
        x = np.asarray(col, dtype=float) * self.pixel_m - self.half_m
        y = self.half_m - np.asarray(row, dtype=float) * self.pixel_m
        lon, lat = inverse.transform(x, y)
        return lon_360(lon), np.asarray(lat)

    def km_to_px(self, km):
        # Stereographic scale is within 0.1% of 1 across a tile this small, so
        # diameters convert with the nominal pixel size.
        return np.asarray(km) * 1000.0 / self.pixel_m


def make_tiles(level_name, level, lat_limit, regions, radius_m=MERCURY_RADIUS_M):
    """Cover |lat| <= lat_limit with overlapping tiles for one level.

    Tile centres sit on a grid with spacing step = size - overlap (in km). Each
    tile reaches overlap/2 past its grid cell on every side, so any crater with
    diameter <= overlap lies wholly inside the tile whose cell holds its centre.
    East-west spacing in a row is set at the row's equatorward edge, where a
    degree of longitude is widest, so cells never exceed the step in width.
    """
    step_km = (level["size_px"] - level["overlap_px"]) * level["pixel_m"] / 1000.0
    step_deg = np.degrees(step_km * 1000.0 / radius_m)
    n_rows = int(np.ceil(2 * lat_limit / step_deg))
    tiles = []
    for r in range(n_rows):
        lat0 = -lat_limit + (r + 0.5) * step_deg
        equatorward = max(abs(lat0) - step_deg / 2, 0.0)
        lon_step = step_deg / np.cos(np.radians(equatorward))
        n_cols = int(np.ceil(360.0 / lon_step))
        lon_step = 360.0 / n_cols                       # spread evenly round the planet
        for c in range(n_cols):
            tiles.append(Tile(
                tile_id=f"{level_name}_r{r:03d}_c{c:04d}", level=level_name,
                lat0=round(float(lat0), 6), lon0=round(float((c + 0.5) * lon_step), 6),
                size_px=level["size_px"], pixel_m=level["pixel_m"],
            ))
    splits = assign_split([t.lat0 for t in tiles], [t.lon0 for t in tiles], regions)
    return [Tile(**{**asdict(t), "split": s}) for t, s in zip(tiles, splits)]


def tiles_frame(tiles):
    return pd.DataFrame([asdict(t) for t in tiles])


def tile_from_row(row):
    return Tile(**{k: row[k] for k in Tile.__dataclass_fields__})


def catalog_in_tile(tile, catalog, margin_px=0.0):
    """Catalog rows whose centres fall inside the tile, with tile pixel coords."""
    # Cheap prefilter on angular distance before projecting.
    reach_deg = np.degrees(tile.half_m * 1.5 / MERCURY_RADIUS_M) + 1.0
    lat_ok = np.abs(catalog["lat"] - tile.lat0) <= reach_deg
    dlon = np.abs((catalog["lon"] - tile.lon0 + 180.0) % 360.0 - 180.0)
    lon_ok = dlon <= reach_deg / max(np.cos(np.radians(min(abs(tile.lat0) + reach_deg, 89.0))), 1e-3)
    near = catalog[lat_ok & lon_ok].copy()
    col, row = tile.lonlat_to_px(near["lon"].to_numpy(), near["lat"].to_numpy())
    near["col"], near["row"] = col, row
    near["radius_px"] = tile.km_to_px(near["diameter_km"] / 2.0)
    inside = (col >= -margin_px) & (col < tile.size_px + margin_px) & (row >= -margin_px) & (row < tile.size_px + margin_px)
    return near[inside]


def render_tile(tile, raster_path):
    """Resample a global equirectangular raster onto the tile's grid.

    Returns float32 (size_px, size_px), NaN where the source has no data.
    Works for rasters centred on 0 or 180 deg longitude, and for tiles that
    straddle the raster's left/right seam.
    """
    cols, rows = np.meshgrid(np.arange(tile.size_px) + 0.5, np.arange(tile.size_px) + 0.5)
    lon, lat = tile.px_to_lonlat(cols.ravel(), rows.ravel())

    with rasterio.open(raster_path) as src:
        to_src = Transformer.from_crs(geographic_crs(), src.crs, always_xy=True)
        x, y = to_src.transform(lon, lat)
        src_col, src_row = ~src.transform @ (np.asarray(x), np.asarray(y))
        width = src.width

        # Unwrap columns around the tile centre so a seam-straddling tile reads
        # one contiguous (wrapped) window instead of the whole planet.
        centre_col = np.median(src_col)
        src_col = centre_col + (src_col - centre_col + width / 2.0) % width - width / 2.0

        # Average down first when the tile is coarser than the source, so the
        # coarse level isn't aliased.
        # Window edges are multiples of the factor so averaged blocks line up.
        factor = max(1, int(tile.pixel_m / abs(src.res[0])))
        pad = 2 * factor
        c0 = snap_down(np.floor(src_col.min()) - pad, factor)
        c1 = snap_up(np.ceil(src_col.max()) + pad, factor)
        r0 = max(snap_down(np.floor(src_row.min()) - pad, factor), 0)
        r1 = min(snap_up(np.ceil(src_row.max()) + pad, factor), snap_down(src.height, factor))
        window = read_wrapped(src, c0, c1, r0, r1, factor)

        # Source coords are continuous (pixel i spans [i, i+1)); array index k
        # is the centre of block k, at c0 + (k + 0.5) * factor.
        sample = map_coordinates(
            window, [(src_row - r0) / factor - 0.5, (src_col - c0) / factor - 0.5],
            order=1, mode="nearest",
        )
    return sample.reshape(tile.size_px, tile.size_px).astype(np.float32)


def read_wrapped(src, c0, c1, r0, r1, factor=1):
    """Read columns [c0, c1) x rows [r0, r1), wrapping columns round the planet.

    Nodata becomes NaN and the band's scale/offset is applied (the DEM stores
    half-metres). With factor > 1 the window is block-averaged on read.
    """
    width = src.width
    pieces = []
    start = c0
    while start < c1:
        wrapped = start % width
        length = min(c1 - start, width - wrapped)
        out_w = length // factor
        out_h = (r1 - r0) // factor
        data = src.read(
            1, window=Window(wrapped, r0, length, r1 - r0), out_shape=(out_h, out_w),
            masked=True, resampling=Resampling.average if factor > 1 else Resampling.nearest,
        )
        pieces.append(data.astype(np.float32).filled(np.nan))
        start += length
    window = np.concatenate(pieces, axis=1)
    return window * src.scales[0] + src.offsets[0]


def snap_down(value, factor):
    return int(value // factor * factor)


def snap_up(value, factor):
    return int(-(-value // factor) * factor)
