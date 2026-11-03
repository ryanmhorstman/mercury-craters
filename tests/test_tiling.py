import numpy as np
import pytest

from craters.tiling import Tile, make_tiles, render_tile


def disk_shape(image):
    """(width, height, centre col, centre row) of the bright pixels, in pixels."""
    rows, cols = np.nonzero(image > 128)
    return cols.max() - cols.min() + 1, rows.max() - rows.min() + 1, cols.mean() + 0.5, rows.mean() + 0.5


def test_pixel_round_trip():
    tile = Tile("t", "fine", 45.0, 300.0, 1024, 166.31)
    lon, lat = tile.px_to_lonlat([0.5, 512.0, 1000.25], [3.0, 512.0, 700.0])
    col, row = tile.lonlat_to_px(lon, lat)
    assert np.allclose(col, [0.5, 512.0, 1000.25], atol=1e-6)
    assert np.allclose(row, [3.0, 512.0, 700.0], atol=1e-6)


def test_tile_centre_is_its_lat_lon():
    tile = Tile("t", "fine", -20.0, 75.0, 1024, 166.31)
    lon, lat = tile.px_to_lonlat([512.0], [512.0])
    assert np.isclose(lon[0], 75.0) and np.isclose(lat[0], -20.0)


def test_crater_at_60_degrees_is_round_after_reprojection(disk_raster):
    """In the raw raster a 200 km crater at 60N is twice as wide as tall; in the tile it is round."""
    path = disk_raster(0.0, [(60.0, 100.0, 200.0)])
    tile = Tile("t", "test", 60.0, 100.0, 128, 4000.0)
    width, height, col, row = disk_shape(render_tile(tile, path))
    assert width == pytest.approx(50, abs=3) and height == pytest.approx(50, abs=3)
    assert abs(width - height) <= 2
    assert col == pytest.approx(64, abs=1) and row == pytest.approx(64, abs=1)


@pytest.mark.parametrize("raster_lon_0, crater_lon", [(0.0, 180.0), (180.0, 0.0), (180.0, 359.0)])
def test_tiles_on_the_raster_seam_render_whole(disk_raster, raster_lon_0, crater_lon):
    path = disk_raster(raster_lon_0, [(10.0, crater_lon, 150.0)])
    tile = Tile("t", "test", 10.0, crater_lon, 128, 4000.0)
    width, height, col, row = disk_shape(render_tile(tile, path))
    assert width == pytest.approx(37.5, abs=3) and height == pytest.approx(37.5, abs=3)
    assert col == pytest.approx(64, abs=1) and row == pytest.approx(64, abs=1)


def test_coarse_tile_averages_instead_of_aliasing(disk_raster):
    path = disk_raster(0.0, [(0.0, 40.0, 300.0)], deg_per_px=0.05)    # 2.1 km source pixels
    tile = Tile("t", "test", 0.0, 40.0, 64, 17000.0)                    # 8x coarser
    image = render_tile(tile, path)
    edge_values = image[(image > 20) & (image < 240)]
    assert edge_values.size > 0          # averaged edge pixels exist; nearest-neighbour would give none


def test_every_crater_up_to_the_overlap_fits_in_some_tile(regions):
    level = {"size_px": 64, "overlap_px": 16, "pixel_m": 20000.0}     # 1280 km tiles, 320 km overlap
    tiles = make_tiles("coarse", level, 60, regions)
    rng = np.random.default_rng(0)
    n = 300
    lat = np.degrees(np.arcsin(rng.uniform(np.sin(np.radians(-55)), np.sin(np.radians(55)), n)))
    lon = rng.uniform(0, 360, n)
    diameter_km = rng.uniform(10, 0.95 * 16 * 20.0, n)

    fits = np.zeros(n, dtype=bool)
    for tile in tiles:
        col, row = tile.lonlat_to_px(lon, lat)
        r = tile.km_to_px(diameter_km / 2)
        fits |= (col - r >= 0) & (col + r <= tile.size_px) & (row - r >= 0) & (row + r <= tile.size_px)
    assert fits.all(), f"{(~fits).sum()} craters not wholly inside any tile"
