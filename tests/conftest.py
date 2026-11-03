import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from craters.geo import MERCURY_RADIUS_M, great_circle_km

REGIONS = {
    "test": {"T1": [270, 300, 0, 30]},
    "validation": {"V1": [150, 180, -30, 0]},
    "train": {"max_abs_lat": 60, "buffer_deg": 5, "exclude": [[180, 240, -90, 90]]},
}
MATCH = {"center_frac": 0.5, "ratio_min": 0.75, "ratio_max": 1.33}


@pytest.fixture
def regions():
    return REGIONS


@pytest.fixture
def match():
    return MATCH


@pytest.fixture
def disk_raster(tmp_path):
    """Write a global equirectangular raster with bright disks at given places.

    Usage: disk_raster(lon_0, [(lat, lon, diameter_km), ...]) -> path
    """
    def make(lon_0, disks, deg_per_px=0.1):
        width, height = int(360 / deg_per_px), int(180 / deg_per_px)
        px_m = np.radians(deg_per_px) * MERCURY_RADIUS_M
        lon = lon_0 - 180 + (np.arange(width) + 0.5) * deg_per_px
        lat = 90 - (np.arange(height) + 0.5) * deg_per_px
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        image = np.full((height, width), 10, dtype=np.uint8)
        for d_lat, d_lon, d_km in disks:
            near = (np.abs(lat_grid - d_lat) < d_km / 30)  # cheap prefilter (~1 deg = 42.6 km)
            dist = great_circle_km(lat_grid[near], lon_grid[near], np.full(near.sum(), d_lat), np.full(near.sum(), d_lon))
            values = image[near]
            values[dist <= d_km / 2] = 250
            image[near] = values
        path = tmp_path / f"disks_{lon_0:g}.tif"
        crs = f"+proj=eqc +lat_ts=0 +lat_0=0 +lon_0={lon_0} +x_0=0 +y_0=0 +R={MERCURY_RADIUS_M} +units=m +no_defs"
        with rasterio.open(path, "w", driver="GTiff", width=width, height=height, count=1, dtype="uint8",
                           crs=crs, transform=from_origin(-180 * px_m / deg_per_px, 90 * px_m / deg_per_px, px_m, px_m)) as dst:
            dst.write(image, 1)
        return path
    return make
