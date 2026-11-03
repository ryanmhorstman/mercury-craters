"""Coordinate conventions and spherical geometry.

Convention used everywhere in this project: planetocentric latitude and
east-positive longitude in [0, 360), matching the thesis catalog. Rasters with
other conventions are handled when they are sampled (see tiling.py), never by
editing coordinates elsewhere.
"""
from functools import lru_cache

import numpy as np
from pyproj import CRS, Transformer

MERCURY_RADIUS_M = 2439400.0


def lon_360(lon):
    """East longitude in [0, 360)."""
    return np.mod(lon, 360.0)


def lon_180(lon):
    """East longitude in [-180, 180)."""
    return np.mod(np.asarray(lon) + 180.0, 360.0) - 180.0


def geographic_crs(radius_m=MERCURY_RADIUS_M):
    return CRS.from_proj4(f"+proj=longlat +R={radius_m} +no_defs")


def stereographic_crs(lat0, lon0, radius_m=MERCURY_RADIUS_M):
    """Stereographic projection centred on (lat0, lon0).

    Stereographic is conformal, so a circular crater stays circular in the tile
    no matter its latitude. That is the whole point of reprojecting tiles.
    """
    return CRS.from_proj4(
        f"+proj=stere +lat_0={lat0} +lon_0={lon_180(lon0)} +k=1 +x_0=0 +y_0=0 +R={radius_m} +units=m +no_defs"
    )


@lru_cache(maxsize=4096)
def local_transformers(lat0, lon0, radius_m=MERCURY_RADIUS_M):
    """(lon/lat -> local x/y, local x/y -> lon/lat) for a tile centre."""
    geo, local = geographic_crs(radius_m), stereographic_crs(lat0, lon0, radius_m)
    forward = Transformer.from_crs(geo, local, always_xy=True)
    inverse = Transformer.from_crs(local, geo, always_xy=True)
    return forward, inverse


def unit_vectors(lat, lon):
    """Points on the unit sphere, shape (n, 3)."""
    lat, lon = np.radians(lat), np.radians(lon)
    return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def great_circle_km(lat1, lon1, lat2, lon2, radius_m=MERCURY_RADIUS_M):
    a, b = unit_vectors(lat1, lon1), unit_vectors(lat2, lon2)
    cos_angle = np.clip(np.sum(a * b, axis=1), -1.0, 1.0)
    return np.arccos(cos_angle) * radius_m / 1000.0


def km_to_chord(distance_km, radius_m=MERCURY_RADIUS_M):
    """Great-circle distance -> straight-line distance between unit vectors."""
    return 2.0 * np.sin(np.asarray(distance_km) * 1000.0 / radius_m / 2.0)


def in_box(lat, lon, box):
    """box = [lon_min_E, lon_max_E, lat_min, lat_max]; lon_min may exceed lon_max to wrap through 0."""
    lon_min, lon_max, lat_min, lat_max = box
    lon = lon_360(lon)
    if lon_min <= lon_max:
        in_lon = (lon >= lon_min) & (lon < lon_max)
    else:
        in_lon = (lon >= lon_min) | (lon < lon_max)
    return in_lon & (np.asarray(lat) >= lat_min) & (np.asarray(lat) < lat_max)


def expand_box(box, deg):
    lon_min, lon_max, lat_min, lat_max = box
    return [lon_360(lon_min - deg), lon_360(lon_max + deg), lat_min - deg, lat_max + deg]
