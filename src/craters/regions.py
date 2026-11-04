"""Assign points to test / validation / train / excluded areas."""
import numpy as np

from .geo import expand_box, in_box, lon_360


def assign_split(lat, lon, regions):
    """Label each point with a test or validation region name, "train", "excluded" or,
    when regions.polar is set, "polar_train" or a polar test box name.

    Training points stay at least buffer_deg away from every test and
    validation box, so overlapping tiles can't leak test pixels into training.
    """
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    split = np.full(lat.shape, "excluded", dtype=object)

    train = regions["train"]
    is_train = np.abs(lat) < train["max_abs_lat"]
    for box in train["exclude"]:
        is_train &= ~in_box(lat, lon, box)
    held_out = {**regions["test"], **regions["validation"]}
    for box in held_out.values():
        is_train &= ~in_box(lat, lon, expand_box(box, train["buffer_deg"]))
    split[is_train] = "train"

    # Optional polar training area (regions.polar): poleward of max_abs_lat, outside the
    # same exclusions and buffers, and outside its own held-out polar test boxes.
    polar = regions.get("polar")
    if polar:
        is_polar = np.abs(lat) >= train["max_abs_lat"]
        for box in train["exclude"]:
            is_polar &= ~in_box(lat, lon, box)
        for box in held_out.values():
            is_polar &= ~in_box(lat, lon, expand_box(box, train["buffer_deg"]))
        for box in polar["test"].values():
            is_polar &= ~in_box(lat, lon, expand_box_km(box, train["buffer_deg"]))
        split[is_polar] = "polar_train"
        for name, box in polar["test"].items():
            split[in_box(lat, lon, box)] = name

    for name, box in held_out.items():
        split[in_box(lat, lon, box)] = name
    return split


def expand_box_km(box, deg):
    """Like expand_box, but the longitude margin is widened by 1/cos(latitude) at the box's
    equatorward edge, so a polar box keeps a buffer of `deg` degrees of arc on every side."""
    lon_min, lon_max, lat_min, lat_max = box
    edge = min(abs(lat_min), abs(lat_max))
    lon_deg = min(deg / np.cos(np.radians(edge)), 90.0)
    return [lon_360(lon_min - lon_deg), lon_360(lon_max + lon_deg), lat_min - deg, lat_max + deg]
