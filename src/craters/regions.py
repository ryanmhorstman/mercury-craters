"""Assign points to test / validation / train / excluded areas."""
import numpy as np

from .geo import expand_box, in_box


def assign_split(lat, lon, regions):
    """Label each point with a test or validation region name, "train" or "excluded".

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

    for name, box in held_out.items():
        split[in_box(lat, lon, box)] = name
    return split
