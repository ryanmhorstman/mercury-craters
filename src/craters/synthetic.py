"""A fake detector with known flaws, to check that the metrics report them.

Not a crater detector: it perturbs the catalog. It misses craters with a
size-dependent probability, jitters centres and diameters, and scatters
false positives with a power-law size distribution.
"""
import numpy as np
import pandas as pd

from .geo import MERCURY_RADIUS_M


def noisy_detections(catalog, boxes, rng, *, recall_small=0.5, recall_large=0.95, half_km=10.0,
                     center_sigma_frac=0.08, diameter_sigma=0.06, fp_per_box=100, fp_min_km=5.0, fp_slope=-2.0):
    """Returns (detections, truth) where truth says which catalog rows were kept."""
    d = catalog["diameter_km"].to_numpy()
    # Detection probability rises from recall_small at small sizes to recall_large.
    p_detect = recall_small + (recall_large - recall_small) * d / (d + half_km)
    found = rng.random(len(catalog)) < p_detect

    kept = catalog[found]
    angle = rng.uniform(0, 2 * np.pi, len(kept))
    offset_km = np.abs(rng.normal(0, center_sigma_frac, len(kept))) * kept["diameter_km"].to_numpy()
    offset_deg = np.degrees(offset_km * 1000.0 / MERCURY_RADIUS_M)
    lat = kept["lat"].to_numpy() + offset_deg * np.sin(angle)
    lon = kept["lon"].to_numpy() + offset_deg * np.cos(angle) / np.cos(np.radians(lat))
    diameter = kept["diameter_km"].to_numpy() * np.exp(rng.normal(0, diameter_sigma, len(kept)))
    real = pd.DataFrame({"lat": lat, "lon": lon % 360, "diameter_km": diameter,
                         "score": rng.uniform(0.5, 1.0, len(kept)), "fake": False})

    fakes = []
    for lon_min, lon_max, lat_min, lat_max in boxes:
        n = fp_per_box
        sin_lat = rng.uniform(np.sin(np.radians(lat_min)), np.sin(np.radians(lat_max)), n)
        # Pareto sizes: N(>D) ~ D^fp_slope above fp_min_km.
        size = fp_min_km * (1 - rng.random(n)) ** (1 / fp_slope)
        fakes.append(pd.DataFrame({"lat": np.degrees(np.arcsin(sin_lat)), "lon": rng.uniform(lon_min, lon_max, n),
                                   "diameter_km": size, "score": rng.uniform(0.0, 0.7, n), "fake": True}))
    detections = pd.concat([real, *fakes], ignore_index=True)
    return detections, found
