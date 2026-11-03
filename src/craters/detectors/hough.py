"""Classical baseline: contrast normalisation, Canny edges and Hough circles.

Roughly the toolbox of a 2015-era automated attempt, the "past me" benchmark.
Radii are searched in octave bands; each band runs on a copy of the image
shrunk so its smallest radius is about band_min_px pixels. That keeps the
Hough accumulator well-conditioned and the run fast.

Score = rim support: the fraction of the fitted circle that lies on (dilated)
Canny edges. Sunlit Mercury craters usually show one bright and one shadowed
rim arc, so good detections score around 0.4-0.8 rather than 1.
"""
import cv2
import numpy as np
import pandas as pd

DEFAULTS = {
    "method": "gradient",       # "gradient" (classic) or "alt" (OpenCV's more selective variant)
    "canny_high": 120,          # Canny high threshold; low = high / 2
    "accumulator": 18,          # classic: vote threshold at the working scale; alt: circle "perfectness" 0-1
    "clahe_clip": 2.0,          # local contrast normalisation; evens out mosaic seams and lighting
    "blur_sigma": 1.2,          # Gaussian blur at the working scale, in pixels
    "min_dist_frac": 0.75,      # minimum centre spacing as a fraction of the band's smallest radius
    "band_min_px": 10,          # smallest radius of each band at the working scale
}


def radius_bands(r_min, r_max):
    bands, lo = [], r_min
    while lo < r_max:
        hi = min(lo * 2, r_max)
        bands.append((lo, hi))
        lo = hi
    return bands


def rim_support(edges, x, y, r):
    """Fraction of points on the circle that land on an edge pixel."""
    n = max(24, int(2 * np.pi * r))
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xs = np.clip(np.round(x + r * np.cos(theta)).astype(int), 0, edges.shape[1] - 1)
    ys = np.clip(np.round(y + r * np.sin(theta)).astype(int), 0, edges.shape[0] - 1)
    return float(edges[ys, xs].mean())


def detect(image, tile, level, params=None):
    p = {**DEFAULTS, **(params or {})}
    r_min = level["detect_km"][0] * 1000.0 / 2 / tile.pixel_m
    r_max = level["detect_km"][1] * 1000.0 / 2 / tile.pixel_m
    r_max = min(r_max, tile.size_px / 2)

    clahe = cv2.createCLAHE(clipLimit=p["clahe_clip"], tileGridSize=(8, 8))
    normalised = clahe.apply(image)
    method = cv2.HOUGH_GRADIENT_ALT if p["method"] == "alt" else cv2.HOUGH_GRADIENT

    rows = []
    for band_lo, band_hi in radius_bands(r_min, r_max):
        f = max(1.0, band_lo / p["band_min_px"])
        size = max(16, int(round(image.shape[0] / f)))
        f = image.shape[0] / size
        small = cv2.resize(normalised, (size, size), interpolation=cv2.INTER_AREA) if f > 1 else normalised
        small = cv2.GaussianBlur(small, (0, 0), p["blur_sigma"])
        found = cv2.HoughCircles(
            small, method, dp=1, minDist=max(2.0, band_lo / f * p["min_dist_frac"]),
            param1=p["canny_high"], param2=p["accumulator"],
            minRadius=int(np.floor(band_lo / f)), maxRadius=int(np.ceil(band_hi / f)),
        )
        if found is None:
            continue
        edges = cv2.dilate(cv2.Canny(small, p["canny_high"] / 2, p["canny_high"]), np.ones((3, 3), np.uint8)) > 0
        for x, y, r in found[0, :, :3]:
            # OpenCV puts pixel centres at integers; this project puts them at +0.5.
            rows.append({"col": (x + 0.5) * f, "row": (y + 0.5) * f, "radius_px": r * f,
                         "score": rim_support(edges, x, y, r)})
    return pd.DataFrame(rows, columns=["col", "row", "radius_px", "score"])
