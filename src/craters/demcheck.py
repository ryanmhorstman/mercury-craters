"""Does the elevation model show a bowl where a crater is drawn?

The thesis catalog and the CNN both work from the image mosaic alone. The USGS
665 m DEM is an independent check: a real crater is a depression with a rim
standing above its floor all the way round. For each circle (lat, lon, D) the
DEM is sampled on a polar grid around the centre and three numbers are kept:

  depth_ratio   (median rim height - floor) / D, where the rim height on each
                azimuth is the highest point between 0.8 and 1.2 radii
  rim_frac      share of azimuths whose rim stands >= rim_min_frac x D above the floor
  floor_drop    (median ground between 1.6 and 2 radii - floor) / D

The pass rule (minimum depth_ratio and rim_frac, floor_drop > 0) is chosen on
the train area for each band of diameter: as many catalog craters as possible
pass while at most max_decoy_pass of "decoy" circles do. A decoy is a catalog crater's circle moved
2-3 diameters away, onto ground the catalog says holds no crater of that size.
V1 is reported with the chosen rule; T1-T3 are never read.

    python -m craters.demcheck          # fit and report; writes outputs/selftrain/dem_check.json

Below ~20 km the DEM can't tell catalog craters from decoys at all (its
effective resolution is much coarser than its 665 m pixels), so the bands start at 20 km.
"""
import json
import warnings

import numpy as np
import pandas as pd
import rasterio

from .geo import MERCURY_RADIUS_M, great_circle_km, lon_360
from .regions import assign_split


class Dem:
    """The whole DEM held in memory as int16 (~0.5 GB), sampled bilinearly in lat/lon."""

    def __init__(self, path):
        with rasterio.open(path) as src:
            self.data = src.read(1)
            self.nodata, self.scale, self.offset = src.nodata, src.scales[0], src.offsets[0]
        self.height, self.width = self.data.shape

    def sample(self, lat, lon):
        """Elevation (m) at each point; NaN where any neighbouring pixel has no data.

        The raster is equirectangular with columns starting at 0 deg E and rows at 90 N.
        """
        col = lon_360(lon) / 360.0 * self.width - 0.5
        row = np.clip((90.0 - np.asarray(lat)) / 180.0 * self.height - 0.5, 0, self.height - 1.001)
        c0, r0 = np.floor(col).astype(int), np.floor(row).astype(int)
        fc, fr = col - c0, row - r0
        out = 0.0
        bad = np.zeros(np.shape(col), dtype=bool)
        for dr, dc, w in ((0, 0, (1 - fr) * (1 - fc)), (0, 1, (1 - fr) * fc), (1, 0, fr * (1 - fc)), (1, 1, fr * fc)):
            v = self.data[r0 + dr, (c0 + dc) % self.width]
            bad |= v == self.nodata
            out = out + w * v.astype(np.float32)
        out = out * self.scale + self.offset
        return np.where(bad, np.nan, out)


def polar_grid(lat, lon, radius_km, fractions, azimuths):
    """Lat/lon of points at radius fraction x azimuth around each centre, shape (..., n_frac, n_az).

    lat, lon and radius_km may be scalars (one centre) or arrays of centres."""
    lat1, lon1 = np.radians(lat)[..., None, None], np.radians(lon)[..., None, None]
    radius = np.asarray(radius_km, dtype=float)[..., None, None]
    delta = np.outer(fractions, np.ones_like(azimuths)) * radius * 1000.0 / MERCURY_RADIUS_M
    theta = np.outer(np.ones_like(fractions), azimuths)
    lat2 = np.arcsin(np.sin(lat1) * np.cos(delta) + np.cos(lat1) * np.sin(delta) * np.cos(theta))
    lon2 = lon1 + np.arctan2(np.sin(theta) * np.sin(delta) * np.cos(lat1), np.cos(delta) - np.sin(lat1) * np.sin(lat2))
    return np.degrees(lat2), np.degrees(lon2)


def bowl_features(dem, lat, lon, diameter_km, params, chunk=2000):
    """depth_ratio, rim_frac and floor_drop for each circle (NaN where the DEM has gaps).

    params.radial_step (default 0.05 radii) and params.azimuths set the sampling grid."""
    fractions = np.arange(0.0, 2.0001, params.get("radial_step", 0.05))
    azimuths = np.linspace(0, 2 * np.pi, params["azimuths"], endpoint=False)
    floor_zone, rim_zone, out_zone = fractions <= 0.4, (fractions >= 0.8) & (fractions <= 1.2), fractions >= 1.6
    lat, lon, diameter_km = (np.asarray(v, dtype=float) for v in (lat, lon, diameter_km))
    out = np.full((len(lat), 3), np.nan)
    for s in range(0, len(lat), chunk):
        part = slice(s, s + chunk)
        z = dem.sample(*polar_grid(lat[part], lon[part], diameter_km[part] / 2.0, fractions, azimuths))
        d_m = diameter_km[part, None] * 1000.0
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)        # all-NaN slices, dropped below anyway
            floor = np.nanmedian(z[:, floor_zone].reshape(len(z), -1), axis=1)[:, None]
            rim = np.nanmax(z[:, rim_zone], axis=1)                # per azimuth
            outside = np.nanmedian(z[:, out_zone].reshape(len(z), -1), axis=1)[:, None]
            features = np.column_stack([(np.median(rim, axis=1)[:, None] - floor)[:, 0] / d_m[:, 0],
                                        np.mean(rim - floor >= params["rim_min_frac"] * d_m, axis=1),
                                        (outside - floor)[:, 0] / d_m[:, 0]])
        features[np.isnan(z).reshape(len(z), -1).mean(axis=1) > 0.05] = np.nan
        out[part] = features
    return pd.DataFrame(out, columns=["depth_ratio", "rim_frac", "floor_drop"])


def passes(features, rule):
    return ((features["depth_ratio"] >= rule["min_depth_ratio"]) & (features["rim_frac"] >= rule["min_rim_frac"])
            & (features["floor_drop"] > 0)).to_numpy()


def passes_banded(features, diameter_km, rules):
    """passes() with each circle judged by the rule of its diameter band; False outside every band."""
    diameter_km = np.asarray(diameter_km)
    ok = np.zeros(len(features), dtype=bool)
    for r in rules:
        band = (diameter_km >= r["min_km"]) & (diameter_km < r["max_km"])
        ok[band] = passes(features[band], r)
    return ok


def make_decoys(catalog, rng):
    """Each crater's circle moved 2-3 diameters at a random bearing, kept only where no catalog
    crater of 0.5-2x its size lies within one diameter."""
    n = len(catalog)
    bearing = rng.uniform(0, 2 * np.pi, n)
    shift_km = rng.uniform(2.0, 3.0, n) * catalog["diameter_km"].to_numpy()
    lat, lon = [], []
    for la, lo, s, b in zip(catalog["lat"], catalog["lon"], shift_km, bearing):
        g_lat, g_lon = polar_grid(la, lo, s, np.array([1.0]), np.array([b]))
        lat.append(g_lat[0, 0])
        lon.append(g_lon[0, 0])
    decoys = pd.DataFrame({"lat": lat, "lon": lon_360(np.array(lon)), "diameter_km": catalog["diameter_km"].to_numpy()})
    km_per_deg = np.radians(1.0) * MERCURY_RADIUS_M / 1000.0
    keep = np.ones(n, dtype=bool)
    for i, (la, lo, d) in enumerate(decoys[["lat", "lon", "diameter_km"]].to_numpy()):
        near = catalog[(np.abs(catalog["lat"] - la) < d / km_per_deg + 0.5)
                       & catalog["diameter_km"].between(0.5 * d, 2.0 * d)]
        if len(near) and great_circle_km(np.full(len(near), la), np.full(len(near), lo), near["lat"], near["lon"]).min() < d:
            keep[i] = False
    return decoys[keep].reset_index(drop=True)


def choose_rule(real, decoy, params):
    """The (min_depth_ratio, min_rim_frac) passing most real craters with decoy pass rate <= max_decoy_pass."""
    best = None
    for depth in params["depth_ratio_grid"]:
        for rim in params["rim_frac_grid"]:
            rule = {"min_depth_ratio": depth, "min_rim_frac": rim}
            real_pass, decoy_pass = passes(real, rule).mean(), passes(decoy, rule).mean()
            if decoy_pass <= params["max_decoy_pass"] and (best is None or real_pass > best["real_pass"]):
                best = {**rule, "real_pass": round(float(real_pass), 4), "decoy_pass": round(float(decoy_pass), 4)}
    return best


def fit(cfg):
    """Fit one pass rule per diameter band on the train area, and report them on V1."""
    params = cfg["selftrain"]["dem_check"]
    rng = np.random.default_rng(cfg["seed"])
    dem = Dem(cfg["paths"]["dem_raw"])
    edges = params["bands_km"]
    catalog = pd.read_parquet(cfg["paths"]["catalog_registered_labels"])
    catalog = catalog[catalog["diameter_km"].between(edges[0], edges[-1], inclusive="left")].reset_index(drop=True)
    catalog["split"] = assign_split(catalog["lat"], catalog["lon"], cfg["regions"])

    features = {}
    for split in ("train", "V1"):
        real = catalog[catalog["split"] == split].reset_index(drop=True)
        decoys = make_decoys(real, rng)
        features[split] = {kind: bowl_features(dem, f["lat"], f["lon"], f["diameter_km"], params)
                           .assign(diameter_km=f["diameter_km"].to_numpy()).dropna()
                           for kind, f in (("catalog", real), ("decoy", decoys))}

    rules, report = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        band = {split: {kind: f[f["diameter_km"].between(lo, hi, inclusive="left")] for kind, f in kinds.items()}
                for split, kinds in features.items()}
        rule = choose_rule(band["train"]["catalog"], band["train"]["decoy"], params)
        rules.append({"min_km": lo, "max_km": hi, **rule})
        for split, kinds in band.items():
            report.append({"split": split, "min_km": lo, "max_km": hi, "catalog_craters": len(kinds["catalog"]),
                           "decoys": len(kinds["decoy"]),
                           "catalog_pass": round(float(passes(kinds["catalog"], rule).mean()), 4),
                           "decoy_pass": round(float(passes(kinds["decoy"], rule).mean()), 4)})
    record = {"rules": rules, "report": report}
    out = cfg["paths"]["selftrain"] / "dem_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))
    print(pd.DataFrame(rules).to_string(index=False))
    print(pd.DataFrame(report).to_string(index=False))
    print(f"wrote {out}")
    return record


def load_rules(cfg):
    return json.loads((cfg["paths"]["selftrain"] / "dem_check.json").read_text())["rules"]


if __name__ == "__main__":
    from .config import load_config

    fit(load_config())
