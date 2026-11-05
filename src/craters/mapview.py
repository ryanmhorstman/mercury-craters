"""Zoomable map of the global CNN run against the thesis catalog (stdlib HTTP server, one HTML page).

    python -m craters.mapview          # then open http://localhost:8766

Serves the MDIS basemap as 256 px PNG tiles read on the fly from the COG (GDAL
picks the overview level), and the crater sets as one JSON document. Craters
are labelled matched / catalog only / CNN only by the same whole-planet
Hungarian match as compare_global; CNN-only craters that passed the DEM bowl
check (outputs/selftrain/new_craters.csv, see docs/self-training.md) get a
category of their own, and big craters found by the DEM scan alone
(outputs/selftrain/demscan_new.csv: in neither set) are two more layers: those
the CNN supports at a low score (`craters.demscan --cnn`) and the rest, off by default. The page re-derives categories when the
score cutoff is raised: a matched pair whose detection falls below the cutoff
becomes catalog only (an approximation: matching is not re-run at each cutoff).

Map coordinates: Leaflet's simple CRS with the whole planet (east longitude
0-360, latitude 90 to -90) as a 2 x 1 grid of tiles at zoom 0, so the tile at
(z, x, y) spans 180 / 2**z degrees in both directions.
"""
import json
import math
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from scipy.spatial import cKDTree

from .adjudicate import verdicts_for
from .compare_global import WHOLE_PLANET, load
from .config import load_config
from .geo import MERCURY_RADIUS_M, km_to_chord, unit_vectors
from .metrics import label_outcomes

PAGE = Path(__file__).parent / "map_page" / "index.html"


def tile_window(z, x, y, width, height):
    """Source-pixel window (col_off, row_off, cols, rows) of map tile (z, x, y), or None above or below the map.

    The raster runs from 180 W to 180 E (lon_0 = 0); the map runs from 0 to 360 E.
    Tile edges always fall on multiples of 180 / 2**z degrees, so no tile straddles
    the raster's 180-degree edge and each tile is one contiguous window.
    """
    span = 180.0 / 2**z
    if z < 0 or not 0 <= y < 2**z:
        return None
    x %= 2 ** (z + 1)                  # the map repeats east-west, so panning across 0/360 never ends
    lon0, lat_top = x * span, 90.0 - y * span
    col_off = ((lon0 + 180.0) % 360.0) / 360.0 * width
    row_off = (90.0 - lat_top) / 180.0 * height
    return col_off, row_off, span / 360.0 * width, span / 180.0 * height


class Basemap:
    """Thread-safe tile reader: one rasterio handle per server thread."""

    def __init__(self, path, tile_px, stretch_percentiles):
        self.path, self.tile_px = path, tile_px
        self.local = threading.local()
        with rasterio.open(path) as src:
            self.width, self.height = src.width, src.height
            # Contrast from the coarsest overview: the same stretch at every zoom, so tiles don't flicker.
            small = src.read(1, out_shape=(src.height // 256, src.width // 256))
        self.lo, self.hi = np.percentile(small[small > 0], stretch_percentiles)
        self.native_zoom = math.ceil(math.log2(self.width / (2 * tile_px)))

    def dataset(self):
        if not hasattr(self.local, "src"):
            self.local.src = rasterio.open(self.path)
        return self.local.src

    def tile_png(self, z, x, y):
        window = tile_window(z, x, y, self.width, self.height)
        if window is None:
            return None
        image = self.dataset().read(1, window=Window(*window), out_shape=(self.tile_px, self.tile_px),
                                    resampling=Resampling.average)
        stretched = np.clip((image.astype(np.float32) - self.lo) / (self.hi - self.lo) * 255, 0, 255)
        stretched[image == 0] = 0                       # nodata stays black
        return cv2.imencode(".png", stretched.astype(np.uint8))[1].tobytes()


# ---------------------------------------------------------------- crater sets

def detection_verdicts(items, verdicts, detections, max_km):
    """Review verdicts on detection items -> index into `detections` -> verdict (nearest within max_km)."""
    reviewed = items[(items["kind"] == "detection") & items["item_id"].isin(verdicts)]
    if reviewed.empty:
        return {}
    tree = cKDTree(unit_vectors(detections["lat"], detections["lon"]))
    dist, idx = tree.query(unit_vectors(reviewed["lat"], reviewed["lon"]), distance_upper_bound=km_to_chord(max_km))
    return {int(i): verdicts[item] for item, i, d in zip(reviewed["item_id"], idx, dist) if np.isfinite(d)}


def dem_confirmed(detections, path):
    """True for each detection listed in new_craters.csv (same calibrated run, so same positions)."""
    flag = np.zeros(len(detections), dtype=bool)
    if not path.exists() or detections.empty:
        return flag
    confirmed = pd.read_csv(path)
    tree = cKDTree(unit_vectors(detections["lat"], detections["lon"]))
    dist, idx = tree.query(unit_vectors(confirmed["lat"], confirmed["lon"]), distance_upper_bound=km_to_chord(0.05))
    flag[idx[np.isfinite(dist)]] = True
    return flag


def scan_set(path):
    """DEM-scan craters in neither the catalog nor the CNN run (craters.demscan), or an empty set."""
    scan = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["lat", "lon", "diameter_km", "depth_ratio"])
    if "cnn_score" not in scan:
        scan = scan.assign(cnn_score=0.0, cnn_ok=False)
    return {"lat": scan["lat"].round(4).tolist(), "lon": scan["lon"].round(4).tolist(),
            "d": scan["diameter_km"].round(1).tolist(), "depth": scan["depth_ratio"].round(3).tolist(),
            "cnn_score": scan["cnn_score"].round(3).tolist(), "cnn_ok": scan["cnn_ok"].astype(bool).tolist(),
            "verdict": [""] * len(scan)}       # never reviewed; the page draws verdict marks for every set


def crater_sets(cfg):
    """Catalog and detections >= the minimum diameter, each with its match partner, any review verdict
    and (detections) whether the DEM check confirmed it as a crater missing from the catalog."""
    name, catalog, detections = load(cfg, calibrated=cfg["mapview"]["calibrated"])
    min_d = cfg["evaluation"]["diameter_bins_km"][0]
    catalog = catalog[catalog["diameter_km"] >= min_d]
    detections = detections[detections["diameter_km"] >= min_d]
    cat, det = label_outcomes(catalog, detections, [WHOLE_PLANET], cfg["match"], min_d)

    cat_verdict, det_verdict = [""] * len(cat), [""] * len(det)
    if cfg["paths"]["review_items"].exists():
        items = pd.read_csv(cfg["paths"]["review_items"])
        verdicts, _ = verdicts_for(items, cfg["paths"]["verdicts"])
        cat_verdict = [verdicts.get(f"cat:{i}", "") for i in cat["id"]]
        for i, verdict in detection_verdicts(items, verdicts, det, cfg["mapview"]["verdict_match_km"]).items():
            det_verdict[i] = verdict

    def columns(frame, extra):
        return {"lat": frame["lat"].round(4).tolist(), "lon": frame["lon"].round(4).tolist(),
                "d": frame["diameter_km"].round(2).tolist(), **extra}

    return {
        "model": name, "min_diameter_km": min_d, "radius_km": MERCURY_RADIUS_M / 1000.0,
        "catalog": columns(cat, {"id": cat["id"].tolist(), "name": cat["name"].fillna("").tolist(),
                                 "partner": cat["matched_det"].tolist(), "verdict": cat_verdict}),
        "cnn": columns(det, {"score": det["score"].round(3).tolist(), "level": det["level"].tolist(),
                             "partner": det["matched_cat"].tolist(), "verdict": det_verdict,
                             "dem": dem_confirmed(det, cfg["paths"]["selftrain"] / "new_craters.csv").tolist()}),
        "scan": scan_set(cfg["paths"]["selftrain"] / "demscan_new.csv"),
    }


def page_config(cfg, basemap):
    view = cfg["mapview"]
    regions = {**cfg["regions"]["test"], **cfg["regions"]["validation"]}
    return {"tile_px": view["tile_px"], "native_zoom": basemap.native_zoom, "max_zoom": view["max_zoom"],
            "min_radius_px": view["min_radius_px"], "score_slider": view["score_slider"], "colors": view["colors"],
            "regions": regions, "train_exclude": cfg["regions"]["train"]["exclude"],
            "train_max_abs_lat": cfg["regions"]["train"]["max_abs_lat"]}


# ---------------------------------------------------------------- server

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, basemap, documents, **kwargs):
        self.basemap, self.documents = basemap, documents
        super().__init__(*args, **kwargs)

    def log_message(self, *args):
        pass

    def send(self, body, content_type, status=200, cache=False):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "max-age=86400" if cache else "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self.send(PAGE.read_bytes(), "text/html; charset=utf-8")
        if path in self.documents:
            return self.send(self.documents[path], "application/json")
        parts = path.removesuffix(".png").split("/")
        if len(parts) == 5 and parts[1] == "tile" and all(p.lstrip("-").isdigit() for p in parts[2:]):
            png = self.basemap.tile_png(*map(int, parts[2:]))
            if png is not None:
                return self.send(png, "image/png", cache=True)
        self.send(b"not found", "text/plain", 404)


def main():
    cfg = load_config()
    started = time.time()
    basemap = Basemap(cfg["paths"]["basemap"], cfg["mapview"]["tile_px"], cfg["mapview"]["stretch_percentiles"])
    craters = crater_sets(cfg)
    documents = {"/api/config": json.dumps(page_config(cfg, basemap)).encode(),
                 "/api/craters": json.dumps(craters).encode()}
    port = cfg["mapview"]["port"]
    server = ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, basemap=basemap, documents=documents))
    print(f"map viewer: http://localhost:{port}  ({len(craters['catalog']['d']):,} catalog, "
          f"{len(craters['cnn']['d']):,} CNN craters; ready in {time.time() - started:.0f} s; Ctrl+C to stop)",
          flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
