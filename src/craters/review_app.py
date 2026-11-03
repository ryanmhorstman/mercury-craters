"""Local review app for Phase 4 (standard library HTTP server, one HTML page).

    python -m craters.phase4 review      # then open http://localhost:8765

Shows one item at a time as an image chip centred on the circle in question.
Verdicts are appended to adjudication/verdicts.csv as they're made.
"""
import json
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np
import pandas as pd

from .adjudicate import append_verdict, verdicts_for
from .tiling import Tile, render_tile

PAGE = Path(__file__).parent / "review" / "index.html"
CHIP_PX = 512
NATIVE_M = 166.31


def chip_pixel_m(diameter_km, view):
    """Close view: the crater spans about a third of the chip, never finer than 2x native.
    Wide view: three times the field."""
    close = max(diameter_km * 3000.0 / CHIP_PX, NATIVE_M / 2)
    return close * (3 if view == "wide" else 1)


class ReviewState:
    def __init__(self, items_path, verdicts_path, basemap, chip_cache):
        items = pd.read_csv(items_path)
        self.items = items[items["auto_verdict"].fillna("") == ""].reset_index(drop=True)
        self.by_id = self.items.set_index("item_id", drop=False)
        self.verdicts_path, self.basemap, self.chip_cache = verdicts_path, basemap, chip_cache
        self.lock = threading.Lock()

    def listing(self):
        verdicts, _ = verdicts_for(self.items, self.verdicts_path)
        rows = []
        for item in self.items.itertuples():
            rows.append({
                "id": item.item_id, "position": int(item.position), "lat": round(item.lat, 4), "lon": round(item.lon, 4),
                "diameter_km": round(item.diameter_km, 2), "verdict": verdicts.get(item.item_id, ""),
                "views": {v: {"radius_frac": item.diameter_km * 500.0 / chip_pixel_m(item.diameter_km, v) / CHIP_PX,
                              "field_km": round(chip_pixel_m(item.diameter_km, v) * CHIP_PX / 1000.0, 1)}
                          for v in ("close", "wide")},
            })
        return rows

    def reveal(self, item_id):
        item = self.by_id.loc[item_id]
        if item["kind"] == "catalog":
            return {"kind": "catalog", "text": f"Catalog crater, missed by: {item['detectors'].replace(';', ', ')}"}
        return {"kind": "detection", "text": f"Not in catalog; proposed by: {item['detectors'].replace(';', ', ')}"}

    def chip(self, item_id, view):
        # The file name includes the circle's position and size: a re-registered
        # catalog crater keeps its id but needs a freshly centred chip.
        item = self.by_id.loc[item_id]
        key = f"{item_id.replace(':', '_')}_{item['lat']:.4f}_{item['lon']:.4f}_{item['diameter_km']:.2f}"
        path = self.chip_cache / view / f"{key}.png"
        if not path.exists():
            tile = Tile(item_id, "chip", float(item["lat"]), float(item["lon"]), CHIP_PX,
                        chip_pixel_m(float(item["diameter_km"]), view))
            image = render_tile(tile, self.basemap)
            lo, hi = np.nanpercentile(image, [1, 99])
            image = np.clip((np.nan_to_num(image, nan=lo) - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), image)
        return path.read_bytes()

    def record(self, item_id, verdict):
        with self.lock:
            append_verdict(self.verdicts_path, self.by_id.loc[item_id], verdict)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, state, **kwargs):
        self.state = state
        super().__init__(*args, **kwargs)

    def log_message(self, *args):
        pass

    def send(self, body, content_type, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        cacheable = status == 200 and content_type == "image/png"      # chips never change; nothing else is cached
        self.send_header("Cache-Control", "max-age=86400" if cacheable else "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        try:
            if url.path in ("/", "/index.html"):
                self.send(PAGE.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/items":
                self.send(json.dumps(self.state.listing()).encode(), "application/json")
            elif url.path.startswith("/api/reveal/"):
                item_id = unquote(url.path.split("/", 3)[3])
                self.send(json.dumps(self.state.reveal(item_id)).encode(), "application/json")
            elif url.path.startswith("/chip/"):
                item_id = unquote(url.path.split("/", 2)[2])
                view = parse_qs(url.query).get("view", ["close"])[0]
                self.send(self.state.chip(item_id, "wide" if view == "wide" else "close"), "image/png")
            else:
                self.send(b"not found", "text/plain", 404)
        except KeyError:
            self.send(b"unknown item", "text/plain", 404)

    def do_POST(self):
        if urlparse(self.path).path != "/api/verdict":
            return self.send(b"not found", "text/plain", 404)
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        try:
            self.state.record(body["item_id"], body["verdict"])
        except (KeyError, ValueError) as error:
            return self.send(json.dumps({"error": str(error)}).encode(), "application/json", 400)
        self.send(b'{"ok": true}', "application/json")


def serve(state, port):
    server = ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, state=state))
    print(f"review app: http://localhost:{port}  ({len(state.items)} items; Ctrl+C to stop)", flush=True)
    server.serve_forever()
