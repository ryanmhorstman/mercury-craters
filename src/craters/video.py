"""A short vertical video (1080 x 1920, for Instagram) flying over the mosaic with both crater sets.

    python -m craters.video                      # writes outputs/video/mercury_craters.mp4
    python -m craters.video --stills 3 12 30     # PNG frames at those seconds, to check the storyboard

The camera path, crater layers and captions are one storyboard below (times in
seconds). Frames are rendered straight from the basemap COG: wide views sample a
whole-planet image held in memory, close views read a window of the COG, so the
zoom runs from the whole planet down to native resolution. ffmpeg encodes H.264.
"""
import argparse
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
from rasterio.enums import Resampling
from rasterio.windows import Window

from .config import load_config
from .mapview import crater_sets

W, H, FPS = 1080, 1920, 30
GLOBE_W = 16384                     # whole-planet image in memory: 45.5 px per degree
FONT_DIR = Path("C:/Windows/Fonts")
BOLD, REGULAR = FONT_DIR / "segoeuib.ttf", FONT_DIR / "segoeui.ttf"

# Camera keyframes: (time, centre lon E, centre lat, view width in degrees).
T1_CENTRE, SMALL_CRATERS = (285.0, 15.0), (274.5, 4.5)
CAMERA = [
    (0.0, 180.0, 0.0, 380.0),
    (4.0, 180.0, 0.0, 330.0),
    (10.0, *T1_CENTRE, 36.0),
    (20.0, *T1_CENTRE, 27.0),
    (26.0, *T1_CENTRE, 26.0),
    (33.0, *SMALL_CRATERS, 4.2),
    (39.0, *SMALL_CRATERS, 3.6),
    (46.0, 180.0, 0.0, 380.0),
    (48.0, 180.0, 0.0, 380.0),
]
DURATION = CAMERA[-1][0]

# Layer opacity / recolouring ramps: (start, end) of each fade.
CATALOG_IN, CNN_IN, RECOLOUR = (10.5, 12.0), (15.5, 17.0), (20.5, 22.0)

CAPTIONS = [   # (start, end, headline, body)
    (0.3, 4.3, "This is Mercury",
     "The whole planet, photographed by NASA's MESSENGER spacecraft and stitched into one map."),
    (4.8, 9.8, "Let's zoom in",
     "Each pixel is 166 m across. The full map is 92,000 × 46,000 pixels."),
    (10.3, 15.0, "A year of hand mapping",
     "For my master's thesis I mapped 20,391 craters (5 km and larger) by hand."),
    (15.3, 20.0, "Then I trained a neural network",
     "It scanned the whole planet in 7 minutes."),
    (20.3, 26.0, "Who found what",
     "Yellow: we both found it. Blue: only I did. Orange: only the network did."),
    (26.3, 33.0, "Small craters are where we differ",
     "On large craters we mostly agree. Below 10 km, the network finds many more."),
    (33.3, 39.0, "Why?",
     "My thesis only counted craters with a clearly defined rim. The network also picks up old, worn-down ones."),
    (39.5, 48.0, "Same planet, two ways of seeing it",
     "github.com/ryanmhorstman/mercury-craters"),
]
CREDIT = "Basemap: MESSENGER MDIS global mosaic (NASA / JHUAPL / Carnegie, USGS)"


def smoothstep(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def ramp(t, span):
    return smoothstep((t - span[0]) / (span[1] - span[0]))


def camera(t):
    """(lon, lat, pixels per degree) at time t. Zooms are logarithmic, and the centre moves in
    step with the view width, so the point being zoomed towards stays put on screen."""
    for (t0, lon0, lat0, w0), (t1, lon1, lat1, w1) in zip(CAMERA, CAMERA[1:]):
        if t <= t1:
            break
    s = smoothstep((t - t0) / (t1 - t0)) if t1 > t0 else 1.0
    width = math.exp(math.log(w0) + (math.log(w1) - math.log(w0)) * s)
    g = (w0 - width) / (w0 - w1) if abs(w0 - w1) > 1e-6 else s
    return lon0 + (lon1 - lon0) * g, lat0 + (lat1 - lat0) * g, W / width


class Planet:
    """Greyscale views of the basemap for any centre and scale."""

    def __init__(self, path, stretch_percentiles=(0.5, 99.5)):
        self.src = rasterio.open(path)
        globe = self.src.read(1, out_shape=(GLOBE_W // 2, GLOBE_W), resampling=Resampling.average)
        globe = np.roll(globe, GLOBE_W // 2, axis=1)       # raster starts at 180 W; column 0 is now 0 E
        lo, hi = np.percentile(globe[globe > 0], stretch_percentiles)
        self.lut = np.clip((np.arange(256) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
        self.lut[0] = 0
        self.pyramid = [self.lut[globe]]                    # halvings, so wide views aren't aliased
        while self.pyramid[-1].shape[1] > 1024:
            self.pyramid.append(cv2.pyrDown(self.pyramid[-1]))
        self.native_ppd = self.src.width / 360.0

    def view(self, lon, lat, ppd):
        globe_ppd = GLOBE_W / 360.0
        if ppd <= globe_ppd:
            return self._from_globe(lon, lat, ppd)
        return self._from_cog(lon, lat, ppd)

    def _from_globe(self, lon, lat, ppd):
        level = min(int(math.log2(GLOBE_W / 360.0 / ppd)), len(self.pyramid) - 1)
        image = self.pyramid[level]
        level_ppd = image.shape[1] / 360.0
        u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        map_x = ((lon + (u - W / 2) / ppd) * level_ppd) % image.shape[1]
        map_y = (90.0 - lat + (v - H / 2) / ppd) * level_ppd
        return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def _from_cog(self, lon, lat, ppd):
        """Close views: a window of the COG. They never reach the raster's 180-degree edge."""
        src_per_out = self.native_ppd / ppd
        col_c = ((lon + 180.0) % 360.0) * self.native_ppd
        row_c = (90.0 - lat) * self.native_ppd
        c0, c1 = math.floor(col_c - W / 2 * src_per_out) - 2, math.ceil(col_c + W / 2 * src_per_out) + 2
        r0 = max(math.floor(row_c - H / 2 * src_per_out) - 2, 0)
        r1 = min(math.ceil(row_c + H / 2 * src_per_out) + 2, self.src.height)
        scale = min(1.0, 2.0 / src_per_out)                 # read at <= 2x the output resolution
        shape = (max(round((r1 - r0) * scale), 1), max(round((c1 - c0) * scale), 1))
        block = self.lut[self.src.read(1, window=Window(c0, r0, c1 - c0, r1 - r0), out_shape=shape,
                                       resampling=Resampling.average)]
        sx, sy = shape[1] / (c1 - c0), shape[0] / (r1 - r0)
        # output pixel -> block pixel
        m = np.array([[src_per_out * sx, 0, (col_c - W / 2 * src_per_out - c0) * sx],
                      [0, src_per_out * sy, (row_c - H / 2 * src_per_out - r0) * sy]], dtype=np.float64)
        return cv2.warpAffine(block, m, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# ---------------------------------------------------------------- craters

def hex_rgb(color):
    return np.array([int(color[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.float32)


class Craters:
    def __init__(self, cfg):
        sets = crater_sets(cfg)
        colors = {k: hex_rgb(v) for k, v in cfg["mapview"]["colors"].items()}
        km_per_deg = sets["radius_km"] * math.pi / 180.0
        self.layers = {}
        for name, start, matched in (("catalog", colors["catalog_only"], colors["matched"]),
                                     ("cnn", colors["cnn_only"], colors["matched"])):
            s = sets[name]
            lat, partner = np.array(s["lat"]), np.array(s["partner"])
            r_lat = np.array(s["d"]) / 2 / km_per_deg
            final = np.where((partner >= 0)[:, None], matched, colors["cnn_only" if name == "cnn" else "catalog_only"])
            self.layers[name] = dict(lon=np.array(s["lon"]), lat=lat, r_lat=r_lat,
                                     r_lon=r_lat / np.maximum(np.cos(np.radians(lat)), 0.01),
                                     start=np.broadcast_to(start, final.shape), final=final,
                                     thickness=np.where(partner >= 0, 1, 2) if name == "cnn" else np.full(len(lat), 2))

    def draw(self, frame, name, lon_c, lat_c, ppd, alpha, recolour):
        if alpha <= 0:
            return frame
        layer = self.layers[name]
        dlon = (layer["lon"] - lon_c + 180.0) % 360.0 - 180.0
        x, y = W / 2 + dlon * ppd, H / 2 + (lat_c - layer["lat"]) * ppd
        rx, ry = layer["r_lon"] * ppd, layer["r_lat"] * ppd
        keep = (ry >= 1.5) & (x + rx > 0) & (x - rx < W) & (y + ry > 0) & (y - ry < H)
        overlay = frame.copy()
        color = layer["start"] + (layer["final"] - layer["start"]) * recolour
        shift = 4                                           # sub-pixel positions for smooth motion
        for i in np.flatnonzero(keep):
            thickness = int(layer["thickness"][i]) if recolour > 0.5 else 2
            cv2.ellipse(overlay, (int(x[i] * 16), int(y[i] * 16)), (int(rx[i] * 16), int(ry[i] * 16)), 0, 0, 360,
                        color[i].tolist(), thickness + (ry[i] > 40), cv2.LINE_AA, shift)
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


# ---------------------------------------------------------------- text

def wrap(draw, text, font, width):
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=font) <= width or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    return lines + [line]


class Overlay:
    def __init__(self, colors):
        self.head, self.body = ImageFont.truetype(str(BOLD), 62), ImageFont.truetype(str(REGULAR), 42)
        self.small, self.legend = ImageFont.truetype(str(REGULAR), 26), ImageFont.truetype(str(BOLD), 34)
        self.colors = colors

    def caption(self, image, t):
        draw = ImageDraw.Draw(image, "RGBA")
        for start, end, head, body in CAPTIONS:
            a = min(ramp(t, (start, start + 0.4)), 1 - ramp(t, (end - 0.4, end)))
            if a <= 0:
                continue
            heads, lines = wrap(draw, head, self.head, 900), wrap(draw, body, self.body, 900)
            body_top = 30 + 80 * len(heads) + 6
            top = 1330                                      # clear of Instagram's caption area at the bottom
            draw.rounded_rectangle([50, top, W - 50, top + body_top + 56 * len(lines) + 40], 28,
                                   fill=(10, 12, 14, int(190 * a)))
            for k, line in enumerate(heads):
                draw.text((90, top + 30 + 80 * k), line, font=self.head, fill=(255, 255, 255, int(255 * a)))
            for k, line in enumerate(lines):
                draw.text((90, top + body_top + 56 * k), line, font=self.body, fill=(225, 225, 225, int(255 * a)))

    def legend_and_scale(self, image, t, ppd, km_per_deg):
        draw = ImageDraw.Draw(image, "RGBA")
        a = ramp(t, RECOLOUR) * (1 - ramp(t, (45.0, 46.0)))
        if a > 0:
            x = 70
            for key, label in (("matched", "both"), ("catalog_only", "thesis only"), ("cnn_only", "network only")):
                rgb = tuple(int(c) for c in hex_rgb(self.colors[key]))
                draw.ellipse([x, 262, x + 34, 296], outline=rgb + (int(255 * a),), width=5)
                draw.text((x + 46, 256), label, font=self.legend, fill=(255, 255, 255, int(255 * a)),
                          stroke_width=3, stroke_fill=(0, 0, 0, int(200 * a)))
                x += 70 + draw.textlength(label, font=self.legend)
        # scale bar, a round length 150-400 px long
        km_per_px = km_per_deg / ppd
        target = 300 * km_per_px
        length = min((m * 10**e for e in range(0, 5) for m in (1, 2, 5) if m * 10**e >= target / 2),
                     key=lambda v: abs(v - target))
        px = length / km_per_px
        draw.rectangle([70, 1250, 70 + px, 1258], fill=(255, 255, 255, 230))
        draw.text((70, 1200), f"{length:,} km", font=self.small, fill=(255, 255, 255, 230),
                  stroke_width=2, stroke_fill=(0, 0, 0, 200))
        if t < 10:
            draw.text((70, 200), CREDIT, font=self.small, fill=(200, 200, 200, int(220 * (1 - ramp(t, (9, 10))))))


# ---------------------------------------------------------------- render

class Renderer:
    def __init__(self, cfg):
        self.planet = Planet(cfg["paths"]["basemap"])
        self.craters = Craters(cfg)
        self.overlay = Overlay(cfg["mapview"]["colors"])
        self.km_per_deg = cfg["body"]["radius_m"] / 1000.0 * math.pi / 180.0

    def frame(self, t):
        lon, lat, ppd = camera(t)
        image = cv2.cvtColor(self.planet.view(lon, lat, ppd), cv2.COLOR_GRAY2RGB)
        recolour = ramp(t, RECOLOUR)
        image = self.craters.draw(image, "catalog", lon, lat, ppd, ramp(t, CATALOG_IN), recolour)
        image = self.craters.draw(image, "cnn", lon, lat, ppd, 0.9 * ramp(t, CNN_IN), recolour)
        pil = Image.fromarray(image)
        self.overlay.legend_and_scale(pil, t, ppd, self.km_per_deg)
        self.overlay.caption(pil, t)
        return np.asarray(pil)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stills", type=float, nargs="*", help="write PNG frames at these times instead")
    args = parser.parse_args()
    cfg = load_config()
    out = cfg["paths"]["video"]
    out.mkdir(parents=True, exist_ok=True)
    renderer = Renderer(cfg)
    if args.stills:
        for t in args.stills:
            cv2.imwrite(str(out / f"still_{t:05.1f}s.png"), cv2.cvtColor(renderer.frame(t), cv2.COLOR_RGB2BGR))
        print(f"stills in {out}")
        return
    path = out / "mercury_craters.mp4"
    ffmpeg = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                               "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow",
                               "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
                              stdin=subprocess.PIPE)
    n = int(DURATION * FPS)
    for k in range(n):
        ffmpeg.stdin.write(renderer.frame(k / FPS).tobytes())
        if k % (5 * FPS) == 0:
            print(f"{k / FPS:4.0f} s of {DURATION:.0f}", flush=True)
    ffmpeg.stdin.close()
    ffmpeg.wait()
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
